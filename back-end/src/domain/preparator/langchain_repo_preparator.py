from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List
from typing import Optional
from urllib.parse import urlparse
from urllib.parse import urlunparse

import requests
from fastapi import HTTPException
from langchain.schema import Document
from shared.logging import get_logger
from shared.settings.advanced_configs import DEFAULT_EXCLUDED_DIRS
from shared.settings.advanced_configs import DEFAULT_EXCLUDED_FILES
from shared.utils import extract_full_repo_name

logger = get_logger(__name__)


class LangChainRepoPreparator:
    """
    LangChain-based repository preparator optimized for GraphRAG.

    This class handles repository download and basic file processing
    without embedding computations, designed for pure graph-based systems.
    """

    def __init__(self):
        self.repo_url = None
        self.repo_paths = None
        self.documents = []

    def prepare(self, repo_url: str, access_token: Optional[str] = None) -> List[Document]:
        """
        Prepare repository for GraphRAG pipeline.

        Args:
            repo_url: Repository URL to download
            access_token: Optional access token for private repositories

        Returns:
            List[Document]: Processed documents ready for graph construction
        """
        logger.info(f'Preparing repository for GraphRAG: {repo_url}')

        # Reset state
        self.repo_url = repo_url
        self.repo_paths = None
        self.documents = []

        # Download repository
        self._download_repository(repo_url, access_token)

        # Read and process files
        documents = self._read_repository_files()

        # Store documents
        self.documents = documents

        logger.info(f'Repository preparation complete: {len(documents)} documents processed')
        return documents

    def _download_repository(self, repo_url: str, access_token: Optional[str] = None) -> None:
        """Download repository to local path."""
        try:
            # Create local paths
            repo_name = extract_full_repo_name(repo_url)
            base_dir = Path.home() / '.autodocs' / 'repos'
            repo_dir = base_dir / repo_name

            # Store paths
            self.repo_paths = {
                'repo_dir': str(repo_dir),
                'repo_name': repo_name,
            }

            # Check if already exists
            if repo_dir.exists() and any(repo_dir.iterdir()):
                logger.info(f'Repository already exists at {repo_dir}')
                return

            # Check accessibility
            access_status = self._check_repository_accessibility(repo_url, access_token)
            if access_status != 200:
                self._handle_access_error(access_status)

            # Clone repository
            self._clone_repository(repo_url, repo_dir, access_token)

        except Exception as e:
            logger.error(f'Failed to download repository: {e}')
            raise

    def _clone_repository(self, repo_url: str, local_path: Path, access_token: Optional[str] = None) -> None:
        """Clone git repository."""
        try:
            # Check git availability
            subprocess.run(['git', '--version'], check=True, capture_output=True)

            # Create directory
            local_path.mkdir(parents=True, exist_ok=True)

            # Prepare clone URL
            clone_url = repo_url
            if access_token:
                parsed = urlparse(repo_url)
                clone_url = urlunparse((
                    parsed.scheme,
                    f'{access_token}@{parsed.netloc}',
                    parsed.path, '', '', '',
                ))

            # Clone
            logger.info(f'Cloning repository to {local_path}...')
            subprocess.run(
                [
                    'git', 'clone', clone_url, str(local_path),
                ], check=True, capture_output=True,
            )

            logger.info('Repository cloned successfully')

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)

            # Sanitize token from error message
            if access_token and access_token in error_msg:
                error_msg = error_msg.replace(access_token, '***TOKEN***')

            if self._is_auth_error(error_msg):
                raise HTTPException(
                    status_code=401,
                    detail='Authentication failed. Please check your access token.',
                )
            elif self._is_not_found_error(error_msg):
                raise HTTPException(
                    status_code=404,
                    detail='Repository not found. Please check the URL.',
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f'Git clone failed: {error_msg}',
                )

    def _read_repository_files(self) -> List[Document]:
        """Read and process repository files into LangChain documents."""
        if not self.repo_paths:
            raise ValueError('Repository not downloaded')

        repo_dir = Path(self.repo_paths['repo_dir'])
        documents = []

        # File extensions to process
        code_extensions = [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.jsx', '.tsx', '.html', '.css', '.php',
            '.swift', '.cs', '.rb', '.scala', '.kt',
        ]

        doc_extensions = [
            '.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.toml', '.ini',
        ]

        all_extensions = code_extensions + doc_extensions

        # Process files
        for ext in all_extensions:
            pattern = f'**/*{ext}'
            for file_path in repo_dir.glob(pattern):
                if self._should_process_file(file_path):
                    doc = self._create_document_from_file(file_path)
                    if doc:
                        documents.append(doc)

        logger.info(f'Processed {len(documents)} files from repository')
        return documents

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on exclusion rules."""
        # Convert to relative path for checking
        repo_dir = Path(self.repo_paths['repo_dir'])
        try:
            rel_path = file_path.relative_to(repo_dir)
        except ValueError:
            return False

        path_parts = rel_path.parts
        file_name = file_path.name

        # Check excluded directories
        for excluded_dir in DEFAULT_EXCLUDED_DIRS:
            clean_excluded = excluded_dir.strip('./').rstrip('/')
            if clean_excluded in path_parts:
                return False

        # Check excluded files
        for excluded_pattern in DEFAULT_EXCLUDED_FILES:
            if re.match(excluded_pattern.replace('*', '.*'), file_name):
                return False

        return True

    def _create_document_from_file(self, file_path: Path) -> Optional[Document]:
        """Create LangChain document from file."""
        try:
            # Read file content
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                return None

            # Create relative path
            repo_dir = Path(self.repo_paths['repo_dir'])
            rel_path = file_path.relative_to(repo_dir)

            # Determine file type
            file_type = self._determine_file_type(file_path)

            # Create metadata
            metadata = {
                'source': str(file_path),
                'file_path': str(rel_path),
                'file_name': file_path.name,
                'file_type': file_type,
                'file_extension': file_path.suffix,
                'repo_url': self.repo_url,
                'repo_name': self.repo_paths['repo_name'],
                'lines_of_code': len(content.splitlines()),
                'char_count': len(content),
            }

            return Document(page_content=content, metadata=metadata)

        except Exception as e:
            logger.warning(f'Failed to process file {file_path}: {e}')
            return None

    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type from extension."""
        ext = file_path.suffix.lower()

        code_types = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'react',
            '.tsx': 'react_typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'header',
            '.hpp': 'cpp_header',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
        }

        doc_types = {
            '.md': 'markdown',
            '.txt': 'text',
            '.rst': 'restructuredtext',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
        }

        web_types = {
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
        }

        # Return specific type
        if ext in code_types:
            return code_types[ext]
        elif ext in doc_types:
            return doc_types[ext]
        elif ext in web_types:
            return web_types[ext]
        else:
            return 'unknown'

    def _check_repository_accessibility(self, repo_url: str, access_token: Optional[str] = None) -> int:
        """Check if repository is accessible."""
        try:
            parsed_url = urlparse(repo_url)

            if 'github.com' in parsed_url.netloc:
                return self._check_github_accessibility(repo_url, access_token)
            elif 'gitlab.com' in parsed_url.netloc:
                return self._check_gitlab_accessibility(repo_url, access_token)
            else:
                logger.warning(f'Unknown git hosting service: {parsed_url.netloc}')
                return 200  # Assume accessible

        except Exception as e:
            logger.error(f'Error checking repository accessibility: {e}')
            return 500

    def _check_github_accessibility(self, repo_url: str, access_token: Optional[str] = None) -> int:
        """Check GitHub repository accessibility."""
        try:
            # Extract owner and repo from URL
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 2:
                return 404

            owner, repo = path_parts[0], path_parts[1]
            if repo.endswith('.git'):
                repo = repo[:-4]

            # Check via GitHub API
            api_url = f'https://api.github.com/repos/{owner}/{repo}'
            headers = {}
            if access_token:
                headers['Authorization'] = f'token {access_token}'

            response = requests.get(api_url, headers=headers, timeout=10)
            return response.status_code

        except Exception:
            return 500

    def _check_gitlab_accessibility(self, repo_url: str, access_token: Optional[str] = None) -> int:
        """Check GitLab repository accessibility."""
        try:
            # Similar implementation for GitLab
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 2:
                return 404

            # Use GitLab API
            project_path = '/'.join(path_parts[:2])
            api_url = f'https://gitlab.com/api/v4/projects/{project_path.replace("/", "%2F")}'
            headers = {}
            if access_token:
                headers['Authorization'] = f'Bearer {access_token}'

            response = requests.get(api_url, headers=headers, timeout=10)
            return response.status_code

        except Exception:
            return 500

    def _handle_access_error(self, status_code: int) -> None:
        """Handle repository access errors."""
        if status_code == 404:
            raise HTTPException(
                status_code=404,
                detail='Repository not found. Please check the URL.',
            )
        elif status_code == 401:
            raise HTTPException(
                status_code=401,
                detail='Repository requires authentication. Please provide a valid access token.',
            )
        else:
            raise HTTPException(
                status_code=400,
                detail='Unable to access repository. Please check the URL and credentials.',
            )

    def _is_auth_error(self, error_msg: str) -> bool:
        """Check if error indicates authentication issues."""
        auth_patterns = [
            r'authentication failed',
            r'invalid username or password',
            r'access denied',
            r'permission denied',
            r'terminal prompts disabled',
            r'authentication required',
        ]

        error_lower = error_msg.lower()
        return any(re.search(pattern, error_lower) for pattern in auth_patterns)

    def _is_not_found_error(self, error_msg: str) -> bool:
        """Check if error indicates repository not found."""
        not_found_patterns = [
            r'repository not found',
            r'could not read from remote repository',
            r'does not exist',
            r'remote repository not found',
        ]

        error_lower = error_msg.lower()
        return any(re.search(pattern, error_lower) for pattern in not_found_patterns)

    def get_documents(self) -> List[Document]:
        """Get processed documents."""
        return self.documents

    def get_repo_path(self) -> Optional[str]:
        """Get local repository path."""
        return self.repo_paths['repo_dir'] if self.repo_paths else None

    def get_repo_name(self) -> Optional[str]:
        """Get repository name."""
        return self.repo_paths['repo_name'] if self.repo_paths else None
