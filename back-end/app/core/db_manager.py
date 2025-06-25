import logging
import os
import subprocess

from fastapi import HTTPException
import requests
import re
import glob
import tiktoken
import adalflow as adal
from adalflow.components.data_process import TextSplitter
from adalflow.core.types import Document, List
from adalflow.core.db import LocalDB
from adalflow.utils import get_adalflow_default_root_path
from app.core.logging import setup_logging
from urllib.parse import urlparse, urlunparse
from app.core.config import *
from app.core.llm.ollama_processor import *

setup_logging()
logger = logging.getLogger(__name__)

def count_tokens(text: str) -> int:
    """
    Count the number of tokens.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        return len(text) // 4


class DBManager:
    def __init__(self):
        self.database = None
        self.repo_url = None
        self.repo_paths = None

    
    def prepare_db(self, repo_url: str, access_token: str = None) -> List[Document]:
        """
        Prepare the database with the given repository URL and access token.
        """
        self.database = None
        self.repo_url = None
        self.repo_paths = None
        
        self._download_repo_into_db(repo_url, access_token)
        return self._prepare_index_db()
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """
        Extracts the repository name from the given URL.
        """
        url_parts = repo_url.rstrip('/').split('/')

        # GitHub URL format: https://github.com/owner/repo
        if  len(url_parts) >= 5:
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        return repo_name
    
    def _download_repo(self, repo_url: str, local_path: str, access_token: str = None) -> str:
        """
        Downloads a Git repository to the specified local path.
        """
        try:
            logger.info(f"Check git version...")
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if os.path.exists(local_path) and os.listdir(local_path):
                logger.info(f"Repository already exists at {local_path}. Skipping download.")
                return local_path
            
            access_status = self._check_repo_accessibility(repo_url, access_token)
            
            if access_status == 404:
                raise HTTPException(
                    status_code=404, 
                    detail="Please check the valid repository URL or provide a valid access token."
                )
            elif access_status == 401:
                raise HTTPException(
                    status_code=401, 
                    detail="Repository requires authentication. Please provide a valid access token."
                )
            elif access_status != 200:
                raise HTTPException(
                    status_code=400, 
                    detail="aUnable to access repository. Please check the URL and credentials."
                )
                         
            os.makedirs(local_path, exist_ok=True)
            clone_url = repo_url
            if access_token:
                parse_url = urlparse(repo_url)
                clone_url = urlunparse((parse_url.scheme, f"{access_token}@{parse_url.netloc}", parse_url.path, '', '', ''))
                
            logger.info(f"Starting clone of repository {repo_url} to {local_path}...")
            
            result = subprocess.run(
                ["git", "clone", clone_url, local_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Repository cloned successfully to {local_path}.")
            return local_path
        
        except HTTPException:
            raise
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8')
            # Sanitize error message to remove any tokens
            if access_token and access_token in error_msg:
                error_msg = error_msg.replace(access_token, "***TOKEN***")
                
            # Check for authentication-related errors
            if self._is_auth_error(error_msg):
                if access_token:
                    raise HTTPException(
                        status_code=403, 
                        detail="Access denied to repository. Invalid or insufficient permissions for access token."
                    )
                else:
                    raise HTTPException(
                        status_code=401, 
                        detail="Repository requires authentication. Please provide a valid access token."
                    )
                    
            # Check for not found errors
            if self._is_not_found_error(error_msg):
                raise HTTPException(
                    status_code=404, 
                    detail="Repository not found. Please check the repository URL."
                )
                
            raise HTTPException(
                status_code=400,
                detail=f"Error during cloning: {error_msg}"
            )
        
    def _read_all(self, path: str):
        """
        Recursively reads all documents in a directory and its subdirectories.
        """
        documents = []
        # File extensions to look for, prioritizing code files
        code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                        ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
        doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

        # Convert back to lists for compatibility
        excluded_dirs = list(set(DEFAULT_EXCLUDED_DIRS))
        excluded_files = list(set(DEFAULT_EXCLUDED_FILES))
        
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

        logger.info(f"Reading documents from {path}")

        def should_process_file(file_path: str, excluded_dirs: List[str], excluded_files: List[str]) -> bool:
            file_path_parts = os.path.normpath(file_path).split(os.sep)
            file_name = os.path.basename(file_path)

            # Exclusion mode: file must not be in excluded directories or match excluded files
            is_excluded = False

            # Check if file is in an excluded directory
            for excluded in excluded_dirs:
                clean_excluded = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            # Check if file matches excluded file patterns
            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

        # Process code files first
        for ext in code_extensions:
            files = glob.glob(f"{path}/**/*{ext}", recursive=True)
            for file_path in files:
                # Check if file should be processed based on inclusion/exclusion rules
                if not should_process_file(file_path, excluded_dirs, excluded_files):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, path)

                        # Determine if this is an implementation file
                        is_implementation = (
                            not relative_path.startswith("test_")
                            and not relative_path.startswith("app_")
                            and "test" not in relative_path.lower()
                        )

                        # Check token count
                        token_count = count_tokens(content)
                        if token_count > MAX_EMBEDDING_TOKENS * 10:
                            logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                            continue

                        doc = Document(
                            text=content,
                            meta_data={
                                "file_path": relative_path,
                                "type": ext[1:],
                                "is_code": True,
                                "is_implementation": is_implementation,
                                "title": relative_path,
                                "token_count": token_count,
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        # Then process documentation files
        for ext in doc_extensions:
            files = glob.glob(f"{path}/**/*{ext}", recursive=True)
            for file_path in files:
                # Check if file should be processed based on inclusion/exclusion rules
                if not should_process_file(file_path, excluded_dirs, excluded_files):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, path)

                        # Check token count
                        token_count = count_tokens(content)
                        if token_count > MAX_EMBEDDING_TOKENS:
                            logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                            continue

                        doc = Document(
                            text=content,
                            meta_data={
                                "file_path": relative_path,
                                "type": ext[1:],
                                "is_code": False,
                                "is_implementation": False,
                                "title": relative_path,
                                "token_count": token_count,
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        logger.info(f"Found {len(documents)} documents")
        return documents
    
    def _prepare_data_pipeline(self):
        """
        Creates and returns the data transformation pipeline.
        """

        splitter = TextSplitter(**configs["text_splitter"])

        embedder = adal.Embedder(
            model_client=configs["embedder"]["model_client"](),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )

        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)

        data_transformer = adal.Sequential(
            splitter, embedder_transformer
        )  # sequential will chain together splitter and embedder
        return data_transformer

    def _transform_documents_and_save_to_db(
        self, documents: List[Document], db_path: str
    ) -> LocalDB:
        """
        Transforms a list of documents and saves them to a local database.
        """
        # Get the data transformer
        data_transformer = self._prepare_data_pipeline()

        # Save the documents to a local database
        db = LocalDB()
        db.register_transformer(transformer=data_transformer, key="split_and_embed")
        db.load(documents)
        db.transform(key="split_and_embed")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.save_state(filepath=db_path)
        return db

    def _prepare_index_db(self) -> List[Document]:
        """
        Prepares the index database by reading all documents from the repository directory, 
        transforming them, and saving to the database.
        """
        # Check if DB exists
        if self.repo_paths and os.path.exists(self.repo_paths['db_file']):
            logger.info(f"Database already exists at {self.repo_paths['db_file']}. Loading existing database...")
            try:
                self.database = LocalDB.load_state(self.repo_paths['db_file'])
                documents = self.database.get_transformed_data(key="split_and_embed")
                if documents:
                    logger.info(f"Successfully loaded {len(documents)} documents from the database.")
                    return documents
            except Exception as e:
                logger.error(f"Failed to load existing database: {e}")
                raise HTTPException(status_code=500, detail="Failed to load existing database.")
        
        # If no existing database, create a new one
        logger.info("No existing database found. Creating a new database...")
        documents = self._read_all(
            path=self.repo_paths['repo_dir']
        )
        
        self.database = self._transform_documents_and_save_to_db(
            documents=documents,
            db_path=self.repo_paths['db_file']
        )
        
        logger.info(f"Database created and saved at {self.repo_paths['db_file']}.")
        embedding_model_name = configs["embedder"]["model_kwargs"]["model"]
        
        check_model_status = check_ollama_model_exists(embedding_model_name)
        if check_model_status == OLLAMA_MODEL_NOT_AVAILABLE_CODE:
            raise HTTPException(
                status_code=404,
                detail=f"Ollama model '{embedding_model_name}' is not available."
            )
        elif check_model_status == OLLAMA_SERVER_NOT_FOUND_ERROR_CODE:
            raise HTTPException(
                status_code=503,
                detail="Ollama server is not reachable. Please check the server status."
            )
        elif check_model_status == OLLAMA_UNKNOWN_ERROR_CODE:
            raise HTTPException(
                status_code=500,
                detail="An unknown error occurred while checking the Ollama model."
            )
        transformed_docs = self.database.get_transformed_data(key="split_and_embed")
        logger.info(f"Transformed {len(transformed_docs)} documents and saved to the database.")
        return transformed_docs
                
    def _check_repo_accessibility(self, repo_url: str, access_token: str = None) -> bool:
        """
            200: Repository accessible
            401: Wrong access token or authentication required
            404: Repository not found or empty token
            500: Other errors
        """
        try:
            parsed_url = urlparse(repo_url)
            if 'github.com' in parsed_url.netloc:
                path_parts = parsed_url.path.strip('/').split('/')
                if len(path_parts) >= 2:
                    owner, repo = path_parts[0], path_parts[1]
                    if repo.endswith('.git'):
                        repo = repo[:-4]
                        
                    # Check GitHub API
                    api_url = f"https://api.github.com/repos/{owner}/{repo}"
                    headers = {}
                    if access_token:
                        headers['Authorization'] = f'token {access_token}'
                    
                    response = requests.get(api_url, headers=headers, timeout=10)
                    return response.status_code
            elif 'gitlab.com' in parsed_url.netloc:
                path_parts = parsed_url.path.strip('/').split('/')
                if len(path_parts) >= 2:
                    project_path = '/'.join(path_parts[:2])
                    if project_path.endswith('.git'):
                        project_path = project_path[:-4]
                    
                    api_url = f"https://gitlab.com/api/v4/projects/{project_path.replace('/', '%2F')}"
                    headers = {}
                    if access_token:
                        headers['Authorization'] = f'Bearer {access_token}'
                    
                    response = requests.get(api_url, headers=headers, timeout=10)
                    return response.status_code
            
            return 500
        
        except Exception:
            return 500
        
    def _is_auth_error(self, error_msg: str) -> bool:
        """Check if error message indicates authentication issues"""
        auth_patterns = [
            r'authentication failed',
            r'invalid username or password',
            r'access denied',
            r'permission denied',
            r'could not read username',
            r'could not read password',
            r'terminal prompts disabled',
            r'authentication required',
            r'401 unauthorized',
            r'403 forbidden'
        ]
        
        error_lower = error_msg.lower()
        return any(re.search(pattern, error_lower) for pattern in auth_patterns)
    
    def _is_not_found_error(self, error_msg: str) -> bool:
        """Check if error message indicates repository not found"""
        not_found_patterns = [
            r'repository not found',
            r'could not read from remote repository',
            r'does not exist',
            r'404 not found',
            r'remote repository not found'
        ]
        
        error_lower = error_msg.lower()
        return any(re.search(pattern, error_lower) for pattern in not_found_patterns)

    def _download_repo_into_db(self, repo_url: str, access_token: str = None) -> None:
        logger.info(f"Preparing for download repo {repo_url}...")
        
        try:
            root_path = get_adalflow_default_root_path()
            
            os.makedirs(root_path, exist_ok=True)
            
            # get repo name
            repo_name = self._extract_repo_name(repo_url)
            logger.info(f"Successfully extracted repo name: {repo_name}")
            
            repo_dir = os.path.join(root_path, "repos", repo_name)
            
            # Check if the repository directory already exists and is not empty
            if os.path.exists(repo_dir) and os.listdir(repo_dir):
                logger.info(f"Repository {repo_name} already exists at {repo_dir}. Skipping download.")
            else:
                self._download_repo(repo_url, repo_dir, access_token)
            
            db_file = os.path.join(root_path, "db", f"{repo_name}.pkl")
            os.makedirs(repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
            
            self.repo_paths = {
                "repo_dir": repo_dir,
                "db_file": db_file
            }
            
            self.repo_url = repo_url
            
        except Exception as e:
            logger.error(f"Failed to download repository: {e}")
            raise
                
                
                
    