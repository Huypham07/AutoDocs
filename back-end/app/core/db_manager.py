import logging
import os
import subprocess

from fastapi import HTTPException
import requests
import re

from adalflow.core.types import Document, List
from adalflow.utils import get_adalflow_default_root_path
from app.core.logging import setup_logging
from urllib.parse import urlparse, urlunparse

setup_logging()
logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self):
        self.database = None
        self.repo_url = None
        self.repo_paths = None

    
    def prepare_db(self, repo_url: str, access_token: str = None, is_ollama_embedding: bool = None,
                    excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                    included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Prepare the database with the given repository URL and access token.
        """
        self.database = None
        self.repo_url = None
        self.repo_paths = None
        
        self._download_repo_into_db(repo_url, access_token)
        
    
    def _extract_repo_name(self, repo_url: str) -> str:
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
        
    def _check_repo_accessibility(self, repo_url: str, access_token: str = None) -> bool:
        """
        Returns:
            200: Repository accessible
            401: Requires authentication
            403: Access denied (with token)
            404: Repository not found
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
                
                
                
    