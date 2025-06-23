from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class RepoFetchingRequest(BaseModel):
    """
    Schemas for repo fetching
    """
    repo_url: HttpUrl = Field(..., description="URL of the repository")
    access_token: Optional[str] = Field(None, description="Access token for private repository")
    is_ollama_embedding: bool = Field(False, description="Use Ollama embedding for processing")
    
    excluded_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to exclude from processing")
    excluded_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to exclude from processing")
    included_dirs: Optional[str] = Field(None, description="Comma-separated list of directories to include exclusively")
    included_files: Optional[str] = Field(None, description="Comma-separated list of file patterns to include exclusively")
