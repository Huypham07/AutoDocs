from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class TaskRequest(BaseModel):
    """
    Schemas for repo fetching
    """
    repo_url: HttpUrl = Field(..., description="URL of the repository")
    access_token: Optional[str] = Field(None, description="Access token for private repository")
