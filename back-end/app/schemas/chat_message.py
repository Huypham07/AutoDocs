from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatCompletionRequest(BaseModel):
    """
    Schemas for requesting a chat completion
    """
    repo_url: HttpUrl = Field(..., description="URL of the repository")
    access_token: Optional[str] = Field(None, description="Access token for private repository")
