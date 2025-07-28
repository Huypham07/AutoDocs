from __future__ import annotations

from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ChatRequest(BaseModel):
    message: str = Field(..., description='User message to process')
    chat_history: Optional[List[dict]] = Field(None, description='Chat history for context')
    repo_url: str = Field(..., description='Repository URL for document retrieval')
    access_token: Optional[str] = Field(None, description='Access token for authentication')


class ChatResponse(BaseModel):
    message: str = Field(..., description='Response message from the model')
    sources: Optional[List[str]] = Field(None, description='Sources used for generating the response')
