from __future__ import annotations

from typing import List
from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[dict]] = None  # Each dict: {'role': 'user'|'assistant', 'content': str}


class ChatResponse(BaseModel):
    message: str
    sources: Optional[List[dict]] = None
