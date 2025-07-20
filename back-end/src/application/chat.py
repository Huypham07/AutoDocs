from __future__ import annotations

from api.models.chat import ChatRequest
from api.models.chat import ChatResponse
from domain.rag import ChatRAG


class ChatApplication:
    def __init__(self, rag: ChatRAG):
        self.rag = rag

    async def process(self, request: ChatRequest) -> ChatResponse:
        # Validate input
        if not request.message.strip():
            raise ValueError('Message cannot be empty')

        return ChatResponse(
            message='Processing your request...',
            sources=[],
        )
