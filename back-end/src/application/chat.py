from __future__ import annotations

from typing import Any

from api.models.chat import ChatRequest
from api.models.chat import ChatResponse
from domain.preparator import LocalDBPreparator
from domain.preparator import PreparatorInput
from domain.rag import ChatRAG


class ChatApplication:
    def __init__(self, rag: ChatRAG, local_db_preparator: LocalDBPreparator):
        self.rag = rag
        self.local_db_preparator = local_db_preparator

    def prepare(self, preparator_input: PreparatorInput):
        transformed_docs = self.local_db_preparator.prepare(preparator_input)
        self.rag.prepare_retriever(transformed_docs)

    async def process(self, request: ChatRequest) -> ChatResponse:
        # Validate input
        if not request.message.strip():
            raise ValueError('Message cannot be empty')

        self.prepare(
            PreparatorInput(
                repo_url=request.repo_url,
                access_token=request.access_token,
            ),
        )

        response = self.rag.call(
            query=request.message,
            conversation_history_str=self._format_chat_history(request.chat_history) or '',
        )

        return ChatResponse(
            message=response['answer'],
            sources=response['sources'],
        )

    def _format_chat_history(self, conversation_history: Any) -> str:
        if not conversation_history:
            return ''
        # Expecting a list of dicts with 'role' and 'content'
        formatted = []
        for turn in conversation_history:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            formatted.append(f'{role.capitalize()}: {content}')
        return '\n'.join(formatted)
