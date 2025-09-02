from __future__ import annotations

from typing import Any
from typing import Optional

from api.models.chat import ChatRequest
from api.models.chat import ChatResponse
from domain.rag import GraphRAG
from domain.rag.graph_rag import GraphQuery


class ChatApplication:
    def __init__(self, rag: GraphRAG):
        self.rag = rag

    def prepare(self, repo_url: str, access_token: Optional[str] = None):
        pass

    async def process(self, request: ChatRequest) -> ChatResponse:
        # Validate input
        if not request.message.strip():
            raise ValueError('Message cannot be empty')

        self.prepare(
            repo_url=request.repo_url,
            access_token=request.access_token,
        )

        graph_query = GraphQuery(
            query_type='chat_qa',
            context=request.message,
            repo_url=request.repo_url,
        )

        # Kết hợp chat history với question
        enhanced_question = self._enhance_question_with_history(
            request.message,
            request.chat_history,
        )

        # Query GraphRAG
        response_text = self.rag.query(
            question=enhanced_question,
            repo_url=request.repo_url,
            query_type='chat_qa',
        )

        # Extract sources from graph context
        sources = self._extract_graph_sources(graph_query)

        return ChatResponse(
            message=response_text,
            sources=sources,
        )

    def _enhance_question_with_history(self, question: str, chat_history: Any) -> str:
        """Enhance question with chat history context."""
        if not chat_history:
            return question

        history_text = self._format_chat_history(chat_history)
        if history_text:
            return f"""Previous conversation:
{history_text}

Current question: {question}

Please answer the current question considering the previous conversation context."""
        return question

    def _extract_graph_sources(self, graph_query: GraphQuery) -> list:
        """Extract source information from graph context."""
        try:
            # Build context để lấy sources
            graph_context = self.rag.context_builder.build_context(graph_query)
            sources = []

            # Thêm các graph-based sources
            if graph_context.structured_data:
                if 'relevant_modules' in graph_context.structured_data:
                    modules = graph_context.structured_data['relevant_modules']
                    for module in modules:
                        if isinstance(module, dict) and 'file_path' in module:
                            sources.append(f"Graph: {module['file_path']}")

                if 'relevant_dataflows' in graph_context.structured_data:
                    sources.append('Graph: Data Flow Analysis')

                if 'system_overview' in graph_context.structured_data:
                    sources.append('Graph: System Architecture Overview')

                if 'clusters' in graph_context.structured_data:
                    sources.append('Graph: Cluster Analysis')

                if 'technology_stack' in graph_context.structured_data:
                    sources.append('Graph: Technology Stack')

            return sources if sources else ['Graph: Repository Analysis']

        except Exception:
            return ['Graph: Repository Analysis']

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
