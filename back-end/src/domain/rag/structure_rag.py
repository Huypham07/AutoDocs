from __future__ import annotations

import adalflow as adal
import google.generativeai as genai
from adalflow.components.model_client import OllamaClient
from adalflow.core.types import ModelType
from domain.preparator.local_db_preparator import count_tokens
from shared.logging import get_logger
from shared.settings.advanced_configs import get_generator_model_config

from .base import BaseRAG
from .base import RAG_TEMPLATE

logger = get_logger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 8192


class StructureRAG(BaseRAG):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, provider='ollama', model=None):
        """
        Initialize the RAG component.

        Args:
            provider: Model provider to use (google, openai, openrouter, ollama)
            model: Model name to use with the provider
        """
        super().__init__(provider=provider, model=model)

        self.generator_config = get_generator_model_config(self.provider, self.model)['model_kwargs']

    async def acall(self, query: str, system_prompt: str):
        """
        Process a query using RAG.

        Args:
            query: The user's query
        """
        try:
            logger.info(f"debug{self.provider}")
            input_too_large = False
            tokens = count_tokens(query, self.provider == 'ollama')
            logger.info(f'Request size: {tokens} tokens')
            if tokens > 8000:
                logger.warning(f'Request exceeds recommended token limit ({tokens} > 7500)')
                input_too_large = True

            if not input_too_large:
                retrieved_documents = self.retriever(query)

                # Fill in the documents
                retrieved_documents[0].documents = [
                    self.transformed_docs[doc_index]
                    for doc_index in retrieved_documents[0].doc_indices
                ]

                context = ''

                documents = retrieved_documents[0].documents
                logger.info(f'Retrieved {len(documents)} documents')

                # Group documents by file path
                docs_by_file: dict = {}
                for doc in documents:
                    file_path = doc.meta_data.get('file_path', 'unknown')
                    if file_path not in docs_by_file:
                        docs_by_file[file_path] = []
                    docs_by_file[file_path].append(doc)

                # Format context text with file path grouping
                context_parts = []
                for file_path, docs in docs_by_file.items():
                    # Add file header with metadata
                    header = f'## File Path: {file_path}\n\n'
                    # Add document content
                    content = '\n\n'.join([doc.text for doc in docs])

                    context_parts.append(f'{header}{content}')

                # Join all parts with clear separation
                context = '\n\n' + '-' * 10 + '\n\n'.join(context_parts)

            prompt = rf"""
            <START_OF_SYS_PROMPT>
            {system_prompt}
            <END_OF_SYS_PROMPT>

            <START_OF_CONTEXT>
            {context}
            <END_OF_CONTEXT>

            <START_OF_USER_PROMPT>
            {query}
            <END_OF_USER_PROMPT>
            """
            if self.provider == 'ollama':
                model = OllamaClient()
                model_kwargs = {
                    'model': self.generator_config['model'],
                    'stream': True,
                    'options': {
                        'temperature': self.generator_config['temperature'],
                        'top_p': self.generator_config['top_p'],
                        'num_ctx': self.generator_config['num_ctx'],
                    },
                }

                api_kwargs = model.convert_inputs_to_api_kwargs(
                    input=prompt,
                    model_kwargs=model_kwargs,
                    model_type=ModelType.LLM,
                )
            else:
                # Initialize Google Generative AI model
                model = genai.GenerativeModel(
                    model_name=self.generator_config['model'],
                    generation_config={
                        'temperature': self.generator_config['temperature'],
                        'top_p': self.generator_config['top_p'],
                        'top_k': self.generator_config['top_k'],
                    },
                )

            try:
                if self.provider == 'ollama':
                    # Get the response and handle it properly using the previously created api_kwargs
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # Handle streaming response from Ollama
                    async for chunk in response:
                        text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                        if text and not text.startswith('model=') and not text.startswith('created_at='):
                            text = text.replace('<think>', '').replace('</think>', '')
                            yield text
                else:
                    # Generate streaming response
                    response = await model.generate_content_async(prompt, stream=True)
                    # Stream the response
                    async for chunk in response:
                        if hasattr(chunk, 'text'):
                            yield chunk.text

            except Exception as e_outer:
                logger.error(f"Error in streaming response: {str(e_outer)}")
                error_message = str(e_outer)
                yield f"\nError: {error_message}"

        except Exception as e:
            logger.error(f'Error in RAG call: {str(e)}')
            raise ValueError('Error processing query')
