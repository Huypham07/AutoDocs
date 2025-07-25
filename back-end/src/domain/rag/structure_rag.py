from __future__ import annotations

from typing import Any
from typing import Dict

import adalflow as adal
from domain.preparator.local_db_preparator import count_tokens
from shared.logging import get_logger

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

        # Format instructions to ensure proper output structure
        format_instructions = """
IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them
"""

        # Set up the main generator
        if provider == 'openai':
            self.generator = adal.Generator(
                template=RAG_TEMPLATE,
                model_client=self.generator_config['model_client'](),
                model_kwargs=self.generator_config['model_kwargs'],
            )
        else:
            self.generator = adal.Generator(
                template=RAG_TEMPLATE,
                prompt_kwargs={
                    'output_format_str': format_instructions,
                },
                model_client=self.generator_config['model_client'](),
                model_kwargs=self.generator_config['model_kwargs'],
            )

    def call(self, query: str, structure_kwargs: Dict[str, Any], is_retrieval: bool = True):
        """
        Process a query using RAG.

        Args:
            query: The user's query
        """
        platform = structure_kwargs.get('platform', 'unknown')
        repo_url = structure_kwargs.get('repo_url', 'unknown')
        repo_name = structure_kwargs.get('repo_name', 'unknown')
        try:

            system_prompt = rf"""
            <role>
            You are an expert code analyst examining the {platform} repository: {repo_url} ({repo_name}).
            You provide direct, concise, and accurate information about code repositories.
            You NEVER start responses with markdown headers or code fences.
            IMPORTANT:You MUST respond in English.
            </role>

            <guidelines>
            - Answer the user's question directly without ANY preamble or filler phrases
            - DO NOT include any rationale, explanation, or extra comments.
            - DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation"
            - DO NOT start with markdown headers like "## Analysis of..." or any file path references
            - DO NOT start with ```markdown code fences
            - DO NOT end your response with ``` closing fences
            - DO NOT start by repeating or acknowledging the question
            - JUST START with the direct answer to the question

            <example_of_what_not_to_do>
            ```markdown
            ## Analysis of `adalflow/adalflow/datasets/gsm8k.py`

            This file contains...
            ```
            </example_of_what_not_to_do>

            - Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
            - For code analysis, organize your response with clear sections
            - Think step by step and structure your answer logically
            - Start with the most relevant information that directly addresses the user's query
            - Be precise and technical when discussing code
            - Your response language should be in the same language as the user's query
            </guidelines>

            <style>
            - Use concise, direct language
            - Prioritize accuracy over verbosity
            - When showing code, include line numbers and file paths when relevant
            - Use markdown formatting to improve readability
            </style>
            """

            input_too_large = False
            tokens = count_tokens(query, self.provider == 'ollama')
            logger.info(f'Request size: {tokens} tokens')
            if tokens > 8000:
                logger.warning(f'Request exceeds recommended token limit ({tokens} > 7500)')
                input_too_large = True

            context = ''
            if is_retrieval and not input_too_large:
                retrieved_documents = self.retriever(query)

                # Fill in the documents
                retrieved_documents[0].documents = [
                    self.transformed_docs[doc_index]
                    for doc_index in retrieved_documents[0].doc_indices
                ]

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

            response = self.generator.call(
                prompt_kwargs={
                    'input_str': query,
                    'contexts': context,
                    'system_prompt': system_prompt,
                },
            ).data

            return response

        except Exception as e:
            logger.error(f'Error in RAG call: {str(e)}')
            raise ValueError('Error processing query')
