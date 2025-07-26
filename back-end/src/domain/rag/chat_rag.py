from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple

import adalflow as adal
from adalflow.components.model_client import GoogleGenAIClient
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core import Conversation
from adalflow.core.types import Document
from shared.logging import get_logger
from shared.settings.advanced_configs import configs
from shared.settings.advanced_configs import get_generator_model_config

logger = get_logger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 8192


@dataclass
class ChatRAGAnswer(adal.DataClass):
    rationale: str = field(default='', metadata={'desc': 'Chain of thoughts for the answer.'})
    answer: str = field(default='', metadata={'desc': 'Answer to the user query, formatted in markdown for beautiful rendering with react-markdown. DO NOT include ``` triple backticks fences at the beginning or end of your answer.'})

    __output_fields__ = ['rationale', 'answer']


system_prompt = r"""/no_think

You are a code assistant which answers user questions on a Github Repo.
You will receive user query, relevant context, and past conversation history.

LANGUAGE DETECTION AND RESPONSE:
- Detect the language of the user's query
- Respond in the SAME language as the user's query
- IMPORTANT:If a specific language is requested in the prompt, prioritize that language over the query language

FORMAT YOUR RESPONSE USING MARKDOWN:
- Use proper markdown syntax for all formatting
- For code blocks, use triple backticks with language specification (```python, ```javascript, etc.)
- Use ## headings for major sections
- Use bullet points or numbered lists where appropriate
- Format tables using markdown table syntax when presenting structured data
- Use **bold** and *italic* for emphasis
- When referencing file paths, use `inline code` formatting

IMPORTANT FORMATTING RULES:
1. DO NOT include ```markdown fences at the beginning or end of your answer
2. Start your response directly with the content
3. The content will already be rendered as markdown, so just provide the raw markdown content

Think step by step and ensure your answer is well-structured and visually organized.
"""

# Template for RAG
RAG_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
{{output_format_str}}
<END_OF_SYS_PROMPT>
{# OrderedDict of DialogTurn #}
{% if conversation_history %}
<START_OF_CONVERSATION_HISTORY>
{% for key, dialog_turn in conversation_history.items() %}
{{key}}.
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index }}.
File Path: {{context.meta_data.get('file_path', 'unknown')}}
Content: {{context.text}}
{% endfor %}
<END_OF_CONTEXT>
{% endif %}
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""


class ChatRAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, provider='ollama', model=None):
        """
        Initialize the RAG component.

        Args:
            provider: Model provider to use (google, openai, openrouter, ollama)
            model: Model name to use with the provider
        """
        super().__init__()

        self.provider = provider
        self.model = model

        # Initialize components
        self.conversation_history = Conversation()
        self.embedder = adal.Embedder(
            model_client=configs['embedder']['model_client'](),
            model_kwargs=configs['embedder']['model_kwargs'],
        )

        # Patch: ensure query embedding is always single string for Ollama
        def single_string_embedder(query):
            # Accepts either a string or a list, always returns embedding for a single string
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError('Ollama embedder only supports a single string')
                query = query[0]
            return self.embedder(input=query)

        # Use single string embedder for Ollama, regular embedder for others
        self.query_embedder = single_string_embedder

        # self.initialize_db_manager()

        # Set up the output parser
        data_parser = adal.DataClassParser(data_class=ChatRAGAnswer, return_data_class=True)

        # Format instructions to ensure proper output structure
        format_instructions = data_parser.get_output_format_str() + """

IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
4. DO NOT wrap your response in any kind of fences
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them"""

        generator_config = get_generator_model_config(self.provider, self.model)

        # Set up the main generator
        if provider == 'openai':
            self.generator = adal.Generator(
                template=RAG_TEMPLATE,
                prompt_kwargs={
                    # 'conversation_history': self.conversation_history,
                    'system_prompt': system_prompt,
                    'contexts': None,
                },
                model_client=generator_config['model_client'](),
                model_kwargs=generator_config['model_kwargs'],
                output_processors=data_parser,
            )
        else:
            self.generator = adal.Generator(
                template=RAG_TEMPLATE,
                prompt_kwargs={
                    'output_format_str': format_instructions,
                    # 'conversation_history': self.conversation_history,
                    'system_prompt': system_prompt,
                    'contexts': None,
                },
                model_client=generator_config['model_client'](),
                model_kwargs=generator_config['model_kwargs'],
                output_processors=data_parser,
            )

    def _validate_and_filter_embeddings(self, documents: List[Document]) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched embedding sizes.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning('No documents provided for embedding validation')
            return []

        valid_documents = []
        embedding_sizes: dict = {}

        # First pass: collect all embedding sizes and count occurrences
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f'Document {i} has no embedding vector, skipping')
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(f'Document {i} has invalid embedding vector type: {type(doc.vector)}, skipping')
                    continue

                if embedding_size == 0:
                    logger.warning(f'Document {i} has empty embedding vector, skipping')
                    continue

                embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

            except Exception as e:
                logger.warning(f'Error checking embedding size for document {i}: {str(e)}, skipping')
                continue

        if not embedding_sizes:
            logger.error('No valid embeddings found in any documents')
            return []

        # Find the most common embedding size (this should be the correct one)
        target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
        logger.info(f'Target embedding size: {target_size} (found in {embedding_sizes[target_size]} documents)')

        # Log all embedding sizes found
        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f'Found {count} documents with incorrect embedding size {size}, will be filtered out')

        # Second pass: filter documents with the target embedding size
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    continue

                if embedding_size == target_size:
                    valid_documents.append(doc)
                else:
                    # Log which document is being filtered out
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Filtering out document '{file_path}' due to embedding size mismatch: {embedding_size} != {target_size}")

            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.warning(f"Error validating embedding for document '{file_path}': {str(e)}, skipping")
                continue

        logger.info(f'Embedding validation complete: {len(valid_documents)}/{len(documents)} documents have valid embeddings')

        if len(valid_documents) == 0:
            logger.error('No documents with valid embeddings remain after filtering')
        elif len(valid_documents) < len(documents):
            filtered_count = len(documents) - len(valid_documents)
            logger.warning(f'Filtered out {filtered_count} documents due to embedding issues')

        return valid_documents

    def prepare_retriever(self, transformed_docs: List[Document]):
        """
        Prepare the retriever for a repository.
        Will load database from local storage if available.
        """
        self.transformed_docs = transformed_docs
        logger.info(f'Loaded {len(self.transformed_docs)} documents for retrieval')

        # Validate and filter embeddings to ensure consistent sizes
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            raise ValueError('No valid documents with embeddings found. Cannot create retriever.')

        logger.info(f'Using {len(self.transformed_docs)} documents with valid embeddings for retrieval')

        try:
            # Use the appropriate embedder for retrieval
            retrieve_embedder = self.query_embedder
            self.retriever = FAISSRetriever(
                **configs['retriever'],
                embedder=retrieve_embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info('FAISS retriever created successfully')
        except Exception as e:
            logger.error(f'Error creating FAISS retriever: {str(e)}')
            # Try to provide more specific error information
            if 'All embeddings should be of the same size' in str(e):
                logger.error('Embedding size validation failed. This suggests there are still inconsistent embedding sizes.')
                # Log embedding sizes for debugging
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):  # Check first 10 docs
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            size = None
                            if isinstance(doc.vector, list):
                                size = len(doc.vector)
                            elif hasattr(doc.vector, 'shape'):
                                size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                            elif hasattr(doc.vector, '__len__'):
                                size = len(doc.vector)
                            sizes.append(f'doc_{i}: {size if size is not None else "unknown"}')
                        except Exception:
                            sizes.append(f'doc_{i}: error')
                logger.error(f"Sample embedding sizes: {', '.join(sizes)}")
            raise

    def __call__(self, query: str):
        """
        Process a query using RAG.

        Args:
            query: The user's query

        Returns:
            Tuple of (RAGAnswer, retrieved_documents)
        """
        try:
            retrieved_documents = self.retriever(query)

            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]
            self.generator(
                input=query,
            )

            return retrieved_documents

        except Exception as e:
            logger.error(f'Error in RAG call: {str(e)}')

            # Create error response
            error_response = ChatRAGAnswer(
                rationale='Error occurred while processing the query.',
                answer='I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.',
            )
            return error_response, []
