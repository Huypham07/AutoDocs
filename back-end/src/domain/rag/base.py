from __future__ import annotations

from typing import List

import adalflow as adal
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document
from shared.logging import get_logger
from shared.settings.advanced_configs import configs
from shared.settings.advanced_configs import get_generator_model_config

logger = get_logger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 8192

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


class BaseRAG(adal.Component):
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

        self.generator_config = get_generator_model_config(self.provider, self.model)

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
