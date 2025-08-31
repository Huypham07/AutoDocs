from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

import adalflow as adal
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
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


system_prompt = r"""You are a code assistant which answers user questions on a Github Repo.
You will receive user query, relevant context, and past conversation history.

IF THE CONTEXT IS IRRELEVANT OR INSUFFICIENT:
- If the context retrieved from the repo is irrelevant or insufficient to confidently answer the user query, reply briefly that you cannot answer due to lack of relevant information in the repo.
- Do not try to guess or hallucinate answers without clear support from the repo context.

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
{{conversation_history}}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index }}.
Content: {{context}}
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

    def call(self, query: str, conversation_history_str: Optional[str] = None):
        """
        Process a query using RAG.

        Args:
            query: The user's query
            conversation_history_str: Optional conversation history as a formatted string
        """
        try:
            input_too_large = False
            # tokens = count_tokens(query, self.provider == 'ollama')
            # logger.info(f'Request size: {tokens} tokens')
            # if tokens > 8000:
            #     logger.warning(f'Request exceeds recommended token limit ({tokens} > 7500)')
            #     input_too_large = True

            context = ''
            if not input_too_large:
                retrieved_documents = self.retriever(query)

                # Fill in the documents
                retrieved_documents[0].documents = [
                    self.transformed_docs[doc_index]
                    for doc_index in retrieved_documents[0].doc_indices
                ]

                documents = retrieved_documents[0].documents
                logger.info(f'Retrieved {len(documents)} documents')

                # Group documents by file path
                sources = []
                docs_by_file: dict = {}
                for doc in documents:
                    file_path = doc.meta_data.get('file_path', 'unknown')
                    if file_path not in docs_by_file:
                        docs_by_file[file_path] = []
                    docs_by_file[file_path].append(doc)
                    sources.append(file_path)

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

                MIN_RELEVANT_DOCS = 1
                MIN_SIMILARITY_SCORE = 0.2
                if not documents or len(documents) < MIN_RELEVANT_DOCS or retrieved_documents[0].doc_scores[0] < MIN_SIMILARITY_SCORE:
                    logger.info('Query is likely irrelevant to the repo. Returning default answer.')
                    return {
                        'answer': 'I’m sorry, but I cannot answer this question because it doesn’t appear to be related to the provided repository.',
                        'sources': [],
                    }

            response = self.generator.call(
                prompt_kwargs={
                    'input_str': query,
                    'contexts': context,
                    'conversation_history': conversation_history_str,
                },
            ).data

            return {
                'answer': response.answer,
                'sources': sources,
            }

        except Exception as e:
            logger.error(f'Error in RAG call: {str(e)}')
            raise ValueError('Error processing query')
