import logging
import os
from typing import Sequence
from copy import deepcopy
from tqdm import tqdm
import adalflow as adal
from adalflow.core.types import Document
from adalflow.core.component import DataComponent
import requests
from app.core.config import OLLAMA_HOST

from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OLLAMA_MODEL_AVAILABLE_CODE = 200
OLLAMA_MODEL_NOT_AVAILABLE_CODE = 404
OLLAMA_SERVER_NOT_FOUND_ERROR_CODE = 503
OLLAMA_UNKNOWN_ERROR_CODE = 500

def check_ollama_model_exists(model_name: str) -> int:
    """
    Check if a model exists on the Ollama server.

    Returns:
        200 - Model is available
        404 - Model not available
        503 - Server unreachable
        500 - Unexpected error
    """
    ollama_host = OLLAMA_HOST
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model.get('name', '').split(':')[0] for model in models_data.get('models', [])]
            model_base_name = model_name.split(':')[0]  # Remove tag if present
            
            if model_base_name in available_models:
                logger.info(f"Ollama model '{model_name}' is available")
                return OLLAMA_MODEL_AVAILABLE_CODE
            else:
                logger.warning(f"Ollama model '{model_name}' is not available. Available models: {available_models}")
                return OLLAMA_MODEL_NOT_AVAILABLE_CODE
        else:
            logger.warning(f"Could not check Ollama models, status code: {response.status_code}")
            return response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error when checking Ollama model: {e}")
        return OLLAMA_SERVER_NOT_FOUND_ERROR_CODE
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return OLLAMA_UNKNOWN_ERROR_CODE

class OllamaDocumentProcessor(DataComponent):
    """
    Process documents for Ollama embeddings by processing one document at a time.
    Adalflow Ollama Client does not support batch embedding, so we need to process each document individually.
    """
    def __init__(self, embedder: adal.Embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents individually for Ollama embeddings")

        successful_docs = []
        expected_embedding_size = None

        for i, doc in enumerate(tqdm(output, desc="Processing documents for Ollama embeddings")):
            try:
                # Get embedding for a single document
                result = self.embedder(input=doc.text)
                if result.data and len(result.data) > 0:
                    embedding = result.data[0].embedding

                    # Validate embedding size consistency
                    if expected_embedding_size is None:
                        expected_embedding_size = len(embedding)
                        logger.info(f"Expected embedding size set to: {expected_embedding_size}")
                    elif len(embedding) != expected_embedding_size:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                        logger.warning(f"Document '{file_path}' has inconsistent embedding size {len(embedding)} != {expected_embedding_size}, skipping")
                        continue

                    # Assign the embedding to the document
                    output[i].vector = embedding
                    successful_docs.append(output[i])
                else:
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Failed to get embedding for document '{file_path}', skipping")
            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.error(f"Error processing document '{file_path}': {e}, skipping")

        logger.info(f"Successfully processed {len(successful_docs)}/{len(output)} documents with consistent embeddings")
        return successful_docs