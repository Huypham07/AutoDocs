"""
Optional embedding and retrieval system for architecture documentation.

This module provides embedding-based knowledge retrieval capabilities
for the generated architecture documentation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
from shared.logging import get_logger

logger = get_logger(__name__)

# Optional MongoDB integration
try:
    from infra.mongo.core import MongoDBConnection
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning('MongoDB not available, using file-based storage only')


@dataclass
class EmbeddingEntry:
    """Represents an embedded piece of architectural knowledge."""

    id: str
    content: str
    content_type: str  # 'module_summary', 'diagram_desc', 'interaction', 'pattern'
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'content_type': self.content_type,
            'metadata': self.metadata,
            'embedding': self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingEntry:
        """Create from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            content_type=data['content_type'],
            metadata=data['metadata'],
            embedding=data.get('embedding'),
        )


class EmbeddingGenerator:
    """Generates embeddings for architectural knowledge."""

    def __init__(self, embedding_model: str = 'sentence-transformers'):
        self.embedding_model = embedding_model
        self._embedder = None

    def _get_embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            try:
                # Try to use sentence-transformers if available
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info('Loaded SentenceTransformer model')
            except ImportError:
                logger.warning('sentence-transformers not available, using simple TF-IDF')
                self._embedder = self._create_tfidf_embedder()
        return self._embedder

    def _create_tfidf_embedder(self):
        """Create a simple TF-IDF based embedder as fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(max_features=384, stop_words='english')

    def generate_embeddings(self, entries: List[EmbeddingEntry]) -> List[EmbeddingEntry]:
        """Generate embeddings for a list of entries."""
        logger.info(f'Generating embeddings for {len(entries)} entries...')

        embedder = self._get_embedder()
        contents = [entry.content for entry in entries]

        if hasattr(embedder, 'encode'):
            # sentence-transformers
            embeddings = embedder.encode(contents)
            for i, entry in enumerate(entries):
                entry.embedding = embeddings[i].tolist()
        else:
            # TF-IDF
            embeddings = embedder.fit_transform(contents)
            for i, entry in enumerate(entries):
                entry.embedding = embeddings[i].toarray().flatten().tolist()

        logger.info('Embeddings generated successfully')
        return entries


class KnowledgeRetriever:
    """Retrieves relevant architectural knowledge using embeddings."""

    def __init__(self, entries: List[EmbeddingEntry]):
        self.entries = entries
        self.embedding_generator = EmbeddingGenerator()

        # Build index
        self._build_index()

    def _build_index(self):
        """Build search index from embeddings."""
        if not self.entries or not self.entries[0].embedding:
            logger.warning('No embeddings available for indexing')
            return

        # Convert embeddings to numpy array for efficient search
        embeddings = np.array([entry.embedding for entry in self.entries])
        self.embedding_matrix = embeddings
        logger.info(f'Built embedding index with {len(self.entries)} entries')

    def search(self, query: str, top_k: int = 5) -> List[Tuple[EmbeddingEntry, float]]:
        """
        Search for relevant knowledge entries.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (entry, similarity_score) tuples
        """
        if not hasattr(self, 'embedding_matrix'):
            logger.warning('No embedding index available')
            return []

        # Generate embedding for query
        embedder = self.embedding_generator._get_embedder()

        if hasattr(embedder, 'encode'):
            query_embedding = embedder.encode([query])[0]
        else:
            # For TF-IDF, we need to transform using the fitted vectorizer
            query_embedding = embedder.transform([query]).toarray().flatten()

        # Calculate similarities
        similarities = np.dot(self.embedding_matrix, query_embedding)

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            entry = self.entries[idx]
            similarity = float(similarities[idx])
            results.append((entry, similarity))

        return results

    def search_by_type(self, content_type: str, query: str = '', top_k: int = 5) -> List[EmbeddingEntry]:
        """Search for entries of a specific type."""
        filtered_entries = [entry for entry in self.entries if entry.content_type == content_type]

        if not query:
            return filtered_entries[:top_k]

        # Create temporary retriever for filtered entries
        temp_retriever = KnowledgeRetriever(filtered_entries)
        results = temp_retriever.search(query, top_k)
        return [entry for entry, _ in results]


class ArchitectureKnowledgeBase:
    """Complete knowledge base for architecture documentation."""

    def __init__(self, use_mongodb: bool = False, collection_name: str = 'architecture_knowledge'):
        self.entries: List[EmbeddingEntry] = []
        self.retriever: Optional[KnowledgeRetriever] = None
        self.use_mongodb = use_mongodb and MONGO_AVAILABLE
        self.collection_name = collection_name
        self._mongo_client = None

        if self.use_mongodb:
            self._init_mongodb()

    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            self._mongo_client = MongoDBConnection()
            logger.info(f'Initialized MongoDB for collection: {self.collection_name}')
        except Exception as e:
            logger.warning(f'Failed to initialize MongoDB: {e}')
            self.use_mongodb = False

    def add_modules_knowledge(self, modules: List[Any], knowledge_list: List[Any]):
        """Add module knowledge to the knowledge base."""
        logger.info('Adding module knowledge to knowledge base...')

        knowledge_map = {k.module_id: k for k in knowledge_list}

        for module in modules:
            knowledge = knowledge_map.get(module.id)
            if not knowledge:
                continue

            # Add module summary
            summary_entry = EmbeddingEntry(
                id=f'module_summary_{module.id}',
                content=knowledge.summary,
                content_type='module_summary',
                metadata={
                    'module_id': module.id,
                    'module_name': module.name,
                    'layer': module.layer,
                },
            )
            self.entries.append(summary_entry)

            # Add detailed description
            if knowledge.detailed_description:
                detail_entry = EmbeddingEntry(
                    id=f'module_detail_{module.id}',
                    content=knowledge.detailed_description,
                    content_type='module_detail',
                    metadata={
                        'module_id': module.id,
                        'module_name': module.name,
                        'layer': module.layer,
                    },
                )
                self.entries.append(detail_entry)

            # Add interactions
            for target_module_id, interaction_desc in knowledge.interactions.items():
                interaction_entry = EmbeddingEntry(
                    id=f'interaction_{module.id}_{target_module_id}',
                    content=f'{module.name} interaction: {interaction_desc}',
                    content_type='interaction',
                    metadata={
                        'source_module': module.id,
                        'target_module': target_module_id,
                        'source_name': module.name,
                    },
                )
                self.entries.append(interaction_entry)

            # Add patterns
            for pattern in knowledge.patterns_used:
                pattern_entry = EmbeddingEntry(
                    id=f'pattern_{module.id}_{pattern}',
                    content=f'{module.name} uses {pattern} pattern',
                    content_type='pattern',
                    metadata={
                        'module_id': module.id,
                        'module_name': module.name,
                        'pattern': pattern,
                    },
                )
                self.entries.append(pattern_entry)

        logger.info(f'Added {len(self.entries)} knowledge entries')

    def add_architecture_overview(self, overview: Any):
        """Add architecture overview knowledge."""
        logger.info('Adding architecture overview to knowledge base...')

        # Add system context
        if overview.system_context:
            context_entry = EmbeddingEntry(
                id='system_context',
                content=overview.system_context,
                content_type='system_context',
                metadata={'section': 'system_context'},
            )
            self.entries.append(context_entry)

        # Add overview
        if overview.overview:
            overview_entry = EmbeddingEntry(
                id='architecture_overview',
                content=overview.overview,
                content_type='overview',
                metadata={'section': 'overview'},
            )
            self.entries.append(overview_entry)

        # Add layer analysis
        if overview.layer_analysis:
            layer_entry = EmbeddingEntry(
                id='layer_analysis',
                content=overview.layer_analysis,
                content_type='layer_analysis',
                metadata={'section': 'layer_analysis'},
            )
            self.entries.append(layer_entry)

        # Add diagrams metadata
        for diagram_type, diagram_content in overview.diagrams.items():
            if diagram_type != 'error':
                diagram_entry = EmbeddingEntry(
                    id=f'diagram_{diagram_type}',
                    content=f"Architecture diagram showing {diagram_type.replace('_', ' ')}",
                    content_type='diagram_description',
                    metadata={
                        'diagram_type': diagram_type,
                        'diagram_content': diagram_content,
                    },
                )
                self.entries.append(diagram_entry)

        logger.info(f'Architecture overview added, total entries: {len(self.entries)}')

    def build_index(self):
        """Build the embedding index for retrieval."""
        logger.info('Building embedding index...')

        if not self.entries:
            logger.warning('No entries to index')
            return

        # Generate embeddings
        embedding_generator = EmbeddingGenerator()
        self.entries = embedding_generator.generate_embeddings(self.entries)

        # Build retriever
        self.retriever = KnowledgeRetriever(self.entries)

        logger.info('Embedding index built successfully')

    def search(self, query: str, top_k: int = 5) -> List[Tuple[EmbeddingEntry, float]]:
        """Search the knowledge base."""
        if not self.retriever:
            logger.warning('Index not built. Call build_index() first.')
            return []

        return self.retriever.search(query, top_k)

    def get_module_knowledge(self, module_name: str) -> List[EmbeddingEntry]:
        """Get all knowledge entries for a specific module."""
        return [
            entry for entry in self.entries
            if entry.metadata.get('module_name') == module_name
        ]

    def get_patterns(self) -> List[str]:
        """Get all design patterns mentioned in the knowledge base."""
        patterns = set()
        for entry in self.entries:
            if entry.content_type == 'pattern':
                pattern = entry.metadata.get('pattern')
                if pattern:
                    patterns.add(pattern)
        return sorted(patterns)

    def get_layers(self) -> List[str]:
        """Get all architectural layers."""
        layers = set()
        for entry in self.entries:
            layer = entry.metadata.get('layer')
            if layer:
                layers.add(layer)
        return sorted(layers)

    def save(self, output_path: str):
        """Save knowledge base to file and optionally MongoDB."""
        # Always save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        serializable_entries = [entry.to_dict() for entry in self.entries]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_entries, f, indent=2)

        logger.info(f'Saved knowledge base with {len(self.entries)} entries to {output_path}')

        # Also save to MongoDB if available
        if self.use_mongodb and self._mongo_client:
            self._save_to_mongodb()

    def _save_to_mongodb(self):
        """Save knowledge base to MongoDB."""
        try:
            db = self._mongo_client.get_database()
            collection = db[self.collection_name]

            # Clear existing data
            collection.delete_many({})

            # Insert new data
            if self.entries:
                documents = [entry.to_dict() for entry in self.entries]
                collection.insert_many(documents)

            logger.info(f'Saved {len(self.entries)} entries to MongoDB collection: {self.collection_name}')

        except Exception as e:
            logger.error(f'Failed to save to MongoDB: {e}')

    def load(self, input_path: str, prefer_mongodb: bool = True):
        """Load knowledge base from file or MongoDB."""
        loaded_from_mongo = False

        # Try MongoDB first if preferred and available
        if prefer_mongodb and self.use_mongodb and self._mongo_client:
            loaded_from_mongo = self._load_from_mongodb()

        # Fall back to file if MongoDB failed or not preferred
        if not loaded_from_mongo:
            self._load_from_file(input_path)

    def _load_from_mongodb(self) -> bool:
        """Load knowledge base from MongoDB."""
        try:
            if self._mongo_client:
                db = self._mongo_client.get_database()
                collection = db[self.collection_name]

                documents = list(collection.find({}))

                if documents:
                    self.entries = [EmbeddingEntry.from_dict(doc) for doc in documents]

                    # Rebuild retriever if embeddings exist
                    if self.entries and self.entries[0].embedding:
                        self.retriever = KnowledgeRetriever(self.entries)

                    logger.info(f'Loaded {len(self.entries)} entries from MongoDB collection: {self.collection_name}')
                    return True
                else:
                    logger.info('No data found in MongoDB collection')
                    return False
            return False
        except Exception as e:
            logger.error(f'Failed to load from MongoDB: {e}')
            return False

    def _load_from_file(self, input_path: str):
        """Load knowledge base from file."""
        with open(input_path, encoding='utf-8') as f:
            serialized_entries = json.load(f)

        self.entries = [EmbeddingEntry.from_dict(data) for data in serialized_entries]

        # Rebuild retriever if embeddings exist
        if self.entries and self.entries[0].embedding:
            self.retriever = KnowledgeRetriever(self.entries)

        logger.info(f'Loaded knowledge base with {len(self.entries)} entries from {input_path}')

    def search_by_metadata(self, metadata_filters: Dict[str, Any], top_k: int = 10) -> List[EmbeddingEntry]:
        """Search entries by metadata filters."""
        filtered_entries = []

        for entry in self.entries:
            match = True
            for key, value in metadata_filters.items():
                if key not in entry.metadata or entry.metadata[key] != value:
                    match = False
                    break

            if match:
                filtered_entries.append(entry)

        return filtered_entries[:top_k]

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the architecture."""
        summary: Dict[str, Any] = {
            'total_entries': len(self.entries),
            'content_types': {},
            'modules': set(),
            'layers': set(),
            'patterns': set(),
        }

        for entry in self.entries:
            # Count content types
            content_type = entry.content_type
            content_types: Dict[str, int] = summary['content_types']
            content_types[content_type] = content_types.get(content_type, 0) + 1

            # Collect modules
            if 'module_name' in entry.metadata:
                modules: Set[str] = summary['modules']
                modules.add(entry.metadata['module_name'])

            # Collect layers
            if 'layer' in entry.metadata:
                layers: Set[str] = summary['layers']
                layers.add(entry.metadata['layer'])

            # Collect patterns
            if entry.content_type == 'pattern' and 'pattern' in entry.metadata:
                patterns: Set[str] = summary['patterns']
                patterns.add(entry.metadata['pattern'])

        # Convert sets to sorted lists
        summary['modules'] = sorted(summary['modules'])
        summary['layers'] = sorted(summary['layers'])
        summary['patterns'] = sorted(summary['patterns'])

        return summary


class ArchitectureRAGService:
    """Retrieval-Augmented Generation service for architecture queries."""

    def __init__(self, knowledge_base: ArchitectureKnowledgeBase):
        self.knowledge_base = knowledge_base

    def answer_query(self, query: str, max_context_entries: int = 5) -> str:
        """
        Answer a query about the architecture using RAG.

        Args:
            query: User question about the architecture
            max_context_entries: Maximum number of knowledge entries to use as context

        Returns:
            Generated answer based on retrieved knowledge
        """
        if not self.knowledge_base.retriever:
            return 'Knowledge base index not available. Please build the index first.'

        # Retrieve relevant knowledge
        results = self.knowledge_base.search(query, max_context_entries)

        if not results:
            return 'No relevant information found in the knowledge base.'

        # Build context from retrieved entries
        context_parts = []
        for entry, similarity in results:
            context_parts.append(f'[{entry.content_type}] {entry.content}')

        context = '\n\n'.join(context_parts)

        # Simple template-based response (could be replaced with LLM integration)
        response = self._generate_template_response(query, context, results)

        return response

    def _generate_template_response(
        self,
        query: str,
        context: str,
        results: List[Tuple[EmbeddingEntry, float]],
    ) -> str:
        """Generate a template-based response (fallback when no LLM is available)."""

        # Categorize the query
        query_lower = query.lower()

        if any(word in query_lower for word in ['module', 'component']):
            return self._answer_module_query(query, results)
        elif any(word in query_lower for word in ['pattern', 'design']):
            return self._answer_pattern_query(query, results)
        elif any(word in query_lower for word in ['layer', 'architecture']):
            return self._answer_architecture_query(query, results)
        elif any(word in query_lower for word in ['interaction', 'dependency']):
            return self._answer_interaction_query(query, results)
        else:
            return self._answer_general_query(query, results)

    def _answer_module_query(self, query: str, results: List[Tuple[EmbeddingEntry, float]]) -> str:
        """Answer module-related questions."""
        response_parts = ['Based on the architecture analysis:\n']

        modules = set()
        for entry, _ in results:
            if entry.content_type in ['module_summary', 'module_detail']:
                module_name = entry.metadata.get('module_name', 'Unknown')
                modules.add(module_name)
                response_parts.append(f'• **{module_name}**: {entry.content}')

        if modules:
            response_parts.insert(1, f'Found information about {len(modules)} relevant module(s):\n')

        return '\n'.join(response_parts)

    def _answer_pattern_query(self, query: str, results: List[Tuple[EmbeddingEntry, float]]) -> str:
        """Answer pattern-related questions."""
        response_parts = ['Design patterns found in the architecture:\n']

        patterns = set()
        for entry, _ in results:
            if entry.content_type == 'pattern':
                pattern = entry.metadata.get('pattern', 'Unknown')
                module_name = entry.metadata.get('module_name', 'Unknown')
                patterns.add(pattern)
                response_parts.append(f'• **{pattern}** pattern used in {module_name}')

        if not patterns:
            response_parts = ['No specific design patterns found for this query.']

        return '\n'.join(response_parts)

    def _answer_architecture_query(self, query: str, results: List[Tuple[EmbeddingEntry, float]]) -> str:
        """Answer architecture-related questions."""
        response_parts = ['Architecture overview:\n']

        for entry, _ in results:
            if entry.content_type in ['system_context', 'overview', 'layer_analysis']:
                response_parts.append(f'{entry.content}\n')

        return '\n'.join(response_parts)

    def _answer_interaction_query(self, query: str, results: List[Tuple[EmbeddingEntry, float]]) -> str:
        """Answer interaction-related questions."""
        response_parts = ['Module interactions:\n']

        for entry, _ in results:
            if entry.content_type == 'interaction':
                response_parts.append(f'• {entry.content}')

        if len(response_parts) == 1:
            response_parts = ['No specific module interactions found for this query.']

        return '\n'.join(response_parts)

    def _answer_general_query(self, query: str, results: List[Tuple[EmbeddingEntry, float]]) -> str:
        """Answer general questions."""
        if not results:
            return 'No relevant information found.'

        response_parts = ['Based on the available documentation:\n']

        for entry, similarity in results[:3]:  # Top 3 results
            response_parts.append(f'• {entry.content}')

        return '\n'.join(response_parts)
