"""
GraphRAG-optimized configuration for the architecture documentation pipeline.

This configuration focuses on graph construction and Neo4j population,
removing unnecessary embedding computations for pure graph-based RAG.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List


@dataclass
class GraphRAGPipelineConfig:
    """Configuration optimized for GraphRAG pipeline."""

    # Graph building - Keep these for graph construction
    extract_semantic_relationships: bool = True
    extract_structural_relationships: bool = True

    # Embeddings - Disable for pure graph mode
    compute_embeddings: bool = False
    compute_cluster_embeddings: bool = False

    # Graph enhancement - Focus on graph quality
    compute_centrality_measures: bool = True
    enhance_classification: bool = True

    # Semantic similarities - Disable since using graph relationships
    compute_semantic_similarities: bool = False

    # Clustering - Still useful for graph organization
    target_clusters: int = 10
    clustering_methods: List[str] = field(
        default_factory=lambda: [
            'community_detection', 'semantic_clustering', 'directory_clustering',
        ],
    )
    consensus_threshold: float = 0.5
    enable_clustering: bool = True

    # RAG optimization - Disable traditional text chunking
    generate_content_chunks: bool = False
    build_retrieval_indexes: bool = False
    optimize_for_traditional_rag: bool = False
    optimize_for_graph_rag: bool = True

    # Performance optimizations
    skip_vector_indexing: bool = True
    skip_faiss_building: bool = True
    focus_on_graph_population: bool = True

    # Neo4j specific
    populate_neo4j: bool = True
    neo4j_batch_size: int = 1000

    # Validation
    perform_validation: bool = True
    validation_threshold: float = 70.0

    # Output
    save_intermediate_results: bool = False  # Disable to save space
    output_directory: str = 'graph_pipeline_output'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'extract_semantic_relationships': self.extract_semantic_relationships,
            'extract_structural_relationships': self.extract_structural_relationships,
            'compute_embeddings': self.compute_embeddings,
            'compute_cluster_embeddings': self.compute_cluster_embeddings,
            'compute_centrality_measures': self.compute_centrality_measures,
            'enhance_classification': self.enhance_classification,
            'compute_semantic_similarities': self.compute_semantic_similarities,
            'target_clusters': self.target_clusters,
            'clustering_methods': self.clustering_methods,
            'consensus_threshold': self.consensus_threshold,
            'enable_clustering': self.enable_clustering,
            'generate_content_chunks': self.generate_content_chunks,
            'build_retrieval_indexes': self.build_retrieval_indexes,
            'optimize_for_traditional_rag': self.optimize_for_traditional_rag,
            'optimize_for_graph_rag': self.optimize_for_graph_rag,
            'skip_vector_indexing': self.skip_vector_indexing,
            'skip_faiss_building': self.skip_faiss_building,
            'focus_on_graph_population': self.focus_on_graph_population,
            'populate_neo4j': self.populate_neo4j,
            'neo4j_batch_size': self.neo4j_batch_size,
            'perform_validation': self.perform_validation,
            'validation_threshold': self.validation_threshold,
            'save_intermediate_results': self.save_intermediate_results,
            'output_directory': self.output_directory,
        }

    @classmethod
    def create_pure_graph_config(cls) -> GraphRAGPipelineConfig:
        """Create configuration for pure graph mode (no embeddings)."""
        return cls(
            compute_embeddings=False,
            compute_cluster_embeddings=False,
            compute_semantic_similarities=False,
            generate_content_chunks=False,
            build_retrieval_indexes=False,
            optimize_for_traditional_rag=False,
            optimize_for_graph_rag=True,
            skip_vector_indexing=True,
            skip_faiss_building=True,
            focus_on_graph_population=True,
            save_intermediate_results=False,
        )

    @classmethod
    def create_hybrid_config(cls) -> GraphRAGPipelineConfig:
        """Create configuration for hybrid mode (graph + minimal embeddings)."""
        return cls(
            compute_embeddings=True,
            compute_cluster_embeddings=False,  # Only module embeddings
            compute_semantic_similarities=False,  # Use graph relationships instead
            generate_content_chunks=True,
            build_retrieval_indexes=False,  # Use graph queries instead of FAISS
            optimize_for_traditional_rag=False,
            optimize_for_graph_rag=True,
            skip_vector_indexing=True,
            skip_faiss_building=True,
            focus_on_graph_population=True,
        )
