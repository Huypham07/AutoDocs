from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from adalflow.core.types import Document
from shared.logging import get_logger

from .base import BasePreparator
from .graph_builder import GraphBuilder
from .graph_builder import MultiModalGraph
from .graph_builder import save_graph
from .graph_enhancer import GraphEnhancer
from .hierarchical_clustering import ClusteringResults
from .hierarchical_clustering import HierarchicalClusterer
from .hierarchical_clustering import save_clustering_results
from .rag_serializer import RAGOptimizedData
from .rag_serializer import RAGSerializer
from .rag_serializer import save_rag_data
from .validation import generate_quality_summary
from .validation import PipelineValidator
from .validation import save_validation_report
from .validation import ValidationReport

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the architecture documentation pipeline."""
    # Graph building
    extract_semantic_relationships: bool = True
    extract_structural_relationships: bool = True
    compute_embeddings: bool = True

    # Graph enhancement
    compute_centrality_measures: bool = True
    enhance_classification: bool = True
    compute_semantic_similarities: bool = True

    # Clustering
    target_clusters: int = 10
    clustering_methods: List[str] = field(
        default_factory=lambda: [
            'community_detection', 'semantic_clustering', 'directory_clustering',
        ],
    )
    consensus_threshold: float = 0.5

    # RAG optimization
    generate_content_chunks: bool = True
    build_retrieval_indexes: bool = True
    compute_cluster_embeddings: bool = True

    # Validation
    perform_validation: bool = True
    validation_threshold: float = 70.0  # Minimum acceptable quality score

    # Output
    save_intermediate_results: bool = True
    output_directory: str = 'pipeline_output'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'extract_semantic_relationships': self.extract_semantic_relationships,
            'extract_structural_relationships': self.extract_structural_relationships,
            'compute_embeddings': self.compute_embeddings,
            'compute_centrality_measures': self.compute_centrality_measures,
            'enhance_classification': self.enhance_classification,
            'compute_semantic_similarities': self.compute_semantic_similarities,
            'target_clusters': self.target_clusters,
            'clustering_methods': self.clustering_methods,
            'consensus_threshold': self.consensus_threshold,
            'generate_content_chunks': self.generate_content_chunks,
            'build_retrieval_indexes': self.build_retrieval_indexes,
            'compute_cluster_embeddings': self.compute_cluster_embeddings,
            'perform_validation': self.perform_validation,
            'validation_threshold': self.validation_threshold,
            'save_intermediate_results': self.save_intermediate_results,
            'output_directory': self.output_directory,
        }


@dataclass
class PipelineResults:
    """Container for complete pipeline results."""
    # Core outputs
    graph: MultiModalGraph
    clustering_results: ClusteringResults
    rag_data: RAGOptimizedData
    validation_report: Optional[ValidationReport] = None

    # Metadata
    pipeline_config: Optional[PipelineConfig] = None
    execution_time: float = 0.0
    timestamp: str = ''
    repo_path: str = ''

    # Quality metrics
    overall_quality_score: float = 0.0
    validation_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'graph': self.graph.to_dict(),
            'clustering_results': self.clustering_results.to_dict(),
            'rag_data': self.rag_data.to_dict(),
            'validation_report': self.validation_report.to_dict() if self.validation_report else None,
            'pipeline_config': self.pipeline_config.to_dict() if self.pipeline_config else None,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'repo_path': self.repo_path,
            'overall_quality_score': self.overall_quality_score,
            'validation_passed': self.validation_passed,
        }


class ArchitecturePipelinePreparator(BasePreparator):
    """
    Complete architecture documentation preparation pipeline.

    This preparator implements the full pipeline from raw source code to
    RAG-optimized documentation data, following the 5-step approach:
    1. Multi-modal graph construction
    2. Graph enhancement
    3. Hierarchical clustering
    4. RAG serialization
    5. Quality validation
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.results: Optional[PipelineResults] = None

    def prepare(self, repo_url: str, access_token: Optional[str] = None) -> List[Document]:
        """
        Execute the complete architecture documentation preparation pipeline.

        Args:
            repo_url (str): The URL of the repository to document.
            access_token (Optional[str]): The access token for the repository.

        Returns:
            List[Document]: The transformed documents after preparation
        """
        logger.info('Starting architecture documentation preparation pipeline...')
        start_time = time.time()

        # First, use the existing local DB preparator to get the repo
        from .local_db_preparator import LocalDBPreparator
        local_preparator = LocalDBPreparator()
        documents = local_preparator.prepare(repo_url, access_token)

        if not documents:
            # Execute the pipeline
            repo_path = local_preparator.repo_paths['repo_dir']
            self.results = self.execute_pipeline(repo_path)

            # Create enhanced documents from the RAG-optimized data
            enhanced_docs = []

            # Add architecture overview
            overview = self.results.rag_data.overview
            enhanced_docs.append({
                'content': f"""# System Architecture Overview: {overview.system_name}

    ## Description
    {overview.architecture_description}

    ## Architecture Patterns
    {', '.join(overview.architectural_patterns)}

    ## Total modules: {overview.total_modules}
    """,
                'metadata': {
                    'type': 'architecture_overview',
                    'system_name': overview.system_name,
                    'source': 'architecture_pipeline',
                },
            })

            # Add module summaries with enhanced metadata
            for module in self.results.rag_data.modules:
                enhanced_docs.append({
                    'content': f"""# Module: {module.name}

    ## Path
    {module.file_path}

    ## Purpose
    {module.purpose}

    ## Dependencies
    {', '.join(module.dependencies)}

    ## Key Functions
    {', '.join(module.key_functions)}
    """,
                    'metadata': {
                        'type': 'module',
                        'name': module.name,
                        'source': 'architecture_pipeline',
                    },
                })

            # Add cluster summaries for architectural understanding
            for cluster in self.results.rag_data.clusters:
                enhanced_docs.append({
                    'content': f"""# Architectural Cluster: {cluster.name}

    ## Description
    {cluster.content_summary}

    ## Purpose
    {cluster.purpose}

    ## Modules
    {', '.join(cluster.modules)}

    ## External Dependencies
    {', '.join(cluster.external_dependencies)}
    """,
                    'metadata': {
                        'type': 'architectural_cluster',
                        'name': cluster.name,
                        'source': 'architecture_pipeline',
                    },
                })

            documents = [Document(text=doc['content'], meta_data=doc.get('metadata', {})) for doc in enhanced_docs]

            # Store execution metadata
            self.results.execution_time = time.time() - start_time
            self.results.timestamp = datetime.now().isoformat()
            self.results.repo_path = repo_path
            self.results.pipeline_config = self.config

            # Save complete results if configured
            if self.config.save_intermediate_results:
                self._save_complete_results()

            logger.info(
                f'Pipeline completed in {self.results.execution_time:.2f} seconds '
                f'with quality score {self.results.overall_quality_score:.1f}/100',
            )
            documents = local_preparator.prepare_index_db(documents)

        # Return traditional output for compatibility
        return documents

    def execute_pipeline(self, repo_path: str) -> PipelineResults:
        """
        Execute the complete pipeline on a repository.

        Args:
            repo_path: Path to the repository to analyze

        Returns:
            PipelineResults: Complete pipeline results
        """
        logger.info(f'Executing pipeline on repository: {repo_path}')

        # Step 1: Build multi-modal graph
        logger.info('Step 1/5: Building multi-modal graph...')
        graph = self._build_graph(repo_path)

        # Step 2: Enhance graph with metrics
        logger.info('Step 2/5: Enhancing graph with metrics...')
        enhanced_graph = self._enhance_graph(graph)

        # Step 3: Perform hierarchical clustering
        logger.info('Step 3/5: Performing hierarchical clustering...')
        clustering_results = self._cluster_modules(enhanced_graph)

        # Step 4: Serialize for RAG
        logger.info('Step 4/5: Serializing for RAG optimization...')
        rag_data = self._serialize_for_rag(enhanced_graph, clustering_results)

        # Step 5: Validate quality
        validation_report = None
        if self.config.perform_validation:
            logger.info('Step 5/5: Validating pipeline quality...')
            validation_report = self._validate_pipeline(enhanced_graph, clustering_results, rag_data)

        # Create results container
        results = PipelineResults(
            graph=enhanced_graph,
            clustering_results=clustering_results,
            rag_data=rag_data,
            validation_report=validation_report,
        )

        # Set quality metrics
        if validation_report:
            results.overall_quality_score = validation_report.quality_metrics.overall_score
            results.validation_passed = results.overall_quality_score >= self.config.validation_threshold

        return results

    def _build_graph(self, repo_path: str) -> MultiModalGraph:
        """Build multi-modal graph from repository."""
        graph_builder = GraphBuilder(repo_path)
        graph = graph_builder.build_graph()

        # Save intermediate result
        if self.config.save_intermediate_results:
            output_path = os.path.join(self.config.output_directory, '01_raw_graph.json')
            save_graph(graph, output_path)

        logger.info(f'Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges')
        return graph

    def _enhance_graph(self, graph: MultiModalGraph) -> MultiModalGraph:
        """Enhance graph with computed metrics."""
        enhancer = GraphEnhancer(graph)
        enhanced_graph = enhancer.enhance_graph()

        # Save intermediate result
        # if self.config.save_intermediate_results:
        #     output_path = os.path.join(self.config.output_directory, "02_enhanced_graph.json")
        #     save_enhanced_graph(enhanced_graph, output_path, include_metrics=True)

        logger.info('Graph enhancement completed')
        return enhanced_graph

    def _cluster_modules(self, graph: MultiModalGraph) -> ClusteringResults:
        """Perform hierarchical clustering."""
        clusterer = HierarchicalClusterer(graph, target_clusters=self.config.target_clusters)
        clustering_results = clusterer.cluster_modules()

        # Save intermediate result
        if self.config.save_intermediate_results:
            output_path = os.path.join(self.config.output_directory, '03_clustering_results.json')
            save_clustering_results(clustering_results, output_path)

        logger.info(f'Clustering completed with {len(clustering_results.final_clusters)} clusters')
        return clustering_results

    def _serialize_for_rag(self, graph: MultiModalGraph, clustering_results: ClusteringResults) -> RAGOptimizedData:
        """Serialize data for RAG optimization."""
        serializer = RAGSerializer(graph, clustering_results)

        # Extract system name from repo path
        system_name = os.path.basename(graph.nodes[next(iter(graph.nodes))].file_path.split('/')[0]) if graph.nodes else 'Unknown System'

        rag_data = serializer.serialize_for_rag(system_name)

        # Save intermediate result
        if self.config.save_intermediate_results:
            output_path = os.path.join(self.config.output_directory, '04_rag_optimized_data.json')
            save_rag_data(rag_data, output_path)

        logger.info(f'RAG serialization completed: {len(rag_data.modules)} modules, {len(rag_data.clusters)} clusters')
        return rag_data

    def _validate_pipeline(
        self, graph: MultiModalGraph, clustering_results: ClusteringResults,
        rag_data: RAGOptimizedData,
    ) -> ValidationReport:
        """Validate pipeline quality."""
        validator = PipelineValidator(graph, clustering_results, rag_data)
        validation_report = validator.validate_pipeline(self.config.to_dict())

        # Save validation report
        if self.config.save_intermediate_results:
            output_path = os.path.join(self.config.output_directory, '05_validation_report.json')
            save_validation_report(validation_report, output_path)

            # Also save human-readable summary
            summary_path = os.path.join(self.config.output_directory, 'validation_summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(generate_quality_summary(validation_report))

        logger.info(
            f'Validation completed: {validation_report.total_issues} issues found, '
            f'score: {validation_report.quality_metrics.overall_score:.1f}/100',
        )

        return validation_report

    def _save_complete_results(self):
        """Save complete pipeline results."""
        if not self.results:
            return

        # os.makedirs(self.config.output_directory, exist_ok=True)

        # Save complete results
        # output_path = os.path.join(self.config.output_directory, "complete_pipeline_results.json")
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(self.results.to_dict(), f, indent=2)

        # # Save configuration
        # config_path = os.path.join(self.config.output_directory, "pipeline_config.json")
        # with open(config_path, 'w', encoding='utf-8') as f:
        #     json.dump(self.config.to_dict(), f, indent=2)

        # logger.info(f"Complete results saved to {self.config.output_directory}")

    def get_results(self) -> Optional[PipelineResults]:
        """Get the results of the last pipeline execution."""
        return self.results

    def load_results(self, results_path: str) -> PipelineResults:
        """Load pipeline results from file."""
        with open(results_path, encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct objects (simplified - would need full reconstruction logic)
        graph = MultiModalGraph.from_dict(data['graph'])
        clustering_results = ClusteringResults.from_dict(data['clustering_results'])

        # Create RAG data (simplified reconstruction)
        from .rag_serializer import ArchitectureOverview, ModuleSummary, ClusterSummary
        rag_data_dict = data['rag_data']

        overview = ArchitectureOverview(**rag_data_dict['overview'])
        modules = [ModuleSummary(**module_data) for module_data in rag_data_dict['modules']]
        clusters = [ClusterSummary(**cluster_data) for cluster_data in rag_data_dict['clusters']]

        rag_data = RAGOptimizedData(
            overview=overview,
            modules=modules,
            clusters=clusters,
            **{k: v for k, v in rag_data_dict.items() if k not in ['overview', 'modules', 'clusters']},
        )

        # Validation report (if available)
        validation_report = None
        if data.get('validation_report'):
            from .validation import ValidationReport, QualityMetrics, ValidationIssue, ValidationSeverity
            vr_data = data['validation_report']

            quality_metrics = QualityMetrics(**vr_data['quality_metrics'])
            validation_issues = [
                ValidationIssue(
                    severity=ValidationSeverity(issue['severity']),
                    category=issue['category'],
                    message=issue['message'],
                    affected_items=issue.get('affected_items', []),
                    suggestion=issue.get('suggestion', ''),
                )
                for issue in vr_data.get('validation_issues', [])
            ]

            validation_report = ValidationReport(
                quality_metrics=quality_metrics,
                validation_issues=validation_issues,
                recommendations=vr_data.get('recommendations', []),
                pipeline_parameters=vr_data.get('pipeline_parameters', {}),
                total_issues=vr_data.get('total_issues', 0),
                critical_issues=vr_data.get('critical_issues', 0),
                error_issues=vr_data.get('error_issues', 0),
                warning_issues=vr_data.get('warning_issues', 0),
                info_issues=vr_data.get('info_issues', 0),
            )

        # Create results
        results = PipelineResults(
            graph=graph,
            clustering_results=clustering_results,
            rag_data=rag_data,
            validation_report=validation_report,
            execution_time=data.get('execution_time', 0.0),
            timestamp=data.get('timestamp', ''),
            repo_path=data.get('repo_path', ''),
            overall_quality_score=data.get('overall_quality_score', 0.0),
            validation_passed=data.get('validation_passed', False),
        )

        if data.get('pipeline_config'):
            results.pipeline_config = PipelineConfig(**data['pipeline_config'])

        self.results = results
        logger.info(f'Pipeline results loaded from {results_path}')
        return results
