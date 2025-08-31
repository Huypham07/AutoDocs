"""
Validation and quality metrics module for pipeline assessment.

This module assesses the quality of clustering and graph construction to ensure
reliable documentation generation through comprehensive validation metrics.
"""
from __future__ import annotations

import json
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
from shared.logging import get_logger

from .graph_builder import GraphEdge
from .graph_builder import GraphNode
from .graph_builder import MultiModalGraph
from .hierarchical_clustering import ClusteringResults
from .hierarchical_clustering import ModuleCluster
from .rag_serializer import RAGOptimizedData

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


@dataclass
class ValidationIssue:
    """Represents a validation issue found during quality assessment."""
    severity: ValidationSeverity
    category: str
    message: str
    affected_items: List[str] = field(default_factory=list)
    suggestion: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'affected_items': self.affected_items,
            'suggestion': self.suggestion,
        }


@dataclass
class QualityMetrics:
    """Container for quality assessment metrics."""
    # Graph metrics
    graph_density: float = 0.0
    graph_modularity: float = 0.0
    average_clustering_coefficient: float = 0.0

    # Clustering metrics
    cluster_coherence: float = 0.0
    silhouette_score: float = 0.0
    cluster_size_variance: float = 0.0

    # Coverage metrics
    coverage_completeness: float = 0.0
    orphaned_modules: int = 0
    singleton_clusters: int = 0

    # Dependency metrics
    circular_dependencies: int = 0
    dependency_depth: int = 0
    critical_path_length: int = 0

    # Interface metrics
    interface_clarity: float = 0.0
    boundary_violations: int = 0

    # Overall quality score (0-100)
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'graph_density': self.graph_density,
            'graph_modularity': self.graph_modularity,
            'average_clustering_coefficient': self.average_clustering_coefficient,
            'cluster_coherence': self.cluster_coherence,
            'silhouette_score': self.silhouette_score,
            'cluster_size_variance': self.cluster_size_variance,
            'coverage_completeness': self.coverage_completeness,
            'orphaned_modules': self.orphaned_modules,
            'singleton_clusters': self.singleton_clusters,
            'circular_dependencies': self.circular_dependencies,
            'dependency_depth': self.dependency_depth,
            'critical_path_length': self.critical_path_length,
            'interface_clarity': self.interface_clarity,
            'boundary_violations': self.boundary_violations,
            'overall_score': self.overall_score,
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for pipeline quality."""
    quality_metrics: QualityMetrics
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    pipeline_parameters: Dict[str, Any] = field(default_factory=dict)

    # Summary statistics
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'quality_metrics': self.quality_metrics.to_dict(),
            'validation_issues': [issue.to_dict() for issue in self.validation_issues],
            'recommendations': self.recommendations,
            'pipeline_parameters': self.pipeline_parameters,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'error_issues': self.error_issues,
            'warning_issues': self.warning_issues,
            'info_issues': self.info_issues,
        }


class PipelineValidator:
    """
    Validates the quality of the architecture analysis pipeline output.
    """

    def __init__(
        self, graph: MultiModalGraph, clustering_results: ClusteringResults,
        rag_data: Optional[RAGOptimizedData] = None,
    ):
        self.graph = graph
        self.clustering_results = clustering_results
        self.rag_data = rag_data
        self.validation_issues: List[ValidationIssue] = []
        self.quality_metrics = QualityMetrics()

    def validate_pipeline(self, pipeline_params: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Perform comprehensive validation of the pipeline output.

        Args:
            pipeline_params: Parameters used in the pipeline for reference

        Returns:
            ValidationReport: Comprehensive validation results
        """
        logger.info('Starting pipeline validation...')

        # Reset state
        self.validation_issues = []
        self.quality_metrics = QualityMetrics()

        # Perform validation checks
        self._validate_graph_quality()
        self._validate_clustering_quality()
        self._validate_coverage_completeness()
        self._detect_circular_dependencies()
        self._validate_interface_boundaries()
        self._validate_architectural_violations()

        if self.rag_data:
            self._validate_rag_data_quality()

        # Compute overall quality score
        self._compute_overall_quality_score()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Create report
        report = ValidationReport(
            quality_metrics=self.quality_metrics,
            validation_issues=self.validation_issues,
            recommendations=recommendations,
            pipeline_parameters=pipeline_params or {},
        )

        # Update issue counts
        report.total_issues = len(self.validation_issues)
        report.critical_issues = sum(
            1 for issue in self.validation_issues
            if issue.severity == ValidationSeverity.CRITICAL
        )
        report.error_issues = sum(
            1 for issue in self.validation_issues
            if issue.severity == ValidationSeverity.ERROR
        )
        report.warning_issues = sum(
            1 for issue in self.validation_issues
            if issue.severity == ValidationSeverity.WARNING
        )
        report.info_issues = sum(
            1 for issue in self.validation_issues
            if issue.severity == ValidationSeverity.INFO
        )

        logger.info(
            f'Validation completed: {report.total_issues} issues found, '
            f'overall score: {self.quality_metrics.overall_score:.1f}/100',
        )

        return report

    def _validate_graph_quality(self):
        """Validate graph construction quality."""
        logger.info('Validating graph quality...')

        total_nodes = len(self.graph.nodes)
        total_edges = len(self.graph.edges)

        if total_nodes == 0:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category='graph_structure',
                    message='No nodes found in graph',
                    suggestion='Check repository parsing and ensure code files are accessible',
                ),
            )
            return

        # Calculate graph density
        max_edges = total_nodes * (total_nodes - 1)
        self.quality_metrics.graph_density = total_edges / max_edges if max_edges > 0 else 0

        # Check for extremely sparse or dense graphs
        if self.quality_metrics.graph_density < 0.01:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='graph_structure',
                    message=f'Graph is very sparse (density: {self.quality_metrics.graph_density:.4f})',
                    suggestion='Consider adjusting edge creation criteria or checking import parsing',
                ),
            )
        elif self.quality_metrics.graph_density > 0.5:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='graph_structure',
                    message=f'Graph is very dense (density: {self.quality_metrics.graph_density:.4f})',
                    suggestion='Consider filtering weak relationships or adjusting weight thresholds',
                ),
            )

        # Check for isolated nodes
        isolated_nodes = []
        for node_id, node in self.graph.nodes.items():
            if not self.graph.get_neighbors(node_id) and not self.graph.get_predecessors(node_id):
                isolated_nodes.append(node_id)

        if isolated_nodes:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='graph_structure',
                    message=f'Found {len(isolated_nodes)} isolated nodes',
                    affected_items=isolated_nodes[:10],  # Limit to first 10
                    suggestion='Review relationship extraction algorithms',
                ),
            )

        # Validate node embeddings
        nodes_without_embeddings = [
            node_id for node_id, node in self.graph.nodes.items()
            if not node.embedding
        ]

        if nodes_without_embeddings:
            severity = ValidationSeverity.WARNING if len(nodes_without_embeddings) < total_nodes * 0.5 else ValidationSeverity.ERROR
            self.validation_issues.append(
                ValidationIssue(
                    severity=severity,
                    category='embeddings',
                    message=f'{len(nodes_without_embeddings)} nodes lack embeddings',
                    suggestion='Ensure embedding generation is working correctly',
                ),
            )

    def _validate_clustering_quality(self):
        """Validate clustering quality."""
        logger.info('Validating clustering quality...')

        clusters = self.clustering_results.final_clusters
        total_clusters = len(clusters)

        if total_clusters == 0:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category='clustering',
                    message='No clusters generated',
                    suggestion='Check clustering algorithm parameters and input data',
                ),
            )
            return

        # Analyze cluster sizes
        cluster_sizes = [cluster.size for cluster in clusters.values()]
        self.quality_metrics.cluster_size_variance = np.var(cluster_sizes) if cluster_sizes else 0

        # Check for singleton clusters
        singleton_clusters = [cid for cid, cluster in clusters.items() if cluster.size == 1]
        self.quality_metrics.singleton_clusters = len(singleton_clusters)

        if self.quality_metrics.singleton_clusters > total_clusters * 0.3:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='clustering',
                    message=f'High number of singleton clusters: {self.quality_metrics.singleton_clusters}',
                    suggestion='Consider increasing cluster merge threshold or adjusting target cluster count',
                ),
            )

        # Check for overly large clusters
        max_size = max(cluster_sizes) if cluster_sizes else 0
        if max_size > len(self.graph.nodes) * 0.5:
            large_clusters = [
                cid for cid, cluster in clusters.items()
                if cluster.size > len(self.graph.nodes) * 0.3
            ]
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='clustering',
                    message=f'Found overly large clusters (max size: {max_size})',
                    affected_items=large_clusters,
                    suggestion='Consider splitting large clusters or adjusting clustering parameters',
                ),
            )

        # Calculate cluster coherence (average internal cohesion)
        cohesions = [cluster.internal_cohesion for cluster in clusters.values()]
        self.quality_metrics.cluster_coherence = np.mean(cohesions) if cohesions else 0

        if self.quality_metrics.cluster_coherence < 0.3:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='clustering',
                    message=f'Low cluster coherence: {self.quality_metrics.cluster_coherence:.3f}',
                    suggestion='Review clustering algorithm or consider different similarity metrics',
                ),
            )

        # Validate cluster metadata
        clusters_without_purpose = [
            cid for cid, cluster in clusters.items()
            if not cluster.cluster_purpose or cluster.cluster_purpose == ''
        ]

        if clusters_without_purpose:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category='metadata',
                    message=f'{len(clusters_without_purpose)} clusters lack purpose descriptions',
                    suggestion='Enhance cluster purpose generation algorithms',
                ),
            )

    def _validate_coverage_completeness(self):
        """Validate coverage completeness."""
        logger.info('Validating coverage completeness...')

        total_nodes = len(self.graph.nodes)
        clustered_nodes = set()

        for cluster in self.clustering_results.final_clusters.values():
            clustered_nodes.update(cluster.modules)

        unclustered_nodes = set(self.graph.nodes.keys()) - clustered_nodes
        self.quality_metrics.orphaned_modules = len(unclustered_nodes)

        if unclustered_nodes:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='coverage',
                    message=f'{len(unclustered_nodes)} modules are not assigned to any cluster',
                    affected_items=list(unclustered_nodes)[:20],  # Limit to first 20
                    suggestion="Ensure all modules are assigned to clusters or create 'miscellaneous' cluster",
                ),
            )

        # Coverage completeness percentage
        self.quality_metrics.coverage_completeness = (total_nodes - len(unclustered_nodes)) / total_nodes if total_nodes > 0 else 0

        if self.quality_metrics.coverage_completeness < 0.95:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='coverage',
                    message=f'Coverage completeness is {self.quality_metrics.coverage_completeness:.1%}',
                    suggestion='Improve clustering to include all modules',
                ),
            )

    def _detect_circular_dependencies(self):
        """Detect circular dependencies."""
        logger.info('Detecting circular dependencies...')

        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs_cycles(node_id, path):
            if node_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:] + [node_id]
                cycles.append(cycle)
                return

            if node_id in visited:
                return

            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in self.graph.get_neighbors(node_id):
                dfs_cycles(neighbor, path + [neighbor])

            rec_stack.remove(node_id)

        for node_id in self.graph.nodes.keys():
            if node_id not in visited:
                dfs_cycles(node_id, [node_id])

        self.quality_metrics.circular_dependencies = len(cycles)

        if cycles:
            severity = ValidationSeverity.ERROR if len(cycles) > 5 else ValidationSeverity.WARNING
            cycle_descriptions = [' -> '.join(cycle) for cycle in cycles[:5]]

            self.validation_issues.append(
                ValidationIssue(
                    severity=severity,
                    category='dependencies',
                    message=f'Found {len(cycles)} circular dependencies',
                    affected_items=cycle_descriptions,
                    suggestion='Review and refactor to break circular dependencies',
                ),
            )

    def _validate_interface_boundaries(self):
        """Validate interface boundary clarity."""
        logger.info('Validating interface boundaries...')

        boundary_violations = 0

        # Check for proper layer separation
        layer_order = ['presentation', 'business', 'data', 'infrastructure']
        layer_indices = {layer: i for i, layer in enumerate(layer_order)}

        for (source, target), edge in self.graph.edges.items():
            source_node = self.graph.nodes.get(source)
            target_node = self.graph.nodes.get(target)

            if source_node and target_node:
                source_layer_idx = layer_indices.get(source_node.layer, 1)
                target_layer_idx = layer_indices.get(target_node.layer, 1)

                # Check for upward dependencies (violation)
                if source_layer_idx > target_layer_idx:
                    boundary_violations += 1

        self.quality_metrics.boundary_violations = boundary_violations

        if boundary_violations > 0:
            severity = ValidationSeverity.ERROR if boundary_violations > 10 else ValidationSeverity.WARNING
            self.validation_issues.append(
                ValidationIssue(
                    severity=severity,
                    category='architecture',
                    message=f'Found {boundary_violations} layer boundary violations',
                    suggestion='Review architecture to ensure proper layer separation',
                ),
            )

        # Calculate interface clarity
        total_edges = len(self.graph.edges)
        self.quality_metrics.interface_clarity = (total_edges - boundary_violations) / total_edges if total_edges > 0 else 0

    def _validate_architectural_violations(self):
        """Detect architectural violations."""
        logger.info('Validating architectural patterns...')

        # Check for god objects (high centrality + high complexity)
        god_objects = []
        for node_id, node in self.graph.nodes.items():
            pagerank = node.centrality_measures.get('pagerank', 0)
            complexity = node.complexity_metrics.get('cyclomatic_complexity', 0)

            if pagerank > 0.1 and complexity > 20:
                god_objects.append(node_id)

        if god_objects:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='architecture',
                    message=f'Potential god objects detected: {len(god_objects)}',
                    affected_items=god_objects,
                    suggestion='Consider refactoring high-complexity, high-centrality modules',
                ),
            )

        # Check for feature envy (modules with many external dependencies)
        feature_envy_modules = []
        for node_id, node in self.graph.nodes.items():
            external_deps = len(self.graph.get_neighbors(node_id))
            if external_deps > 10:
                feature_envy_modules.append(node_id)

        if feature_envy_modules:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category='design',
                    message=f'Modules with high external dependencies: {len(feature_envy_modules)}',
                    affected_items=feature_envy_modules[:10],
                    suggestion='Consider whether these modules have too many responsibilities',
                ),
            )

    def _validate_rag_data_quality(self):
        """Validate RAG data quality."""
        logger.info('Validating RAG data quality...')

        if not self.rag_data:
            return

        # Check module summaries
        modules_without_embeddings = [
            module.id for module in self.rag_data.modules
            if not module.embedding
        ]

        if modules_without_embeddings:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='rag_data',
                    message=f'{len(modules_without_embeddings)} modules lack embeddings in RAG data',
                    suggestion='Ensure embedding generation for all modules',
                ),
            )

        # Check content chunks
        modules_without_content = [
            module.id for module in self.rag_data.modules
            if not module.content_chunks
        ]

        if modules_without_content:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category='rag_data',
                    message=f'{len(modules_without_content)} modules lack content chunks',
                    suggestion='Improve content extraction for better RAG performance',
                ),
            )

        # Validate indexes
        if not self.rag_data.content_index:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='rag_data',
                    message='Content index is empty',
                    suggestion='Ensure content indexing is functioning correctly',
                ),
            )

    def _compute_overall_quality_score(self):
        """Compute overall quality score (0-100)."""
        logger.info('Computing overall quality score...')

        # Weight different aspects
        weights = {
            'graph_quality': 0.25,
            'clustering_quality': 0.25,
            'coverage': 0.20,
            'dependencies': 0.15,
            'interfaces': 0.15,
        }

        scores = {}

        # Graph quality score
        density_score = min(self.quality_metrics.graph_density * 100, 50)  # Cap at 50 for density
        graph_completeness = 100 if len(self.graph.nodes) > 0 else 0
        scores['graph_quality'] = (density_score + graph_completeness) / 2

        # Clustering quality score
        coherence_score = self.quality_metrics.cluster_coherence * 100
        singleton_penalty = min(self.quality_metrics.singleton_clusters * 5, 50)
        scores['clustering_quality'] = max(coherence_score - singleton_penalty, 0)

        # Coverage score
        scores['coverage'] = self.quality_metrics.coverage_completeness * 100

        # Dependencies score (inverse of violations)
        dep_penalty = min(self.quality_metrics.circular_dependencies * 10, 80)
        scores['dependencies'] = max(100 - dep_penalty, 20)

        # Interface score
        scores['interfaces'] = self.quality_metrics.interface_clarity * 100

        # Compute weighted average
        self.quality_metrics.overall_score = sum(
            scores[aspect] * weight
            for aspect, weight in weights.items()
        )

        # Apply penalty for critical issues
        critical_penalty = sum(
            5 for issue in self.validation_issues
            if issue.severity == ValidationSeverity.CRITICAL
        )
        error_penalty = sum(
            2 for issue in self.validation_issues
            if issue.severity == ValidationSeverity.ERROR
        )

        self.quality_metrics.overall_score = max(
            self.quality_metrics.overall_score - critical_penalty - error_penalty,
            0,
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for pipeline improvement."""
        recommendations = []

        # Critical issues first
        critical_issues = [
            issue for issue in self.validation_issues
            if issue.severity == ValidationSeverity.CRITICAL
        ]
        if critical_issues:
            recommendations.append('Address critical issues immediately - pipeline may not function correctly')

        # Specific recommendations based on metrics
        if self.quality_metrics.graph_density < 0.01:
            recommendations.append('Consider relaxing edge creation criteria to capture more relationships')

        if self.quality_metrics.cluster_coherence < 0.3:
            recommendations.append('Improve clustering algorithms or use different similarity metrics')

        if self.quality_metrics.coverage_completeness < 0.9:
            recommendations.append('Ensure all modules are properly clustered')

        if self.quality_metrics.circular_dependencies > 5:
            recommendations.append('Refactor code to eliminate circular dependencies')

        if self.quality_metrics.boundary_violations > 10:
            recommendations.append('Review and enforce proper architectural layer separation')

        if self.quality_metrics.overall_score < 70:
            recommendations.append('Consider adjusting pipeline parameters and re-running analysis')

        # Add general recommendations
        if self.quality_metrics.overall_score >= 80:
            recommendations.append('Pipeline quality is good - consider fine-tuning for optimization')
        elif self.quality_metrics.overall_score >= 60:
            recommendations.append('Pipeline quality is acceptable - address major issues for improvement')
        else:
            recommendations.append('Pipeline quality needs significant improvement - review all validation issues')

        return recommendations


def save_validation_report(report: ValidationReport, output_path: str):
    """Save validation report to JSON file."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f'Validation report saved to {output_path}')


def generate_quality_summary(report: ValidationReport) -> str:
    """Generate a human-readable quality summary."""
    summary_lines = [
        'Pipeline Quality Assessment',
        '=' * 30,
        f'Overall Score: {report.quality_metrics.overall_score:.1f}/100',
        '',
        'Issues Summary:',
        f'- Critical: {report.critical_issues}',
        f'- Errors: {report.error_issues}',
        f'- Warnings: {report.warning_issues}',
        f'- Info: {report.info_issues}',
        '',
        'Key Metrics:',
        f'- Graph Density: {report.quality_metrics.graph_density:.4f}',
        f'- Cluster Coherence: {report.quality_metrics.cluster_coherence:.3f}',
        f'- Coverage: {report.quality_metrics.coverage_completeness:.1%}',
        f'- Circular Dependencies: {report.quality_metrics.circular_dependencies}',
        '',
        'Recommendations:',
    ]

    for i, rec in enumerate(report.recommendations, 1):
        summary_lines.append(f'{i}. {rec}')

    return '\n'.join(summary_lines)
