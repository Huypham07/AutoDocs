"""
Graph serialization module for RAG (Retrieval-Augmented Generation) optimization.

This module transforms clustered graph data into formats optimized for:
- RAG retrieval and documentation generation
- Human-readable summaries
- LLM consumption
- Interface and dependency information
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from shared.logging import get_logger

from .graph_builder import GraphEdge
from .graph_builder import GraphNode
from .graph_builder import MultiModalGraph
from .hierarchical_clustering import ClusteringResults
from .hierarchical_clustering import ModuleCluster

logger = get_logger(__name__)

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning('Scikit-learn not available, using simplified similarity')


@dataclass
class ModuleSummary:
    """Human-readable summary of a module."""
    id: str
    name: str
    file_path: str
    module_type: str
    layer: str
    domain: str

    # Content summaries
    purpose: str
    key_functions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

    # Metrics
    complexity_level: str = 'medium'
    lines_of_code: int = 0
    centrality_score: float = 0.0

    # For RAG
    content_chunks: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'module_type': self.module_type,
            'layer': self.layer,
            'domain': self.domain,
            'purpose': self.purpose,
            'key_functions': self.key_functions,
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'complexity_level': self.complexity_level,
            'lines_of_code': self.lines_of_code,
            'centrality_score': self.centrality_score,
            'content_chunks': self.content_chunks,
            'keywords': self.keywords,
            'embedding': self.embedding,
        }


@dataclass
class ClusterSummary:
    """Human-readable summary of a cluster."""
    id: str
    name: str
    purpose: str
    modules: List[str] = field(default_factory=list)

    # Cluster characteristics
    layer: str = ''
    domain: str = ''
    size: int = 0
    complexity_level: str = 'medium'

    # Quality metrics
    cohesion: float = 0.0
    coupling: float = 0.0

    # Relationships
    interfaces: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)

    # For RAG
    content_summary: str = ''
    interaction_patterns: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'purpose': self.purpose,
            'modules': self.modules,
            'layer': self.layer,
            'domain': self.domain,
            'size': self.size,
            'complexity_level': self.complexity_level,
            'cohesion': self.cohesion,
            'coupling': self.coupling,
            'interfaces': self.interfaces,
            'external_dependencies': self.external_dependencies,
            'content_summary': self.content_summary,
            'interaction_patterns': self.interaction_patterns,
            'embedding': self.embedding,
        }


@dataclass
class ArchitectureOverview:
    """High-level architecture overview."""
    system_name: str
    total_modules: int
    total_clusters: int

    # Layer distribution
    layer_distribution: Dict[str, int] = field(default_factory=dict)
    domain_distribution: Dict[str, int] = field(default_factory=dict)

    # Key patterns
    architectural_patterns: List[str] = field(default_factory=list)
    design_principles: List[str] = field(default_factory=list)

    # Dependencies
    critical_dependencies: List[str] = field(default_factory=list)
    circular_dependencies: List[str] = field(default_factory=list)

    # Quality metrics
    overall_modularity: float = 0.0
    complexity_distribution: Dict[str, int] = field(default_factory=dict)

    # Narrative summary
    executive_summary: str = ''
    architecture_description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'system_name': self.system_name,
            'total_modules': self.total_modules,
            'total_clusters': self.total_clusters,
            'layer_distribution': self.layer_distribution,
            'domain_distribution': self.domain_distribution,
            'architectural_patterns': self.architectural_patterns,
            'design_principles': self.design_principles,
            'critical_dependencies': self.critical_dependencies,
            'circular_dependencies': self.circular_dependencies,
            'overall_modularity': self.overall_modularity,
            'complexity_distribution': self.complexity_distribution,
            'executive_summary': self.executive_summary,
            'architecture_description': self.architecture_description,
        }


@dataclass
class RAGOptimizedData:
    """Container for RAG-optimized architecture data."""
    overview: ArchitectureOverview
    modules: List[ModuleSummary] = field(default_factory=list)
    clusters: List[ClusterSummary] = field(default_factory=list)

    # Retrieval indexes
    content_index: Dict[str, List[str]] = field(default_factory=dict)
    dependency_index: Dict[str, List[str]] = field(default_factory=dict)
    keyword_index: Dict[str, List[str]] = field(default_factory=dict)

    # Graph data for traversal
    adjacency_lists: Dict[str, List[str]] = field(default_factory=dict)
    reverse_adjacency_lists: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    generation_timestamp: str = ''
    version: str = '1.0'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overview': self.overview.to_dict(),
            'modules': [module.to_dict() for module in self.modules],
            'clusters': [cluster.to_dict() for cluster in self.clusters],
            'content_index': self.content_index,
            'dependency_index': self.dependency_index,
            'keyword_index': self.keyword_index,
            'adjacency_lists': self.adjacency_lists,
            'reverse_adjacency_lists': self.reverse_adjacency_lists,
            'generation_timestamp': self.generation_timestamp,
            'version': self.version,
        }


class RAGSerializer:
    """
    Transforms clustered graph data into RAG-optimized formats.
    """

    def __init__(self, graph: MultiModalGraph, clustering_results: ClusteringResults):
        self.graph = graph
        self.clustering_results = clustering_results

    def serialize_for_rag(self, system_name: str = 'Code Architecture') -> RAGOptimizedData:
        """
        Transform graph and clustering data into RAG-optimized format.

        Args:
            system_name: Name of the system being analyzed

        Returns:
            RAGOptimizedData: Optimized data for RAG consumption
        """
        logger.info('Serializing graph data for RAG optimization...')

        # Create architecture overview
        overview = self._create_architecture_overview(system_name)

        # Create module summaries
        modules = self._create_module_summaries()

        # Create cluster summaries
        clusters = self._create_cluster_summaries()

        # Build retrieval indexes
        content_index = self._build_content_index(modules, clusters)
        dependency_index = self._build_dependency_index(modules)
        keyword_index = self._build_keyword_index(modules, clusters)

        # Build graph traversal data
        adjacency_lists, reverse_adjacency_lists = self._build_adjacency_data()

        # Create RAG data container
        rag_data = RAGOptimizedData(
            overview=overview,
            modules=modules,
            clusters=clusters,
            content_index=content_index,
            dependency_index=dependency_index,
            keyword_index=keyword_index,
            adjacency_lists=adjacency_lists,
            reverse_adjacency_lists=reverse_adjacency_lists,
            generation_timestamp=datetime.now().isoformat(),
            version='1.0',
        )
        self.rag_data = rag_data

        logger.info(f'RAG serialization completed: {len(modules)} modules, {len(clusters)} clusters')
        return rag_data

    def _create_architecture_overview(self, system_name: str) -> ArchitectureOverview:
        """Create high-level architecture overview."""
        logger.info('Creating architecture overview...')

        # Basic metrics
        total_modules = len(self.graph.nodes)
        total_clusters = len(self.clustering_results.final_clusters)

        # Distribution analysis
        layer_distribution: defaultdict[str, int] = defaultdict(int)
        domain_distribution: defaultdict[str, int] = defaultdict(int)
        complexity_distribution: dict[str, int] = {'low': 0, 'medium': 0, 'high': 0}

        for node in self.graph.nodes.values():
            layer_distribution[node.layer] += 1
            domain_distribution[node.domain] += 1

            # Complexity classification
            complexity = node.complexity_metrics.get('cyclomatic_complexity', 0)
            if complexity < 5:
                complexity_distribution['low'] += 1
            elif complexity < 15:
                complexity_distribution['medium'] += 1
            else:
                complexity_distribution['high'] += 1

        # Identify patterns
        patterns = self._identify_architectural_patterns()
        principles = self._identify_design_principles()

        # Critical dependencies
        critical_deps = self._identify_critical_dependencies()
        circular_deps = self._identify_circular_dependencies()

        # Overall modularity
        modularity = self.clustering_results.quality_metrics.get('modularity', 0.0)

        # Generate narrative
        executive_summary = self._generate_executive_summary(
            system_name, total_modules, total_clusters, layer_distribution,
        )
        architecture_description = self._generate_architecture_description(
            layer_distribution, domain_distribution, patterns,
        )

        return ArchitectureOverview(
            system_name=system_name,
            total_modules=total_modules,
            total_clusters=total_clusters,
            layer_distribution=dict(layer_distribution),
            domain_distribution=dict(domain_distribution),
            architectural_patterns=patterns,
            design_principles=principles,
            critical_dependencies=critical_deps,
            circular_dependencies=circular_deps,
            overall_modularity=modularity,
            complexity_distribution=complexity_distribution,
            executive_summary=executive_summary,
            architecture_description=architecture_description,
        )

    def _create_module_summaries(self) -> List[ModuleSummary]:
        """Create summaries for all modules."""
        logger.info('Creating module summaries...')

        summaries = []

        for node_id, node in self.graph.nodes.items():
            # Generate purpose description
            purpose = self._generate_module_purpose(node)

            # Extract key functions
            key_functions = self._extract_key_functions(node)

            # Get dependencies and dependents
            dependencies = self.graph.get_neighbors(node_id)
            dependents = self.graph.get_predecessors(node_id)

            # Classify complexity
            complexity_level = self._classify_complexity(node)

            # Generate content chunks for RAG
            content_chunks = self._generate_content_chunks(node)

            # Extract keywords
            keywords = self._extract_keywords(node)

            # Centrality score
            centrality_score = node.centrality_measures.get('pagerank', 0.0)

            summary = ModuleSummary(
                id=node_id,
                name=os.path.basename(node.relative_path),
                file_path=node.relative_path,
                module_type=node.module_type,
                layer=node.layer,
                domain=node.domain,
                purpose=purpose,
                key_functions=key_functions,
                dependencies=dependencies,
                dependents=dependents,
                complexity_level=complexity_level,
                lines_of_code=node.lines_of_code,
                centrality_score=centrality_score,
                content_chunks=content_chunks,
                keywords=keywords,
                embedding=node.embedding,
            )

            summaries.append(summary)

        return summaries

    def _create_cluster_summaries(self) -> List[ClusterSummary]:
        """Create summaries for all clusters."""
        logger.info('Creating cluster summaries...')

        summaries = []

        for cluster_id, cluster in self.clustering_results.final_clusters.items():
            # Identify interfaces (modules that connect to other clusters)
            interfaces = self._identify_cluster_interfaces(cluster)

            # External dependencies
            external_deps = self._identify_external_dependencies(cluster)

            # Generate content summary
            content_summary = self._generate_cluster_content_summary(cluster)

            # Interaction patterns
            interaction_patterns = self._identify_interaction_patterns(cluster)

            # Generate cluster embedding (average of module embeddings)
            cluster_embedding = self._compute_cluster_embedding(cluster)

            summary = ClusterSummary(
                id=cluster_id,
                name=cluster.name,
                purpose=cluster.cluster_purpose,
                modules=cluster.modules,
                layer=cluster.dominant_layer,
                domain=cluster.dominant_domain,
                size=cluster.size,
                complexity_level=cluster.complexity_level,
                cohesion=cluster.internal_cohesion,
                coupling=cluster.external_coupling,
                interfaces=interfaces,
                external_dependencies=external_deps,
                content_summary=content_summary,
                interaction_patterns=interaction_patterns,
                embedding=cluster_embedding,
            )

            summaries.append(summary)

        return summaries

    def _generate_module_purpose(self, node: GraphNode) -> str:
        """Generate a purpose description for a module."""
        # Template-based generation
        templates = {
            ('core', 'authentication'): 'Handles user authentication and session management',
            ('core', 'data_access'): 'Provides data access and persistence functionality',
            ('interface', 'api_layer'): 'Exposes API endpoints for client interaction',
            ('utility', 'core'): 'Provides utility functions and helper methods',
            ('model', 'core'): 'Defines data models and schemas',
            ('infrastructure', 'database'): 'Manages database connections and operations',
            ('infrastructure', 'messaging'): 'Handles message queuing and event processing',
        }

        key = (node.module_type, node.domain)
        if key in templates:
            return templates[key]

        # Generic template
        return f"{node.module_type.replace('_', ' ').title()} module for {node.domain.replace('_', ' ')}"

    def _extract_key_functions(self, node: GraphNode) -> List[str]:
        """Extract key functions from module content."""
        # This is simplified - could be enhanced with AST analysis
        functions = []

        if node.content:
            lines = node.content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('def ') and '(' in line:
                    func_name = line.split('(')[0].replace('def ', '').strip()
                    if not func_name.startswith('_'):  # Public functions
                        functions.append(func_name)
                elif line.startswith('class ') and ':' in line:
                    class_name = line.split(':')[0].replace('class ', '').strip()
                    functions.append(class_name)

        return functions[:10]  # Limit to top 10

    def _classify_complexity(self, node: GraphNode) -> str:
        """Classify module complexity level."""
        complexity = node.complexity_metrics.get('cyclomatic_complexity', 0)
        loc = node.lines_of_code

        # Combined complexity score
        score = complexity + (loc / 100)

        if score < 5:
            return 'low'
        elif score < 15:
            return 'medium'
        else:
            return 'high'

    def _generate_content_chunks(self, node: GraphNode) -> List[str]:
        """Generate content chunks for RAG retrieval."""
        chunks = []

        # Add module metadata as searchable content
        metadata_chunk = f"""
        Module: {node.relative_path}
        Type: {node.module_type}
        Layer: {node.layer}
        Domain: {node.domain}
        Lines of Code: {node.lines_of_code}
        Complexity: {node.complexity_metrics.get('cyclomatic_complexity', 0)}
        """.strip()
        chunks.append(metadata_chunk)

        # Add content chunks if available
        if node.content:
            # Split content into smaller chunks (roughly 200 characters each)
            content = node.content
            chunk_size = 200
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())

        return chunks

    def _extract_keywords(self, node: GraphNode) -> List[str]:
        """Extract keywords from module for search indexing."""
        keywords = set()

        # Add semantic tags
        keywords.update(node.semantic_tags)

        # Add path components
        path_parts = node.relative_path.replace('.py', '').split('/')
        keywords.update(path_parts)

        # Add module characteristics
        keywords.update([node.module_type, node.layer, node.domain])

        # Extract from content (simple keyword extraction)
        if node.content:
            # Find class and function names
            lines = node.content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('def ', 'class ')):
                    name = line.split('(')[0].split(':')[0]
                    name = name.replace('def ', '').replace('class ', '')
                    keywords.add(name)

        return list(keywords)

    def _identify_cluster_interfaces(self, cluster: ModuleCluster) -> List[str]:
        """Identify modules that serve as interfaces to other clusters."""
        interfaces = []

        for module_id in cluster.modules:
            # Check if module has connections outside the cluster
            neighbors = self.graph.get_neighbors(module_id)
            external_connections = [n for n in neighbors if n not in cluster.modules]

            if external_connections:
                interfaces.append(module_id)

        return interfaces

    def _identify_external_dependencies(self, cluster: ModuleCluster) -> List[str]:
        """Identify dependencies outside the cluster."""
        external_deps = set()

        for module_id in cluster.modules:
            dependencies = self.graph.get_neighbors(module_id)
            for dep in dependencies:
                if dep not in cluster.modules:
                    external_deps.add(dep)

        return list(external_deps)

    def _generate_cluster_content_summary(self, cluster: ModuleCluster) -> str:
        """Generate a content summary for the cluster."""
        return f"""
        Cluster: {cluster.name}
        Purpose: {cluster.cluster_purpose}
        Size: {cluster.size} modules
        Layer: {cluster.dominant_layer}
        Domain: {cluster.dominant_domain}
        Complexity: {cluster.complexity_level}
        Cohesion: {cluster.internal_cohesion:.2f}
        Coupling: {cluster.external_coupling:.2f}
        """.strip()

    def _identify_interaction_patterns(self, cluster: ModuleCluster) -> List[str]:
        """Identify common interaction patterns within the cluster."""
        patterns = []

        # Analyze internal connections
        internal_edges = 0
        for source in cluster.modules:
            for target in cluster.modules:
                if source != target and (source, target) in self.graph.edges:
                    internal_edges += 1

        if internal_edges > cluster.size:
            patterns.append('High internal connectivity')

        # Check for hub pattern
        hub_threshold = cluster.size // 2
        for module_id in cluster.modules:
            connections = len([n for n in self.graph.get_neighbors(module_id) if n in cluster.modules])
            if connections > hub_threshold:
                patterns.append(f'Hub pattern around {module_id}')

        return patterns

    def _compute_cluster_embedding(self, cluster: ModuleCluster) -> Optional[List[float]]:
        """Compute cluster embedding as average of module embeddings."""
        embeddings = []

        for module_id in cluster.modules:
            if module_id in self.graph.nodes:
                node = self.graph.nodes[module_id]
                if node.embedding:
                    embeddings.append(node.embedding)

        if not embeddings:
            return None

        # Compute average embedding
        import numpy as np
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding.tolist()

    def _build_content_index(self, modules: List[ModuleSummary], clusters: List[ClusterSummary]) -> Dict[str, List[str]]:
        """Build content-based retrieval index."""
        index = defaultdict(list)

        # Index modules by keywords
        for module in modules:
            for keyword in module.keywords:
                index[keyword.lower()].append(f'module:{module.id}')

            # Index by content chunks
            for i, chunk in enumerate(module.content_chunks):
                words = chunk.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        index[word].append(f'module:{module.id}:chunk:{i}')

        # Index clusters
        for cluster in clusters:
            words = cluster.content_summary.lower().split()
            for word in words:
                if len(word) > 3:
                    index[word].append(f'cluster:{cluster.id}')

        return dict(index)

    def _build_dependency_index(self, modules: List[ModuleSummary]) -> Dict[str, List[str]]:
        """Build dependency-based retrieval index."""
        index = defaultdict(list)

        for module in modules:
            # Index by dependencies
            for dep in module.dependencies:
                index[dep].append(f'dependent:{module.id}')

            # Index by dependents
            for dependent in module.dependents:
                index[dependent].append(f'dependency:{module.id}')

        return dict(index)

    def _build_keyword_index(self, modules: List[ModuleSummary], clusters: List[ClusterSummary]) -> Dict[str, List[str]]:
        """Build keyword-based retrieval index."""
        index = defaultdict(list)

        # Index modules
        for module in modules:
            for keyword in module.keywords:
                index[keyword.lower()].append(f'module:{module.id}')

            # Index by attributes
            index[module.layer.lower()].append(f'module:{module.id}')
            index[module.domain.lower()].append(f'module:{module.id}')
            index[module.module_type.lower()].append(f'module:{module.id}')

        # Index clusters
        for cluster in clusters:
            index[cluster.layer.lower()].append(f'cluster:{cluster.id}')
            index[cluster.domain.lower()].append(f'cluster:{cluster.id}')
            index[cluster.name.lower()].append(f'cluster:{cluster.id}')

        return dict(index)

    def _build_adjacency_data(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Build adjacency lists for graph traversal."""
        adjacency_lists = {}
        reverse_adjacency_lists = {}

        for node_id in self.graph.nodes.keys():
            adjacency_lists[node_id] = self.graph.get_neighbors(node_id)
            reverse_adjacency_lists[node_id] = self.graph.get_predecessors(node_id)

        return adjacency_lists, reverse_adjacency_lists

    def _identify_architectural_patterns(self) -> List[str]:
        """Identify architectural patterns in the system."""
        patterns = []

        # Layered architecture
        layers = {node.layer for node in self.graph.nodes.values()}
        if len(layers) >= 3:
            patterns.append('Layered Architecture')

        # Domain-driven design
        domains = {node.domain for node in self.graph.nodes.values()}
        if len(domains) >= 4:
            patterns.append('Domain-Driven Design')

        # Service-oriented
        service_modules = [n for n in self.graph.nodes.values() if 'service' in n.module_type.lower()]
        if len(service_modules) > 3:
            patterns.append('Service-Oriented Architecture')

        return patterns

    def _identify_design_principles(self) -> List[str]:
        """Identify design principles evident in the architecture."""
        principles = []

        # Separation of concerns
        unique_domains = len({node.domain for node in self.graph.nodes.values()})
        if unique_domains > 3:
            principles.append('Separation of Concerns')

        # Single responsibility
        avg_complexity = sum(
            node.complexity_metrics.get('cyclomatic_complexity', 0)
            for node in self.graph.nodes.values()
        ) / len(self.graph.nodes)

        if avg_complexity < 10:
            principles.append('Single Responsibility Principle')

        # Dependency inversion
        interface_modules = [n for n in self.graph.nodes.values() if n.module_type == 'interface']
        if len(interface_modules) > 2:
            principles.append('Dependency Inversion')

        return principles

    def _identify_critical_dependencies(self) -> List[str]:
        """Identify critical dependencies in the system."""
        critical_deps = []

        # High centrality nodes
        for node_id, node in self.graph.nodes.items():
            pagerank = node.centrality_measures.get('pagerank', 0)
            if pagerank > 0.1:  # High PageRank indicates importance
                critical_deps.append(node_id)

        return critical_deps[:10]  # Top 10 critical dependencies

    def _identify_circular_dependencies(self) -> List[str]:
        """Identify circular dependencies."""
        # This is simplified - could use more sophisticated cycle detection
        circular_deps = []

        # Look for mutual dependencies
        for (source, target), edge in self.graph.edges.items():
            reverse_key = (target, source)
            if reverse_key in self.graph.edges:
                circular_deps.append(f'{source} <-> {target}')

        return circular_deps[:10]  # Limit results

    def _generate_executive_summary(
        self, system_name: str, total_modules: int,
        total_clusters: int, layer_distribution: Dict[str, int],
    ) -> str:
        """Generate executive summary of the architecture."""
        return f"""
        {system_name} is a {total_modules}-module system organized into {total_clusters} logical clusters.
        The architecture follows a {len(layer_distribution)}-layer design with {layer_distribution.get('business', 0)} business logic modules,
        {layer_distribution.get('data', 0)} data access modules, and {layer_distribution.get('presentation', 0)} presentation modules.
        The system demonstrates good modular design with clear separation of concerns across different domains.
        """.strip()

    def _generate_architecture_description(
        self, layer_distribution: Dict[str, int],
        domain_distribution: Dict[str, int], patterns: List[str],
    ) -> str:
        """Generate detailed architecture description."""
        description_parts = [
            f'The system is organized into {len(layer_distribution)} architectural layers:',
        ]

        for layer, count in layer_distribution.items():
            description_parts.append(f'- {layer.title()} Layer: {count} modules')

        description_parts.append(f'\nThe system spans {len(domain_distribution)} functional domains:')

        for domain, count in sorted(domain_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
            description_parts.append(f"- {domain.replace('_', ' ').title()}: {count} modules")

        if patterns:
            description_parts.append(f"\nArchitectural patterns identified: {', '.join(patterns)}")

        return '\n'.join(description_parts)


def save_rag_data(rag_data: RAGOptimizedData, output_path: str):
    """Save RAG-optimized data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_data.to_dict(), f, indent=2)

    logger.info(f'RAG-optimized data saved to {output_path}')


def load_rag_data(input_path: str) -> RAGOptimizedData:
    """Load RAG-optimized data from JSON file."""
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    # Reconstruct objects
    overview = ArchitectureOverview(**data['overview'])
    modules = [ModuleSummary(**module_data) for module_data in data['modules']]
    clusters = [ClusterSummary(**cluster_data) for cluster_data in data['clusters']]

    rag_data = RAGOptimizedData(
        overview=overview,
        modules=modules,
        clusters=clusters,
        content_index=data.get('content_index', {}),
        dependency_index=data.get('dependency_index', {}),
        keyword_index=data.get('keyword_index', {}),
        adjacency_lists=data.get('adjacency_lists', {}),
        reverse_adjacency_lists=data.get('reverse_adjacency_lists', {}),
        generation_timestamp=data.get('generation_timestamp', ''),
        version=data.get('version', '1.0'),
    )

    logger.info(f'Loaded RAG data with {len(modules)} modules and {len(clusters)} clusters')
    return rag_data
