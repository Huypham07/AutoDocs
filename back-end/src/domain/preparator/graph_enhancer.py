"""
Graph enhancement module for computing metrics and enriching graph data.

This module enhances the multi-modal graph with:
- Centrality measures (betweenness, degree, PageRank)
- Semantic embeddings
- Enhanced classification
- Edge weight computation
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
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
from .graph_builder import RelationshipType

logger = get_logger(__name__)

# Optional dependencies for advanced features
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning('NetworkX not available, using simplified centrality calculations')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning('Scikit-learn not available, using simplified similarity calculations')


@dataclass
class GraphMetrics:
    """Container for graph-level metrics."""
    total_nodes: int
    total_edges: int
    density: float
    average_degree: float
    max_degree: int
    strongly_connected_components: int
    diameter: Optional[int] = None
    clustering_coefficient: float = 0.0


class GraphEnhancer:
    """
    Enhances multi-modal graphs with computed metrics and enriched metadata.
    """

    def __init__(self, graph: MultiModalGraph):
        self.graph = graph
        self.nx_graph: Optional[nx.DiGraph] = None
        self.embedding_model = None
        self._build_networkx_graph()

    def enhance_graph(self) -> MultiModalGraph:
        """
        Enhance the graph with computed metrics and enriched data.

        Returns:
            MultiModalGraph: Enhanced graph with computed metrics
        """
        logger.info('Enhancing graph with computed metrics...')

        # Step 1: Compute centrality measures
        self._compute_centrality_measures()

        # Step 2: Generate semantic embeddings
        self._generate_semantic_embeddings()

        # Step 3: Compute edge weights
        self._compute_edge_weights()

        # Step 4: Enhanced classification
        self._enhance_classification()

        # Step 5: Compute semantic similarities
        self._compute_semantic_similarities()

        logger.info('Graph enhancement completed')
        return self.graph

    def _build_networkx_graph(self):
        """Build NetworkX graph for centrality calculations."""
        if not NETWORKX_AVAILABLE:
            return

        self.nx_graph = nx.DiGraph()

        # Add nodes
        for node_id in self.graph.nodes.keys():
            self.nx_graph.add_node(node_id)

        # Add edges with weights
        for (source, target), edge in self.graph.edges.items():
            weight = edge.interaction_weight if edge.interaction_weight > 0 else 0.1
            self.nx_graph.add_edge(source, target, weight=weight)

    def _compute_centrality_measures(self):
        """Compute various centrality measures for nodes."""
        logger.info('Computing centrality measures...')

        if NETWORKX_AVAILABLE and self.nx_graph:
            self._compute_networkx_centrality()
        else:
            self._compute_simple_centrality()

    def _compute_networkx_centrality(self):
        """Compute centrality using NetworkX."""
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.nx_graph)

            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.nx_graph)

            # PageRank
            pagerank = nx.pagerank(self.nx_graph, weight='weight')

            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(self.nx_graph)

            # In-degree and out-degree
            in_degree = dict(self.nx_graph.in_degree())
            out_degree = dict(self.nx_graph.out_degree())

            # Update node centrality measures
            for node_id, node in self.graph.nodes.items():
                node.centrality_measures = {
                    'degree': degree_centrality.get(node_id, 0.0),
                    'betweenness': betweenness_centrality.get(node_id, 0.0),
                    'pagerank': pagerank.get(node_id, 0.0),
                    'closeness': closeness_centrality.get(node_id, 0.0),
                    'in_degree': in_degree.get(node_id, 0),
                    'out_degree': out_degree.get(node_id, 0),
                    'total_degree': in_degree.get(node_id, 0) + out_degree.get(node_id, 0),
                }

            logger.info('NetworkX centrality measures computed successfully')

        except Exception as e:
            logger.warning(f'Error computing NetworkX centrality: {e}, falling back to simple centrality')
            self._compute_simple_centrality()

    def _compute_simple_centrality(self):
        """Compute simplified centrality measures without NetworkX."""
        logger.info('Computing simplified centrality measures...')

        total_nodes = len(self.graph.nodes)

        for node_id, node in self.graph.nodes.items():
            # Simple degree centrality
            neighbors = self.graph.get_neighbors(node_id)
            predecessors = self.graph.get_predecessors(node_id)

            in_degree = len(predecessors)
            out_degree = len(neighbors)
            total_degree = in_degree + out_degree

            # Normalized degree centrality
            degree_centrality = total_degree / max(total_nodes - 1, 1)

            # Simple PageRank approximation based on in-degree
            pagerank = (in_degree + 1) / (total_nodes + 1)

            node.centrality_measures = {
                'degree': degree_centrality,
                'betweenness': 0.0,  # Not computed in simple mode
                'pagerank': pagerank,
                'closeness': 0.0,   # Not computed in simple mode
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': total_degree,
            }

    def _generate_semantic_embeddings(self):
        """Generate semantic embeddings for nodes."""
        logger.info('Generating semantic embeddings...')

        if SKLEARN_AVAILABLE:
            self._generate_tfidf_embeddings()
        else:
            self._generate_simple_embeddings()

    def _generate_tfidf_embeddings(self):
        """Generate TF-IDF based embeddings."""
        try:
            # Collect content from all nodes
            documents = []
            node_ids = []

            for node_id, node in self.graph.nodes.items():
                # Combine content with semantic tags and metadata
                content_parts = [
                    node.content,
                    ' '.join(node.semantic_tags),
                    node.module_type,
                    node.domain,
                    node.layer,
                    node.relative_path,
                ]
                content = ' '.join(filter(None, content_parts))
                documents.append(content)
                node_ids.append(node_id)

            # Generate TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=384,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )

            tfidf_matrix = vectorizer.fit_transform(documents)

            # Store embeddings in nodes
            for i, node_id in enumerate(node_ids):
                embedding = tfidf_matrix[i].toarray().flatten()
                self.graph.nodes[node_id].embedding = embedding.tolist()

            logger.info(f'Generated TF-IDF embeddings for {len(node_ids)} nodes')

        except Exception as e:
            logger.warning(f'Error generating TF-IDF embeddings: {e}, falling back to simple embeddings')
            self._generate_simple_embeddings()

    def _generate_simple_embeddings(self):
        """Generate simple embeddings based on features."""
        logger.info('Generating simple feature-based embeddings...')

        # Create simple embeddings based on available features
        for node_id, node in self.graph.nodes.items():
            # Create a simple feature vector
            features = [
                node.lines_of_code / 1000.0,  # Normalize LOC
                node.complexity_metrics.get('cyclomatic_complexity', 0) / 20.0,
                node.complexity_metrics.get('function_count', 0) / 10.0,
                node.complexity_metrics.get('class_count', 0) / 5.0,
                node.centrality_measures.get('degree', 0),
                node.centrality_measures.get('pagerank', 0) * 10,
                len(node.semantic_tags) / 10.0,
            ]

            # One-hot encode categorical features
            module_types = ['core', 'utility', 'interface', 'config', 'model', 'test', 'infrastructure']
            for mt in module_types:
                features.append(1.0 if node.module_type == mt else 0.0)

            layers = ['presentation', 'business', 'data', 'infrastructure']
            for layer in layers:
                features.append(1.0 if node.layer == layer else 0.0)

            # Pad to fixed size (32 dimensions)
            while len(features) < 32:
                features.append(0.0)
            features = features[:32]  # Truncate if too long

            node.embedding = features

    def _compute_edge_weights(self):
        """Compute enhanced edge weights based on multiple factors."""
        logger.info('Computing enhanced edge weights...')

        # Validate edges before processing
        invalid_edges = []
        for edge_key, edge in self.graph.edges.items():
            if edge.source not in self.graph.nodes or edge.target not in self.graph.nodes:
                invalid_edges.append(edge_key)
                logger.warning(f'Invalid edge found: {edge.source} -> {edge.target} (missing nodes)')

        # Remove invalid edges
        for edge_key in invalid_edges:
            del self.graph.edges[edge_key]

        if invalid_edges:
            logger.info(f'Removed {len(invalid_edges)} invalid edges')

        for edge_key, edge in self.graph.edges.items():
            source_node = self.graph.nodes[edge.source]
            target_node = self.graph.nodes[edge.target]

            # Base weight from relationship types
            base_weight = self._compute_relationship_weight(edge.relationship_types)

            # Adjust based on centrality
            centrality_factor = (
                source_node.centrality_measures.get('pagerank', 0) +
                target_node.centrality_measures.get('pagerank', 0)
            ) / 2.0

            # Adjust based on complexity similarity
            complexity_factor = self._compute_complexity_similarity(source_node, target_node)

            # Adjust based on domain similarity
            domain_factor = 1.2 if source_node.domain == target_node.domain else 0.8

            # Adjust based on layer proximity
            layer_factor = self._compute_layer_proximity(source_node.layer, target_node.layer)

            # Compute final weight
            final_weight = base_weight * (1 + centrality_factor) * complexity_factor * domain_factor * layer_factor
            final_weight = min(final_weight, 1.0)  # Cap at 1.0

            edge.interaction_weight = final_weight

            # Determine dependency strength
            if final_weight > 0.7:
                edge.dependency_strength = 'strong'
            elif final_weight > 0.4:
                edge.dependency_strength = 'medium'
            else:
                edge.dependency_strength = 'weak'

    def _compute_relationship_weight(self, relationship_types: List[RelationshipType]) -> float:
        """Compute weight based on relationship types."""
        type_weights = {
            RelationshipType.INHERITS: 0.9,
            RelationshipType.COMPOSES: 0.8,
            RelationshipType.CALLS: 0.7,
            RelationshipType.IMPORTS: 0.5,
            RelationshipType.DEPENDENCY: 0.6,
            RelationshipType.SEMANTIC: 0.4,
            RelationshipType.STRUCTURAL: 0.3,
        }

        if not relationship_types:
            return 0.1

        # Use maximum weight if multiple relationship types
        return max(type_weights.get(rt, 0.1) for rt in relationship_types)

    def _compute_complexity_similarity(self, node1: GraphNode, node2: GraphNode) -> float:
        """Compute similarity based on complexity metrics."""
        metrics1 = node1.complexity_metrics
        metrics2 = node2.complexity_metrics

        # Compare cyclomatic complexity
        cc1 = metrics1.get('cyclomatic_complexity', 0)
        cc2 = metrics2.get('cyclomatic_complexity', 0)

        if cc1 == 0 and cc2 == 0:
            cc_similarity = 1.0
        else:
            cc_similarity = 1.0 - abs(cc1 - cc2) / max(cc1 + cc2, 1)

        # Compare lines of code
        loc1 = node1.lines_of_code
        loc2 = node2.lines_of_code

        if loc1 == 0 and loc2 == 0:
            loc_similarity = 1.0
        else:
            loc_similarity = 1.0 - abs(loc1 - loc2) / max(loc1 + loc2, 1)

        return (cc_similarity + loc_similarity) / 2.0

    def _compute_layer_proximity(self, layer1: str, layer2: str) -> float:
        """Compute proximity factor based on architectural layers."""
        layer_order = ['presentation', 'business', 'data', 'infrastructure']

        try:
            idx1 = layer_order.index(layer1)
            idx2 = layer_order.index(layer2)
            distance = abs(idx1 - idx2)

            # Closer layers have higher proximity
            if distance == 0:
                return 1.2
            elif distance == 1:
                return 1.0
            elif distance == 2:
                return 0.8
            else:
                return 0.6
        except ValueError:
            return 1.0  # Default if layers not in standard order

    def _enhance_classification(self):
        """Enhance module classification based on graph structure."""
        logger.info('Enhancing module classification...')

        for node_id, node in self.graph.nodes.items():
            # Enhance classification based on centrality
            if node.centrality_measures.get('pagerank', 0) > 0.1:
                if 'central' not in node.semantic_tags:
                    node.semantic_tags.append('central')

            if node.centrality_measures.get('betweenness', 0) > 0.1:
                if 'bridge' not in node.semantic_tags:
                    node.semantic_tags.append('bridge')

            # Enhance based on connectivity
            in_degree = node.centrality_measures.get('in_degree', 0)
            out_degree = node.centrality_measures.get('out_degree', 0)

            if in_degree > 5:
                if 'highly_used' not in node.semantic_tags:
                    node.semantic_tags.append('highly_used')

            if out_degree > 5:
                if 'highly_dependent' not in node.semantic_tags:
                    node.semantic_tags.append('highly_dependent')

            # Refine module type based on patterns
            if node.module_type == 'core' and in_degree > out_degree * 2:
                node.module_type = 'service'

            if out_degree > in_degree * 2 and node.module_type != 'interface':
                node.module_type = 'coordinator'

    def _compute_semantic_similarities(self):
        """Compute semantic similarities between nodes and add to edges."""
        logger.info('Computing semantic similarities...')

        if not SKLEARN_AVAILABLE:
            logger.warning('Scikit-learn not available, skipping semantic similarity computation')
            return

        # Get all embeddings
        node_ids = list(self.graph.nodes.keys())
        embeddings = []

        for node_id in node_ids:
            embedding = self.graph.nodes[node_id].embedding
            if embedding:
                embeddings.append(embedding)
            else:
                # Create zero embedding as fallback
                embeddings.append([0.0] * 32)

        if not embeddings:
            return

        try:
            # Compute pairwise cosine similarities
            similarity_matrix = cosine_similarity(embeddings)

            # Update edges with semantic similarities
            for i, source_id in enumerate(node_ids):
                for j, target_id in enumerate(node_ids):
                    if i != j:
                        edge_key = (source_id, target_id)
                        if edge_key in self.graph.edges:
                            similarity = float(similarity_matrix[i][j])
                            self.graph.edges[edge_key].semantic_similarity = similarity

                            # Add semantic relationship if similarity is high
                            if similarity > 0.7:
                                edge = self.graph.edges[edge_key]
                                if RelationshipType.SEMANTIC not in edge.relationship_types:
                                    edge.relationship_types.append(RelationshipType.SEMANTIC)

            logger.info('Semantic similarities computed successfully')

        except Exception as e:
            logger.warning(f'Error computing semantic similarities: {e}')

    def get_graph_metrics(self) -> GraphMetrics:
        """Compute and return overall graph metrics."""
        total_nodes = len(self.graph.nodes)
        total_edges = len(self.graph.edges)

        # Compute density
        max_possible_edges = total_nodes * (total_nodes - 1)
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        # Compute average degree
        total_degree = sum(
            node.centrality_measures.get('total_degree', 0)
            for node in self.graph.nodes.values()
        )
        average_degree = total_degree / total_nodes if total_nodes > 0 else 0.0

        # Find maximum degree
        max_degree = int(
            max(
                (node.centrality_measures.get('total_degree', 0) for node in self.graph.nodes.values()),
                default=0,
            ),
        )

        # Use NetworkX for advanced metrics if available
        diameter = None
        clustering_coefficient = 0.0
        strongly_connected_components = 1

        if NETWORKX_AVAILABLE and self.nx_graph:
            try:
                if nx.is_weakly_connected(self.nx_graph):
                    undirected = self.nx_graph.to_undirected()
                    diameter = nx.diameter(undirected)

                clustering_coefficient = nx.average_clustering(self.nx_graph.to_undirected())
                strongly_connected_components = nx.number_strongly_connected_components(self.nx_graph)

            except Exception as e:
                logger.warning(f'Error computing advanced graph metrics: {e}')

        return GraphMetrics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            density=density,
            average_degree=average_degree,
            max_degree=max_degree,
            diameter=diameter,
            clustering_coefficient=clustering_coefficient,
            strongly_connected_components=strongly_connected_components,
        )

    def identify_architectural_patterns(self) -> Dict[str, List[str]]:
        """Identify common architectural patterns in the graph."""
        patterns: Dict[str, List[str]] = {
            'hubs': [],
            'bridges': [],
            'singletons': [],
            'circular_dependencies': [],
            'layer_violations': [],
        }

        # Identify hubs (high in-degree)
        for node_id, node in self.graph.nodes.items():
            in_degree = node.centrality_measures.get('in_degree', 0)
            if in_degree > 5:
                patterns['hubs'].append(node_id)

        # Identify bridges (high betweenness centrality)
        for node_id, node in self.graph.nodes.items():
            betweenness = node.centrality_measures.get('betweenness', 0)
            if betweenness > 0.1:
                patterns['bridges'].append(node_id)

        # Identify singletons (no connections)
        for node_id, node in self.graph.nodes.items():
            total_degree = node.centrality_measures.get('total_degree', 0)
            if total_degree == 0:
                patterns['singletons'].append(node_id)

        # Detect circular dependencies using NetworkX if available
        if NETWORKX_AVAILABLE and self.nx_graph:
            try:
                cycles = list(nx.simple_cycles(self.nx_graph))
                for cycle in cycles[:10]:  # Limit to first 10 cycles
                    patterns['circular_dependencies'].append(' -> '.join(cycle))
            except Exception as e:
                logger.warning(f'Error detecting cycles: {e}')

        # Identify layer violations
        layer_order = {'presentation': 0, 'business': 1, 'data': 2, 'infrastructure': 3}

        for (source, target), edge in self.graph.edges.items():
            source_node = self.graph.nodes[source]
            target_node = self.graph.nodes[target]

            source_layer_idx = layer_order.get(source_node.layer, 1)
            target_layer_idx = layer_order.get(target_node.layer, 1)

            # Check for upward dependencies (violation)
            if source_layer_idx > target_layer_idx:
                violation = f'{source} -> {target} ({source_node.layer} -> {target_node.layer})'
                patterns['layer_violations'].append(violation)

        return patterns


def enhance_graph(graph: MultiModalGraph) -> MultiModalGraph:
    """
    Convenience function to enhance a graph with all available metrics.

    Args:
        graph: The graph to enhance

    Returns:
        Enhanced graph
    """
    enhancer = GraphEnhancer(graph)
    return enhancer.enhance_graph()


def save_enhanced_graph(graph: MultiModalGraph, output_path: str, include_metrics: bool = True):
    """
    Save enhanced graph with optional metrics report.

    Args:
        graph: Enhanced graph to save
        output_path: Output file path
        include_metrics: Whether to include a metrics report
    """
    import os

    # Save the graph
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph.to_dict(), f, indent=2)

    if include_metrics:
        # Generate and save metrics report
        enhancer = GraphEnhancer(graph)
        metrics = enhancer.get_graph_metrics()
        patterns = enhancer.identify_architectural_patterns()

        metrics_path = output_path.replace('.json', '_metrics.json')
        metrics_report = {
            'graph_metrics': {
                'total_nodes': metrics.total_nodes,
                'total_edges': metrics.total_edges,
                'density': metrics.density,
                'average_degree': metrics.average_degree,
                'max_degree': metrics.max_degree,
                'diameter': metrics.diameter,
                'clustering_coefficient': metrics.clustering_coefficient,
                'strongly_connected_components': metrics.strongly_connected_components,
            },
            'architectural_patterns': patterns,
        }

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_report, f, indent=2)

        logger.info(f'Saved enhanced graph to {output_path}')
        logger.info(f'Saved metrics report to {metrics_path}')

    logger.info('Enhanced graph saved successfully')
