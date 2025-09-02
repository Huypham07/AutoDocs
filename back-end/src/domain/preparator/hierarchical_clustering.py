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
from typing import Union

import numpy as np
from shared.logging import get_logger

from .graph_builder import GraphEdge
from .graph_builder import GraphNode
from .graph_builder import MultiModalGraph

logger = get_logger(__name__)

# Optional dependencies
try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning('NetworkX not available, using simplified clustering')

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning('Scikit-learn not available, using simplified clustering')


class ClusteringMethod(Enum):
    """Available clustering methods."""
    COMMUNITY_DETECTION = 'community_detection'
    SEMANTIC_CLUSTERING = 'semantic_clustering'
    DIRECTORY_CLUSTERING = 'directory_clustering'
    COMPLEXITY_CLUSTERING = 'complexity_clustering'
    HYBRID_CLUSTERING = 'hybrid_clustering'


@dataclass
class ModuleCluster:
    """Represents a cluster of related modules."""
    id: str
    name: str
    modules: List[str] = field(default_factory=list)
    cluster_purpose: str = ''

    # Quality metrics
    internal_cohesion: float = 0.0
    external_coupling: float = 0.0
    size: int = 0

    # Cluster characteristics
    dominant_layer: str = ''
    dominant_domain: str = ''
    dominant_type: str = ''
    complexity_level: str = 'medium'  # low, medium, high

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'modules': self.modules,
            'cluster_purpose': self.cluster_purpose,
            'internal_cohesion': self.internal_cohesion,
            'external_coupling': self.external_coupling,
            'size': self.size,
            'dominant_layer': self.dominant_layer,
            'dominant_domain': self.dominant_domain,
            'dominant_type': self.dominant_type,
            'complexity_level': self.complexity_level,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModuleCluster:
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            modules=data.get('modules', []),
            cluster_purpose=data.get('cluster_purpose', ''),
            internal_cohesion=data.get('internal_cohesion', 0.0),
            external_coupling=data.get('external_coupling', 0.0),
            size=data.get('size', 0),
            dominant_layer=data.get('dominant_layer', ''),
            dominant_domain=data.get('dominant_domain', ''),
            dominant_type=data.get('dominant_type', ''),
            complexity_level=data.get('complexity_level', 'medium'),
            metadata=data.get('metadata', {}),
        )


@dataclass
class ClusteringResults:
    """Container for clustering results."""
    final_clusters: Dict[str, ModuleCluster] = field(default_factory=dict)
    cluster_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    method_results: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'final_clusters': {cid: cluster.to_dict() for cid, cluster in self.final_clusters.items()},
            'cluster_hierarchy': self.cluster_hierarchy,
            'quality_metrics': self.quality_metrics,
            'method_results': self.method_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClusteringResults:
        """Create from dictionary."""
        final_clusters = {
            cid: ModuleCluster.from_dict(cluster_data)
            for cid, cluster_data in data.get('final_clusters', {}).items()
        }

        return cls(
            final_clusters=final_clusters,
            cluster_hierarchy=data.get('cluster_hierarchy', {}),
            quality_metrics=data.get('quality_metrics', {}),
            method_results=data.get('method_results', {}),
        )


class HierarchicalClusterer:
    """
    Performs hierarchical clustering on the enhanced graph to group related modules.
    """

    def __init__(self, graph: MultiModalGraph, target_clusters: int = 10):
        self.graph = graph
        self.target_clusters = target_clusters
        self.clustering_results = ClusteringResults()
        self.nx_graph: Optional[nx.Graph] = None
        self._build_networkx_graph()

    def cluster_modules(self) -> ClusteringResults:
        """
        Perform hierarchical clustering using multiple methods and consensus.

        Returns:
            ClusteringResults: Comprehensive clustering results
        """
        logger.info(f'Starting hierarchical clustering with target of {self.target_clusters} clusters...')

        # Step 1: Apply different clustering methods
        clustering_methods = {
            ClusteringMethod.COMMUNITY_DETECTION: self._community_detection_clustering,
            ClusteringMethod.SEMANTIC_CLUSTERING: self._semantic_clustering,
            ClusteringMethod.DIRECTORY_CLUSTERING: self._directory_clustering,
            ClusteringMethod.COMPLEXITY_CLUSTERING: self._complexity_clustering,
        }

        method_results = {}
        for method, clustering_func in clustering_methods.items():
            try:
                result = clustering_func()
                method_results[method.value] = result
                logger.info(f'{method.value} produced {len(result)} clusters')
            except Exception as e:
                logger.warning(f'Error in {method.value}: {e}')
                method_results[method.value] = {}

        self.clustering_results.method_results = method_results

        # Step 2: Create consensus clustering
        consensus_clusters = self._create_consensus_clustering(method_results)

        # Step 3: Build final clusters with metadata
        final_clusters = self._build_final_clusters(consensus_clusters)

        # Step 4: Compute quality metrics
        quality_metrics = self._compute_quality_metrics(final_clusters)

        # Step 5: Create hierarchy
        hierarchy = self._create_cluster_hierarchy(final_clusters)

        # Store results
        self.clustering_results.final_clusters = final_clusters
        self.clustering_results.quality_metrics = quality_metrics
        self.clustering_results.cluster_hierarchy = hierarchy

        logger.info(f'Hierarchical clustering completed with {len(final_clusters)} final clusters')
        return self.clustering_results

    def _build_networkx_graph(self):
        """Build undirected NetworkX graph for community detection."""
        if not NETWORKX_AVAILABLE:
            return

        self.nx_graph = nx.Graph()

        # Add nodes
        for node_id in self.graph.nodes.keys():
            self.nx_graph.add_node(node_id)

        # Add edges (convert to undirected)
        for (source, target), edge in self.graph.edges.items():
            weight = edge.interaction_weight
            if self.nx_graph.has_edge(source, target):
                # Combine weights if edge exists
                existing_weight = self.nx_graph[source][target].get('weight', 0)
                weight = max(weight, existing_weight)

            self.nx_graph.add_edge(source, target, weight=weight)

    def _community_detection_clustering(self) -> Dict[str, List[str]]:
        """Perform community detection clustering."""
        logger.info('Performing community detection clustering...')

        if not NETWORKX_AVAILABLE or not self.nx_graph:
            return self._simple_connectivity_clustering()

        try:
            # Try multiple community detection algorithms
            algorithms = [
                ('louvain', lambda g: community.louvain_communities(g, weight='weight')),
                ('greedy_modularity', lambda g: community.greedy_modularity_communities(g, weight='weight')),
                ('label_propagation', lambda g: community.asyn_lpa_communities(g, weight='weight')),
            ]

            best_communities = None
            best_modularity = -1

            for name, algorithm in algorithms:
                try:
                    communities = algorithm(self.nx_graph)
                    modularity = community.modularity(self.nx_graph, communities, weight='weight')

                    logger.info(f'{name} modularity: {modularity:.3f}, communities: {len(communities)}')

                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_communities = communities

                except Exception as e:
                    logger.warning(f'Error in {name}: {e}')

            if best_communities:
                # Convert to cluster format
                clusters = {}
                for i, community_set in enumerate(best_communities):
                    cluster_id = f'community_{i}'
                    clusters[cluster_id] = list(community_set)

                return clusters

        except Exception as e:
            logger.warning(f'Community detection failed: {e}')

        return self._simple_connectivity_clustering()

    def _simple_connectivity_clustering(self) -> Dict[str, List[str]]:
        """Simple connectivity-based clustering as fallback."""
        logger.info('Using simple connectivity clustering...')

        clusters = {}
        visited = set()
        cluster_id = 0

        for node_id in self.graph.nodes.keys():
            if node_id not in visited:
                # BFS to find connected components
                cluster = []
                queue = [node_id]
                visited.add(node_id)

                while queue:
                    current = queue.pop(0)
                    cluster.append(current)

                    # Add neighbors
                    neighbors = self.graph.get_neighbors(current) + self.graph.get_predecessors(current)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if cluster:
                    clusters[f'connectivity_{cluster_id}'] = cluster
                    cluster_id += 1

        return clusters

    def _semantic_clustering(self) -> Dict[str, List[str]]:
        """Perform semantic clustering based on embeddings."""
        logger.info('Performing semantic clustering...')

        if not SKLEARN_AVAILABLE:
            return self._simple_semantic_clustering()

        # Collect embeddings
        node_ids = []
        embeddings = []

        for node_id, node in self.graph.nodes.items():
            if node.embedding:
                node_ids.append(node_id)
                embeddings.append(node.embedding)

        if len(embeddings) < 2:
            logger.warning('Not enough embeddings for semantic clustering')
            return {}

        try:
            # Standardize embeddings
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)

            # Try different numbers of clusters
            best_score = -1
            best_labels = None
            best_k = 2

            for k in range(2, min(len(embeddings) // 2, self.target_clusters + 5)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings_scaled)

                    # Only compute silhouette score if we have more than 1 cluster
                    if len(set(labels)) > 1:
                        score = silhouette_score(embeddings_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_k = k

                except Exception as e:
                    logger.warning(f'K-means with k={k} failed: {e}')

            if best_labels is not None:
                # Convert to cluster format
                clusters = defaultdict(list)
                for i, label in enumerate(best_labels):
                    clusters[f'semantic_{label}'].append(node_ids[i])

                logger.info(f'Semantic clustering: {best_k} clusters, silhouette: {best_score:.3f}')
                return dict(clusters)

        except Exception as e:
            logger.warning(f'Semantic clustering failed: {e}')

        return self._simple_semantic_clustering()

    def _simple_semantic_clustering(self) -> Dict[str, List[str]]:
        """Simple semantic clustering based on tags and metadata."""
        logger.info('Using simple semantic clustering...')

        clusters = defaultdict(list)

        for node_id, node in self.graph.nodes.items():
            # Group by domain
            domain_key = f'semantic_{node.domain}'
            clusters[domain_key].append(node_id)

        return dict(clusters)

    def _directory_clustering(self) -> Dict[str, List[str]]:
        """Cluster modules based on directory structure."""
        logger.info('Performing directory-based clustering...')

        clusters = defaultdict(list)

        for node_id, node in self.graph.nodes.items():
            # Get directory path
            directory = node.relative_path.split('/')[:-1]  # Remove filename

            if len(directory) >= 2:
                # Use first two directory levels
                cluster_key = f'dir_{directory[0]}_{directory[1]}'
            elif len(directory) == 1:
                cluster_key = f'dir_{directory[0]}'
            else:
                cluster_key = 'dir_root'

            clusters[cluster_key].append(node_id)

        return dict(clusters)

    def _complexity_clustering(self) -> Dict[str, List[str]]:
        """Cluster modules based on complexity metrics."""
        logger.info('Performing complexity-based clustering...')

        # Categorize modules by complexity
        low_complexity = []
        medium_complexity = []
        high_complexity = []

        for node_id, node in self.graph.nodes.items():
            complexity = node.complexity_metrics.get('cyclomatic_complexity', 0)
            loc = node.lines_of_code

            # Simple complexity scoring
            complexity_score = complexity + (loc / 100)

            if complexity_score < 5:
                low_complexity.append(node_id)
            elif complexity_score < 15:
                medium_complexity.append(node_id)
            else:
                high_complexity.append(node_id)

        clusters = {}
        if low_complexity:
            clusters['complexity_low'] = low_complexity
        if medium_complexity:
            clusters['complexity_medium'] = medium_complexity
        if high_complexity:
            clusters['complexity_high'] = high_complexity

        return clusters

    def _create_consensus_clustering(self, method_results: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """Create consensus clustering from multiple methods."""
        logger.info('Creating consensus clustering...')

        if not method_results:
            logger.warning('No clustering results available for consensus')
            return {}

        # Collect all nodes
        all_nodes = set(self.graph.nodes.keys())

        # Score each possible pair of nodes based on how often they appear together
        pair_scores: Dict[Tuple[str, str], float] = defaultdict(float)

        for method, clusters in method_results.items():
            for cluster_id, nodes in clusters.items():
                # Give weight based on method reliability
                weight = self._get_method_weight(method)

                # Score all pairs in this cluster
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:]:
                        sorted_nodes = sorted([node1, node2])
                        pair_key = (sorted_nodes[0], sorted_nodes[1])
                        pair_scores[pair_key] += weight

        # Normalize scores
        if pair_scores:
            max_possible_score = sum(self._get_method_weight(method) for method in method_results.keys())
            for pair_key in pair_scores:
                pair_scores[pair_key] /= max_possible_score

        # Build consensus clusters using threshold
        consensus_threshold = 0.5
        consensus_clusters = self._build_clusters_from_pairs(dict(pair_scores), consensus_threshold, all_nodes)

        # Ensure target number of clusters
        consensus_clusters = self._adjust_cluster_count(consensus_clusters, self.target_clusters)

        return consensus_clusters

    def _get_method_weight(self, method: str) -> float:
        """Get reliability weight for clustering method."""
        weights = {
            'community_detection': 1.0,
            'semantic_clustering': 0.8,
            'directory_clustering': 0.6,
            'complexity_clustering': 0.4,
        }
        return weights.get(method, 0.5)

    def _build_clusters_from_pairs(
        self, pair_scores: Dict[Tuple[str, str], float],
        threshold: float, all_nodes: Set[str],
    ) -> Dict[str, List[str]]:
        """Build clusters from pairwise similarity scores."""
        clusters: Dict[str, List[str]] = {}
        assigned_nodes = set()
        cluster_id = 0

        # Sort pairs by score (highest first)
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)

        for (node1, node2), score in sorted_pairs:
            if score < threshold:
                break

            # Find existing clusters for these nodes
            cluster1 = None
            cluster2 = None

            for cid, nodes in clusters.items():
                if node1 in nodes:
                    cluster1 = cid
                if node2 in nodes:
                    cluster2 = cid

            if cluster1 is None and cluster2 is None:
                # Create new cluster
                new_cluster_id = f'consensus_{cluster_id}'
                clusters[new_cluster_id] = [node1, node2]
                assigned_nodes.update([node1, node2])
                cluster_id += 1

            elif cluster1 is not None and cluster2 is None:
                # Add node2 to cluster1
                clusters[cluster1].append(node2)
                assigned_nodes.add(node2)

            elif cluster1 is None and cluster2 is not None:
                # Add node1 to cluster2
                clusters[cluster2].append(node1)
                assigned_nodes.add(node1)

            elif cluster1 is not None and cluster2 is not None and cluster1 != cluster2:
                # Merge clusters
                clusters[cluster1].extend(clusters[cluster2])
                del clusters[cluster2]

        # Add remaining nodes as singleton clusters
        for node in all_nodes:
            if node not in assigned_nodes:
                singleton_id = f'consensus_{cluster_id}'
                clusters[singleton_id] = [node]
                cluster_id += 1

        return clusters

    def _adjust_cluster_count(self, clusters: Dict[str, List[str]], target: int) -> Dict[str, List[str]]:
        """Adjust cluster count to target by merging or splitting."""
        current_count = len(clusters)

        if current_count == target:
            return clusters

        elif current_count > target:
            # Need to merge clusters
            return self._merge_smallest_clusters(clusters, target)

        else:
            # Need to split clusters
            return self._split_largest_clusters(clusters, target)

    def _merge_smallest_clusters(self, clusters: Dict[str, List[str]], target: int) -> Dict[str, List[str]]:
        """Merge smallest clusters to reach target count."""
        while len(clusters) > target:
            # Find two smallest clusters
            sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]))

            if len(sorted_clusters) < 2:
                break

            # Merge first two smallest
            cluster1_id, cluster1_nodes = sorted_clusters[0]
            cluster2_id, cluster2_nodes = sorted_clusters[1]

            # Create merged cluster
            merged_id = f'merged_{cluster1_id}_{cluster2_id}'
            merged_nodes = cluster1_nodes + cluster2_nodes

            # Remove old clusters and add merged
            del clusters[cluster1_id]
            del clusters[cluster2_id]
            clusters[merged_id] = merged_nodes

        return clusters

    def _split_largest_clusters(self, clusters: Dict[str, List[str]], target: int) -> Dict[str, List[str]]:
        """Split largest clusters to reach target count."""
        while len(clusters) < target:
            # Find largest cluster that can be split
            largest_cluster = max(clusters.items(), key=lambda x: len(x[1]))
            cluster_id, nodes = largest_cluster

            if len(nodes) < 2:
                break  # Can't split further

            # Simple split: divide in half
            mid = len(nodes) // 2
            split1 = nodes[:mid]
            split2 = nodes[mid:]

            # Remove original cluster and add splits
            del clusters[cluster_id]
            clusters[f'{cluster_id}_1'] = split1
            clusters[f'{cluster_id}_2'] = split2

        return clusters

    def _build_final_clusters(self, consensus_clusters: Dict[str, List[str]]) -> Dict[str, ModuleCluster]:
        """Build final cluster objects with metadata."""
        logger.info('Building final clusters with metadata...')

        final_clusters = {}

        for cluster_id, node_ids in consensus_clusters.items():
            if not node_ids:
                continue

            # Create cluster object
            cluster = ModuleCluster(
                id=cluster_id,
                name=self._generate_cluster_name(node_ids),
                modules=node_ids,
                size=len(node_ids),
            )

            # Compute cluster characteristics
            cluster.dominant_layer = self._find_dominant_attribute(node_ids, 'layer')
            cluster.dominant_domain = self._find_dominant_attribute(node_ids, 'domain')
            cluster.dominant_type = self._find_dominant_attribute(node_ids, 'module_type')
            cluster.complexity_level = self._compute_cluster_complexity(node_ids)
            cluster.cluster_purpose = self._generate_cluster_purpose(cluster)

            # Compute quality metrics
            cluster.internal_cohesion = self._compute_internal_cohesion(node_ids)
            cluster.external_coupling = self._compute_external_coupling(node_ids)

            final_clusters[cluster_id] = cluster

        return final_clusters

    def _generate_cluster_name(self, node_ids: List[str]) -> str:
        """Generate a descriptive name for the cluster."""
        # Extract common patterns from node IDs
        common_parts = []

        if node_ids:
            # Split all node IDs and find common prefixes
            split_ids = [node_id.split('.') for node_id in node_ids]

            if split_ids:
                # Find longest common prefix
                min_length = min(len(parts) for parts in split_ids)

                for i in range(min_length):
                    parts_at_i = [parts[i] for parts in split_ids]
                    if len(set(parts_at_i)) == 1:  # All same
                        common_parts.append(parts_at_i[0])
                    else:
                        break

        if common_parts:
            return '_'.join(common_parts) + '_cluster'
        else:
            # Fallback: use dominant characteristics
            nodes = [self.graph.nodes[nid] for nid in node_ids if nid in self.graph.nodes]
            if nodes:
                domains = [node.domain for node in nodes]
                dominant_domain = Counter(domains).most_common(1)[0][0]
                return f'{dominant_domain}_cluster'

        return 'mixed_cluster'

    def _find_dominant_attribute(self, node_ids: List[str], attribute: str) -> str:
        """Find the dominant value of an attribute in the cluster."""
        values = []

        for node_id in node_ids:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                value = getattr(node, attribute, 'unknown')
                values.append(value)

        if values:
            return Counter(values).most_common(1)[0][0]

        return 'unknown'

    def _compute_cluster_complexity(self, node_ids: List[str]) -> str:
        """Compute overall complexity level of the cluster."""
        complexities = []

        for node_id in node_ids:
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                complexity = node.complexity_metrics.get('cyclomatic_complexity', 0)
                loc = node.lines_of_code
                complexities.append(complexity + loc / 100)

        if not complexities:
            return 'medium'

        avg_complexity = sum(complexities) / len(complexities)

        if avg_complexity < 5:
            return 'low'
        elif avg_complexity < 15:
            return 'medium'
        else:
            return 'high'

    def _generate_cluster_purpose(self, cluster: ModuleCluster) -> str:
        """Generate a description of the cluster's purpose."""
        templates = {
            ('authentication', 'core'): 'User authentication and authorization services',
            ('data_access', 'core'): 'Data persistence and database operations',
            ('api_layer', 'interface'): 'API endpoints and request handling',
            ('communication', 'core'): 'Communication and messaging services',
            ('documentation', 'core'): 'Documentation generation and management',
            ('ai_services', 'core'): 'AI and machine learning services',
            ('embedding_services', 'infrastructure'): 'Embedding generation and processing',
            ('database', 'infrastructure'): 'Database connectivity and management',
            ('messaging', 'infrastructure'): 'Message queue and event handling',
        }

        key = (cluster.dominant_domain, cluster.dominant_type)
        if key in templates:
            return templates[key]

        # Generic template
        return f"{cluster.dominant_domain.replace('_', ' ').title()} components for {cluster.dominant_layer} layer"

    def _compute_internal_cohesion(self, node_ids: List[str]) -> float:
        """Compute internal cohesion (connectivity within cluster)."""
        if len(node_ids) < 2:
            return 1.0

        total_possible_edges = len(node_ids) * (len(node_ids) - 1)
        internal_edges = 0

        for source in node_ids:
            for target in node_ids:
                if source != target:
                    edge_key = (source, target)
                    if edge_key in self.graph.edges:
                        internal_edges += 1

        return internal_edges / total_possible_edges if total_possible_edges > 0 else 0.0

    def _compute_external_coupling(self, node_ids: List[str]) -> float:
        """Compute external coupling (connectivity to other clusters)."""
        external_edges = 0
        total_edges = 0

        for source in node_ids:
            neighbors = self.graph.get_neighbors(source) + self.graph.get_predecessors(source)

            for neighbor in neighbors:
                total_edges += 1
                if neighbor not in node_ids:
                    external_edges += 1

        return external_edges / total_edges if total_edges > 0 else 0.0

    def _compute_quality_metrics(self, clusters: Dict[str, ModuleCluster]) -> Dict[str, float]:
        """Compute overall clustering quality metrics."""
        logger.info('Computing clustering quality metrics...')

        if not clusters:
            return {}

        # Silhouette-like metric
        total_cohesion = sum(cluster.internal_cohesion for cluster in clusters.values())
        total_coupling = sum(cluster.external_coupling for cluster in clusters.values())
        avg_cohesion = total_cohesion / len(clusters)
        avg_coupling = total_coupling / len(clusters)

        # Size distribution
        sizes = [cluster.size for cluster in clusters.values()]
        size_variance = np.var(sizes) if sizes else 0

        # Coverage (percentage of nodes clustered)
        total_clustered = sum(cluster.size for cluster in clusters.values())
        total_nodes = len(self.graph.nodes)
        coverage = total_clustered / total_nodes if total_nodes > 0 else 0

        # Modularity approximation
        modularity = avg_cohesion - avg_coupling

        return {
            'average_internal_cohesion': avg_cohesion,
            'average_external_coupling': avg_coupling,
            'modularity': modularity,
            'coverage': coverage,
            'size_variance': size_variance,
            'cluster_count': len(clusters),
            'average_cluster_size': np.mean(sizes) if sizes else 0,
            'largest_cluster_size': max(sizes) if sizes else 0,
            'smallest_cluster_size': min(sizes) if sizes else 0,
        }

    def _create_cluster_hierarchy(self, clusters: Dict[str, ModuleCluster]) -> Dict[str, List[str]]:
        """Create hierarchical organization of clusters."""
        logger.info('Creating cluster hierarchy...')

        hierarchy: Dict[str, List[str]] = {
            'level_1': [],  # High-level categories
            'level_2': [],   # Specific clusters
        }

        # Group clusters by layer
        layer_groups = defaultdict(list)
        for cluster_id, cluster in clusters.items():
            layer_groups[cluster.dominant_layer].append(cluster_id)

        # Level 1: Architectural layers
        hierarchy['level_1'] = list(layer_groups.keys())

        # Level 2: All clusters
        hierarchy['level_2'] = list(clusters.keys())

        # Add layer-specific groupings
        for layer, cluster_ids in layer_groups.items():
            hierarchy[f'layer_{layer}'] = cluster_ids

        return hierarchy


def save_clustering_results(results: ClusteringResults, output_path: str):
    """Save clustering results to JSON file."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f'Clustering results saved to {output_path}')


def load_clustering_results(input_path: str) -> ClusteringResults:
    """Load clustering results from JSON file."""
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    results = ClusteringResults.from_dict(data)
    logger.info(f'Loaded clustering results with {len(results.final_clusters)} clusters')
    return results
