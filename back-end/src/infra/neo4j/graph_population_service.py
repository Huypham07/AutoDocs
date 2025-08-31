from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from infra.neo4j.graph_repository import GraphRepository
from infra.neo4j.graph_schema import ClusterNode
from infra.neo4j.graph_schema import ModuleNode
from infra.neo4j.graph_schema import RelationshipType
from infra.neo4j.graph_schema import TechnologyNode
from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResults:
    """Container for architecture pipeline results."""
    # Core outputs - match with pipeline.py structure
    graph: Any  # MultiModalGraph
    clustering_results: Any  # ClusteringResults
    rag_data: Any  # RAGOptimizedData
    architecture_knowledge: Optional[Any] = None

    # Metadata
    repo_url: str = ''
    execution_time: float = 0.0
    timestamp: str = ''
    repo_path: str = ''

    # Quality metrics
    overall_quality_score: float = 0.0
    validation_passed: bool = False


class GraphPopulationService:
    """Service for populating Neo4j from pipeline results."""

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def populate_from_pipeline(self, results: PipelineResults) -> bool:
        """Populate Neo4j graph from architecture pipeline results."""
        try:
            logger.info(f'Starting graph population for {results.repo_url}')

            # Clear existing data for this repository
            self._clear_repository_data(results.repo_url)

            # Populate in order: modules -> clusters -> relationships
            self._populate_modules(results)
            self._populate_clusters(results)
            self._populate_relationships(results)
            self._populate_technologies(results)

            logger.info(f'Successfully populated graph for {results.repo_url}')
            return True

        except Exception as e:
            logger.error(f'Error populating graph: {e}')
            return False

    def _clear_repository_data(self, repo_url: str):
        """Clear existing data for the repository."""
        try:
            # Delete all nodes and relationships for this repository
            self.graph_repo.delete_repository_data(repo_url)
            logger.info(f'Cleared existing data for {repo_url}')
        except Exception as e:
            logger.error(f'Error clearing repository data: {e}')

    def _populate_modules(self, results: PipelineResults):
        """Populate module nodes from graph."""
        graph = results.graph

        if not graph or not graph.nodes:
            logger.warning('No nodes found in graph')
            return

        modules_created = 0

        for node_id, node in graph.nodes.items():
            try:
                # Extract module information
                module_data = self._extract_module_data_from_node(node, results.repo_url)

                if module_data:
                    # Create module node
                    module_node = ModuleNode(
                        name=module_data['name'],
                        file_path=module_data['file_path'],
                        module_type=module_data['module_type'],
                        layer=module_data['layer'],
                        domain=module_data['domain'],
                        complexity_score=module_data['complexity_score'],
                        lines_of_code=module_data['lines_of_code'],
                        repo_url=results.repo_url,
                        metadata=module_data.get('metadata', {}),
                    )

                    # Save to Neo4j
                    self.graph_repo.save_module(module_node)
                    modules_created += 1

            except Exception as e:
                logger.error(f'Error creating module node: {e}')
                continue

        logger.info(f'Created {modules_created} module nodes')

    def _extract_module_data_from_node(self, node, repo_url: str) -> Optional[Dict[str, Any]]:
        """Extract module data from GraphNode object."""
        try:
            # GraphNode has direct attributes
            name = node.id or node.module_path
            file_path = node.file_path

            if not name or not file_path:
                return None

            # Extract or infer module type
            module_type = node.module_type if node.module_type != 'unknown' else self._infer_module_type(file_path, {})

            # Extract architectural information
            layer = node.layer if node.layer != 'unknown' else 'application'
            domain = node.domain if node.domain != 'unknown' else 'core'

            # Extract metrics
            complexity_score = float(sum(node.complexity_metrics.values()) / len(node.complexity_metrics) if node.complexity_metrics else 0)
            lines_of_code = int(node.lines_of_code)

            # Additional metadata
            metadata = {
                'relative_path': node.relative_path,
                'complexity_metrics': node.complexity_metrics,
                'centrality_measures': node.centrality_measures,
                'semantic_tags': node.semantic_tags,
                'node_type': node.node_type,
            }

            return {
                'name': name,
                'file_path': file_path,
                'module_type': module_type,
                'layer': layer,
                'domain': domain,
                'complexity_score': complexity_score,
                'lines_of_code': lines_of_code,
                'metadata': metadata,
            }

        except Exception as e:
            logger.error(f'Error extracting module data from node: {e}')
            return None

    def _infer_module_type(self, file_path: str, node_data: Dict[str, Any]) -> str:
        """Infer module type from file path and content."""
        # Check explicit type first
        if 'module_type' in node_data:
            return node_data['module_type']

        # Infer from file path
        file_path_lower = file_path.lower()

        if any(keyword in file_path_lower for keyword in ['test', 'spec']):
            return 'test'
        elif any(keyword in file_path_lower for keyword in ['api', 'router', 'endpoint']):
            return 'api'
        elif any(keyword in file_path_lower for keyword in ['model', 'schema']):
            return 'model'
        elif any(keyword in file_path_lower for keyword in ['service', 'application']):
            return 'service'
        elif any(keyword in file_path_lower for keyword in ['domain', 'entity']):
            return 'domain'
        elif any(keyword in file_path_lower for keyword in ['infra', 'infrastructure']):
            return 'infrastructure'
        elif any(keyword in file_path_lower for keyword in ['util', 'helper']):
            return 'utility'
        elif 'config' in file_path_lower:
            return 'configuration'
        else:
            return 'module'

    def _populate_clusters(self, results: PipelineResults):
        """Populate cluster nodes."""
        clusters_created = 0

        # Get clusters from clustering results
        clustering_results = results.clustering_results
        if not clustering_results or not clustering_results.final_clusters:
            logger.warning('No clusters found in clustering results')
            return

        for cluster_id, cluster in clustering_results.final_clusters.items():
            try:
                # Create cluster node
                cluster_node = ClusterNode(
                    name=cluster.name,
                    purpose=f'Cluster containing {len(cluster.modules)} modules',
                    size=cluster.size,
                    cohesion=0.5,  # Default value
                    coupling=0.5,  # Default value
                    repo_url=results.repo_url,
                    metadata={
                        'cluster_id': cluster_id,
                        'dominant_layer': getattr(cluster, 'dominant_layer', 'unknown'),
                        'dominant_domain': getattr(cluster, 'dominant_domain', 'unknown'),
                        'dominant_type': getattr(cluster, 'dominant_type', 'unknown'),
                    },
                )

                # Save to Neo4j
                self.graph_repo.save_cluster(cluster_node)
                clusters_created += 1

                # Create cluster-module relationships
                for module_name in cluster.modules:
                    self.graph_repo.create_relationship(
                        cluster.name, module_name,
                        RelationshipType.CONTAINS.value,
                        results.repo_url,
                    )

            except Exception as e:
                logger.error(f'Error creating cluster node: {e}')
                continue

        logger.info(f'Created {clusters_created} cluster nodes')

    def _extract_cluster_data(self, cluster: Dict[str, Any], repo_url: str) -> Optional[Dict[str, Any]]:
        """Extract cluster data from pipeline results."""
        try:
            name = cluster.get('name') or cluster.get('id')
            if not name:
                return None

            # Extract cluster metrics
            purpose = cluster.get('purpose', 'Functional grouping')
            size = len(cluster.get('modules', []))
            cohesion = float(cluster.get('cohesion', 0.5))
            coupling = float(cluster.get('coupling', 0.5))

            # Modules in cluster
            modules = cluster.get('modules', [])

            # Additional metadata
            metadata = {
                'cluster_type': cluster.get('type'),
                'level': cluster.get('level'),
                'description': cluster.get('description'),
                'external_dependencies': cluster.get('external_dependencies', []),
            }

            return {
                'name': name,
                'purpose': purpose,
                'size': size,
                'cohesion': cohesion,
                'coupling': coupling,
                'modules': modules,
                'metadata': metadata,
            }

        except Exception as e:
            logger.error(f'Error extracting cluster data: {e}')
            return None

    def _populate_relationships(self, results: PipelineResults):
        """Populate relationships between nodes."""
        graph = results.graph

        if not graph or not graph.edges:
            logger.warning('No edges found in graph')
            return

        relationships_created = 0

        # graph.edges is a dictionary with (source, target) tuple as key and GraphEdge as value
        for (source, target), edge in graph.edges.items():
            try:
                # Extract relationship information
                # Get the primary relationship type
                rel_type = edge.relationship_types[0].value if edge.relationship_types else 'DEPENDS_ON'

                if source and target:
                    # Map relationship type
                    mapped_type = self._map_relationship_type(rel_type)

                    # Create relationship
                    self.graph_repo.create_relationship(
                        source, target, mapped_type, results.repo_url,
                        properties={
                            'interaction_weight': edge.interaction_weight,
                            'call_frequency': edge.call_frequency,
                            'dependency_strength': edge.dependency_strength,
                            'semantic_similarity': edge.semantic_similarity,
                        },
                    )
                    relationships_created += 1

            except Exception as e:
                logger.error(f'Error creating relationship {source}->{target}: {e}')
                continue

        logger.info(f'Created {relationships_created} relationships')

    def _map_relationship_type(self, rel_type: str) -> str:
        """Map pipeline relationship types to schema types."""
        type_mapping = {
            'imports': RelationshipType.IMPORTS.value,
            'import': RelationshipType.IMPORTS.value,
            'calls': RelationshipType.CALLS.value,
            'depends': RelationshipType.DEPENDS_ON.value,
            'depends_on': RelationshipType.DEPENDS_ON.value,
            'dependency': RelationshipType.DEPENDS_ON.value,
            'inherits': RelationshipType.INHERITS.value,
            'inheritance': RelationshipType.INHERITS.value,
            'contains': RelationshipType.CONTAINS.value,
            'uses': RelationshipType.USES.value,
            'implements': RelationshipType.IMPLEMENTS.value,
            'communicates': RelationshipType.COMMUNICATES.value,
            'flows': RelationshipType.DATA_FLOW.value,
            'data_flow': RelationshipType.DATA_FLOW.value,
            'semantic': RelationshipType.DEPENDS_ON.value,
            'structural': RelationshipType.DEPENDS_ON.value,
            'composes': RelationshipType.CONTAINS.value,
        }

        return type_mapping.get(rel_type.lower(), RelationshipType.DEPENDS_ON.value)

    def _populate_technologies(self, results: PipelineResults):
        """Populate technology nodes from pipeline results."""
        technologies_created = 0

        try:
            # For simplified pipeline, we'll extract basic technology info from file extensions
            graph = results.graph
            tech_extensions = set()

            for node_id, node in graph.nodes.items():
                if node.file_path:
                    ext = node.file_path.split('.')[-1].lower()
                    if ext in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'go', 'rs', 'rb', 'php']:
                        tech_extensions.add(ext)

            # Map extensions to technology names
            tech_mapping = {
                'py': 'Python',
                'js': 'JavaScript',
                'ts': 'TypeScript',
                'java': 'Java',
                'cpp': 'C++',
                'c': 'C',
                'go': 'Go',
                'rs': 'Rust',
                'rb': 'Ruby',
                'php': 'PHP',
            }

            for ext in tech_extensions:
                if ext in tech_mapping:
                    tech_node = TechnologyNode(
                        name=tech_mapping[ext],
                        tech_type='language',
                        version=None,
                        repo_url=results.repo_url,
                        metadata={'file_extension': ext},
                    )

                    self.graph_repo.save_technology(tech_node)
                    technologies_created += 1

            logger.info(f'Created {technologies_created} technology nodes')

        except Exception as e:
            logger.error(f'Error populating technologies: {e}')

    def populate_from_files(
        self,
        repo_url: str,
        enhanced_graph_file: str,
        clusters_file: str,
        rag_data_file: str,
        validation_file: str,
    ) -> bool:
        """Populate graph from result files."""
        try:
            # Load data from files
            with open(enhanced_graph_file, encoding='utf-8') as f:
                enhanced_graph = json.load(f)

            with open(clusters_file, encoding='utf-8') as f:
                clusters = json.load(f)

            with open(rag_data_file, encoding='utf-8') as f:
                rag_data = json.load(f)

            with open(validation_file, encoding='utf-8') as f:
                validation_report = json.load(f)

            # Create results object
            results = PipelineResults(
                repo_url=repo_url,
                graph=enhanced_graph,
                clustering_results=clusters,
                rag_data=rag_data,
                overall_quality_score=validation_report.get('overall_score', 0.0) if validation_report else 0.0,
                validation_passed=validation_report.get('validation_passed', False) if validation_report else False,
            )

            # Populate graph
            return self.populate_from_pipeline(results)

        except Exception as e:
            logger.error(f'Error loading files for graph population: {e}')
            return False

    def get_population_status(self, repo_url: str) -> Dict[str, Any]:
        """Get status of graph population for a repository."""
        try:
            # Check if there are any nodes with this repo_url
            with self.graph_repo.connection.get_session() as session:
                query = """
                MATCH (n)
                WHERE n.repo_url = $repo_url
                RETURN count(n) as node_count
                """

                result = session.run(query, repo_url=repo_url)
                record = result.single()

                node_count = record['node_count'] if record else 0

                if node_count > 0:
                    # Get detailed overview if nodes exist
                    overview = self.graph_repo.get_system_overview(repo_url)

                    return {
                        'populated': True,
                        'modules': overview.get('total_modules', 0),
                        'clusters': overview.get('total_clusters', 0),
                        'technologies': overview.get('total_technologies', 0),
                        'node_count': node_count,
                    }
                else:
                    return {
                        'populated': False,
                        'modules': 0,
                        'clusters': 0,
                        'technologies': 0,
                        'node_count': 0,
                    }

        except Exception as e:
            logger.error(f'Error checking population status: {e}')
            return {
                'populated': False,
                'error': str(e),
                'modules': 0,
                'clusters': 0,
                'technologies': 0,
                'node_count': 0,
            }
