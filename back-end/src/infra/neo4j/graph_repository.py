from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError
from neo4j.exceptions import ServiceUnavailable
from shared.logging import get_logger

from .graph_schema import ClusterNode
from .graph_schema import CypherQueries
from .graph_schema import ModuleNode
from .graph_schema import TechnologyNode

logger = get_logger(__name__)


class Neo4jConnection:
    """Neo4j database connection manager."""

    def __init__(self, uri: str, username: str, password: str, database: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,  # 60 seconds
            )

            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run('RETURN 1')

            logger.info(f'Connected to Neo4j at {self.uri}')

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f'Failed to connect to Neo4j: {e}')
            raise

    def get_session(self):
        """Get a new Neo4j session."""
        if not self.driver:
            self._connect()
        return self.driver.session(database=self.database)

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info('Neo4j connection closed')


class GraphRepository:
    """Repository for managing code architecture graph in Neo4j."""

    def __init__(self, uri: str, username: str, password: str, database: str):
        self.connection = Neo4jConnection(uri, username, password, database)
        self.database = database

    def create_constraints(self):
        """Create Neo4j constraints and indexes for better performance."""
        constraints = [
            'CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE',
            'CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (c:Cluster) REQUIRE c.id IS UNIQUE',
            'CREATE CONSTRAINT dataflow_id IF NOT EXISTS FOR (df:DataFlow) REQUIRE df.id IS UNIQUE',
            'CREATE CONSTRAINT technology_id IF NOT EXISTS FOR (t:Technology) REQUIRE t.id IS UNIQUE',
            'CREATE INDEX module_name IF NOT EXISTS FOR (m:Module) ON (m.name)',
            'CREATE INDEX module_layer IF NOT EXISTS FOR (m:Module) ON (m.layer)',
            'CREATE INDEX module_domain IF NOT EXISTS FOR (m:Module) ON (m.domain)',
            'CREATE INDEX repository_url IF NOT EXISTS FOR (n) ON (n.repository_url)',
        ]

        with self.connection.get_session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f'Constraint creation warning: {e}')

        logger.info('Neo4j constraints and indexes created')

    def delete_repository_data(self, repo_url: str):
        """Delete all data for a specific repository."""
        with self.connection.get_session() as session:
            session.run(
                CypherQueries.DELETE_REPOSITORY_DATA,
                repo_url=repo_url,
            )
            logger.info(f'Deleted data for repository: {repo_url}')

    def clear_repository_data(self, repo_url: str):
        """Clear all data for a specific repository."""
        with self.connection.get_session() as session:
            session.run(
                CypherQueries.DELETE_REPOSITORY_DATA,
                repo_url=repo_url,
            )
            logger.info(f'Cleared data for repository: {repo_url}')

    def save_module(self, module: ModuleNode) -> bool:
        """Save a module node to the graph."""
        try:
            with self.connection.get_session() as session:
                query = """
                CREATE (m:Module {
                    id: $id,
                    name: $name,
                    file_path: $file_path,
                    lines_of_code: $lines_of_code,
                    complexity_score: $complexity_score,
                    layer: $layer,
                    domain: $domain,
                    module_type: $module_type,
                    repo_url: $repo_url
                })
                RETURN m
                """

                result = session.run(
                    query,
                    id=module.id,
                    name=module.name,
                    file_path=module.file_path,
                    lines_of_code=module.lines_of_code,
                    complexity_score=module.complexity_score,
                    layer=module.layer,
                    domain=module.domain,
                    module_type=module.module_type,
                    repo_url=module.repo_url,
                )
                return result.single() is not None

        except Exception as e:
            logger.error(f'Error saving module {module.name}: {e}')
            return False

    def save_cluster(self, cluster: ClusterNode) -> bool:
        """Save a cluster node to the graph."""
        try:
            with self.connection.get_session() as session:
                query = """
                CREATE (c:Cluster {
                    id: $id,
                    name: $name,
                    purpose: $purpose,
                    size: $size,
                    cohesion: $cohesion,
                    coupling: $coupling,
                    repo_url: $repo_url
                })
                RETURN c
                """

                result = session.run(
                    query,
                    id=cluster.id,
                    name=cluster.name,
                    purpose=cluster.purpose,
                    size=cluster.size,
                    cohesion=cluster.cohesion,
                    coupling=cluster.coupling,
                    repo_url=cluster.repo_url,
                )
                return result.single() is not None

        except Exception as e:
            logger.error(f'Error saving cluster {cluster.name}: {e}')
            return False

    def save_technology(self, technology: TechnologyNode) -> bool:
        """Save a technology node to the graph."""
        try:
            with self.connection.get_session() as session:
                query = """
                CREATE (t:Technology {
                    id: $id,
                    name: $name,
                    tech_type: $tech_type,
                    version: $version,
                    repo_url: $repo_url
                })
                RETURN t
                """

                result = session.run(
                    query,
                    id=technology.id,
                    name=technology.name,
                    tech_type=technology.tech_type,
                    version=technology.version,
                    repo_url=technology.repo_url,
                )
                return result.single() is not None

        except Exception as e:
            logger.error(f'Error saving technology {technology.name}: {e}')
            return False

    def create_relationship(
        self, source_name: str, target_name: str,
        relationship_type: str, repo_url: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between nodes."""
        try:
            properties = properties or {}

            with self.connection.get_session() as session:
                query = f"""
                MATCH (source:Module {{name: $source_name, repo_url: $repo_url}})
                MATCH (target:Module {{name: $target_name, repo_url: $repo_url}})
                CREATE (source)-[r:{relationship_type}]->(target)
                SET r += $properties
                RETURN r
                """

                result = session.run(
                    query,
                    source_name=source_name,
                    target_name=target_name,
                    repo_url=repo_url,
                    properties=properties,
                )
                return result.single() is not None

        except Exception as e:
            logger.error(f'Error creating relationship {source_name}->{target_name}: {e}')
            return False

    def get_system_overview(self, repo_url: str) -> Dict[str, Any]:
        """Get high-level system overview."""
        try:
            with self.connection.get_session() as session:
                result = session.run(CypherQueries.SYSTEM_OVERVIEW, repo_url=repo_url)
                record = result.single()

                if record:
                    return {
                        'total_modules': record['total_modules'],
                        'total_clusters': record['total_clusters'],
                        'total_technologies': record['total_technologies'],
                        'layers': record['layers'],
                        'domains': record['domains'],
                        'module_types': record['module_types'],
                    }
                return {}

        except Exception as e:
            logger.error(f'Error getting system overview: {e}')
            return {}

    def get_module_details(self, module_name: str, repo_url: str) -> Dict[str, Any]:
        """Get detailed information about a specific module."""
        try:
            with self.connection.get_session() as session:
                query = """
                MATCH (m:Module {name: $module_name, repo_url: $repo_url})
                OPTIONAL MATCH (m)-[dep:DEPENDS_ON]->(target)
                WHERE target.repo_url = $repo_url
                OPTIONAL MATCH (source)-[rdep:DEPENDS_ON]->(m)
                WHERE source.repo_url = $repo_url
                OPTIONAL MATCH (cluster:Cluster)-[:CONTAINS]->(m)
                WHERE cluster.repo_url = $repo_url
                RETURN m,
                    collect(DISTINCT target.name) as dependencies,
                    collect(DISTINCT source.name) as dependents,
                    cluster.name as cluster_name
                """

                result = session.run(query, module_name=module_name, repo_url=repo_url)
                record = result.single()

                if record and record['m']:
                    module = record['m']
                    return {
                        'id': module['id'],
                        'name': module['name'],
                        'file_path': module['file_path'],
                        'layer': module['layer'],
                        'domain': module['domain'],
                        'module_type': module['module_type'],
                        'complexity_score': module['complexity_score'],
                        'lines_of_code': module['lines_of_code'],
                        'dependencies': [dep for dep in record['dependencies'] if dep],
                        'dependents': [dep for dep in record['dependents'] if dep],
                        'cluster_name': record['cluster_name'],
                    }
                return {}

        except Exception as e:
            logger.error(f'Error getting module details: {e}')
            return {}

    def get_data_flows(self, repo_url: str, source_module: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get data flow information."""
        try:
            with self.connection.get_session() as session:
                if source_module:
                    query = """
                    MATCH (source:Module {name: $source_module, repo_url: $repo_url})
                    MATCH (source)-[:FLOWS_TO]->(target)
                    WHERE target.repo_url = $repo_url
                    RETURN source.name as source_name,
                        target.name as target_name,
                        'direct' as flow_type
                    """
                    result = session.run(query, source_module=source_module, repo_url=repo_url)
                else:
                    query = """
                    MATCH (source:Module)-[:FLOWS_TO]->(target:Module)
                    WHERE source.repo_url = $repo_url AND target.repo_url = $repo_url
                    RETURN source.name as source_name,
                        target.name as target_name,
                        'direct' as flow_type
                    """
                    result = session.run(query, repo_url=repo_url)

                flows = []
                for record in result:
                    flows.append({
                        'source': record['source_name'],
                        'target': record['target_name'],
                        'flow_type': record['flow_type'],
                    })

                return flows

        except Exception as e:
            logger.error(f'Error getting data flows: {e}')
            return []

    def get_cluster_analysis(self, repo_url: str, cluster_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cluster analysis information."""
        try:
            with self.connection.get_session() as session:
                if cluster_name:
                    query = """
                    MATCH (c:Cluster {name: $cluster_name, repo_url: $repo_url})
                    OPTIONAL MATCH (c)-[:CONTAINS]->(m:Module)
                    OPTIONAL MATCH (m)-[ext:DEPENDS_ON]->(external)
                    WHERE NOT (c)-[:CONTAINS]->(external) AND external.repo_url = $repo_url
                    RETURN c,
                        collect(DISTINCT m.name) as modules,
                        collect(DISTINCT external.name) as external_dependencies,
                        count(DISTINCT m) as module_count
                    """
                    result = session.run(query, cluster_name=cluster_name, repo_url=repo_url)
                else:
                    query = """
                    MATCH (c:Cluster)
                    WHERE c.repo_url = $repo_url
                    OPTIONAL MATCH (c)-[:CONTAINS]->(m:Module)
                    OPTIONAL MATCH (m)-[ext:DEPENDS_ON]->(external)
                    WHERE NOT (c)-[:CONTAINS]->(external) AND external.repo_url = $repo_url
                    RETURN c,
                        collect(DISTINCT m.name) as modules,
                        collect(DISTINCT external.name) as external_dependencies,
                        count(DISTINCT m) as module_count
                    """
                    result = session.run(query, repo_url=repo_url)

                clusters = []
                for record in result:
                    cluster = record['c']
                    clusters.append({
                        'id': cluster['id'],
                        'name': cluster['name'],
                        'purpose': cluster['purpose'],
                        'size': record['module_count'],
                        'modules': [mod for mod in record['modules'] if mod],
                        'external_dependencies': [dep for dep in record['external_dependencies'] if dep],
                        'cohesion': cluster['cohesion'],
                        'coupling': cluster['coupling'],
                    })

                return clusters

        except Exception as e:
            logger.error(f'Error getting cluster analysis: {e}')
            return []

    def get_technology_stack(self, repo_url: str) -> Dict[str, List[str]]:
        """Get technology stack information."""
        try:
            with self.connection.get_session() as session:
                result = session.run(CypherQueries.TECHNOLOGY_STACK, repo_url=repo_url)

                tech_stack: Dict[str, List[str]] = {}
                for record in result:
                    tech_type = record['category'] or 'unknown'
                    technologies = [tech for tech in record['technologies'] if tech]

                    if tech_type not in tech_stack:
                        tech_stack[tech_type] = []
                    tech_stack[tech_type].extend(technologies)

                return tech_stack

        except Exception as e:
            logger.error(f'Error getting technology stack: {e}')
            return {}

    def get_communication_patterns(self, repo_url: str, module_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get communication patterns between modules."""
        try:
            with self.connection.get_session() as session:
                if module_name:
                    query = """
                    MATCH (source:Module {repo_url: $repo_url})-[comm:COMMUNICATES_WITH]->(target:Module {repo_url: $repo_url})
                    WHERE source.name = $module_name OR target.name = $module_name
                    RETURN source.name as source_module,
                        target.name as target_module,
                        comm.pattern_type as communication_type,
                        comm.protocol as protocol,
                        comm.frequency as frequency
                    """
                    result = session.run(query, repo_url=repo_url, module_name=module_name)
                else:
                    result = session.run(CypherQueries.COMMUNICATION_PATTERNS, repo_url=repo_url, target_module=None)

                patterns = []
                for record in result:
                    if 'pattern' in record:
                        pattern = record['pattern']
                        patterns.append({
                            'source_module': pattern['source_module'],
                            'target_module': pattern['target_module'],
                            'communication_type': pattern['communication_type'],
                            'protocol': pattern['protocol'],
                            'frequency': pattern['frequency'],
                        })
                    else:
                        patterns.append({
                            'source_module': record['source_module'],
                            'target_module': record['target_module'],
                            'communication_type': record['communication_type'],
                            'protocol': record['protocol'],
                            'frequency': record['frequency'],
                        })

                return patterns

        except Exception as e:
            logger.error(f'Error getting communication patterns: {e}')
            return []

    def get_circular_dependencies(self, repo_url: str) -> List[Tuple[str, str]]:
        """Find circular dependencies in the system."""
        try:
            with self.connection.get_session() as session:
                result = session.run(CypherQueries.CIRCULAR_DEPENDENCIES, repo_url=repo_url)

                circular_deps = []
                for record in result:
                    circular_deps.append((record['module1'], record['module2']))

                return circular_deps

        except Exception as e:
            logger.error(f'Error getting circular dependencies: {e}')
            return []

    def get_high_coupling_modules(self, repo_url: str, threshold: int = 5) -> List[Dict[str, Any]]:
        """Get modules with high coupling (many dependencies)."""
        try:
            with self.connection.get_session() as session:
                result = session.run(CypherQueries.HIGH_COUPLING_MODULES, repo_url=repo_url)

                high_coupling = []
                for record in result:
                    if 'high_coupling_module' in record:
                        module = record['high_coupling_module']
                        high_coupling.append({
                            'module_name': module['module_name'],
                            'outgoing_dependencies': module['outgoing_dependencies'],
                            'incoming_dependencies': module['incoming_dependencies'],
                            'total_coupling': module['total_coupling'],
                        })
                    else:
                        high_coupling.append({
                            'module_name': record['module_name'],
                            'outgoing_dependencies': record['outgoing'],
                            'incoming_dependencies': record['incoming'],
                            'total_coupling': record['total_coupling'],
                        })

                return high_coupling

        except Exception as e:
            logger.error(f'Error getting high coupling modules: {e}')
            return []
