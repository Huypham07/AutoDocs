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
            # Composite unique constraints để đảm bảo unique per repo
            'CREATE CONSTRAINT module_repo_id IF NOT EXISTS FOR (m:Module) REQUIRE (m.id, m.repo_url) IS UNIQUE',
            'CREATE CONSTRAINT cluster_repo_id IF NOT EXISTS FOR (c:Cluster) REQUIRE (c.id, c.repo_url) IS UNIQUE',
            'CREATE CONSTRAINT dataflow_repo_id IF NOT EXISTS FOR (df:DataFlow) REQUIRE (df.id, df.repo_url) IS UNIQUE',
            'CREATE CONSTRAINT technology_repo_id IF NOT EXISTS FOR (t:Technology) REQUIRE (t.id, t.repo_url) IS UNIQUE',

            # Indexes để tăng performance
            'CREATE INDEX module_name_repo IF NOT EXISTS FOR (m:Module) ON (m.name, m.repo_url)',
            'CREATE INDEX module_layer_repo IF NOT EXISTS FOR (m:Module) ON (m.layer, m.repo_url)',
            'CREATE INDEX module_domain_repo IF NOT EXISTS FOR (m:Module) ON (m.domain, m.repo_url)',
            'CREATE INDEX repo_url_index IF NOT EXISTS FOR (n) ON (n.repo_url)',
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
                MERGE (m:Module {id: $id, repo_url: $repo_url})
                SET m.name = $name,
                    m.file_path = $file_path,
                    m.lines_of_code = $lines_of_code,
                    m.complexity_score = $complexity_score,
                    m.layer = $layer,
                    m.domain = $domain,
                    m.module_type = $module_type
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
                MERGE (c:Cluster {id: $id, repo_url: $repo_url})
                SET c.name = $name,
                    c.purpose = $purpose,
                    c.size = $size,
                    c.cohesion = $cohesion,
                    c.coupling = $coupling
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
                MERGE (t:Technology {id: $id, repo_url: $repo_url})
                SET t.name = $name,
                    t.tech_type = $tech_type,
                    t.version = $version
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
                OPTIONAL MATCH (m)-[:BELONGS_TO]->(cluster:Cluster)
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
                        'layer': module.get('layer', 'unknown'),
                        'domain': module.get('domain', 'unknown'),
                        'module_type': module.get('module_type', 'unknown'),
                        'complexity_score': module.get('complexity_score', 0.0),
                        'lines_of_code': module.get('lines_of_code', 0),
                        'dependencies': [dep for dep in record['dependencies'] if dep],
                        'dependents': [dep for dep in record['dependents'] if dep],
                        'cluster_name': record['cluster_name'],  # Có thể null nếu không có relationship
                    }
                return {}

        except Exception as e:
            logger.error(f'Error getting module details: {e}')
            return {}

    def get_data_flows(self, repo_url: str, source_module: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Data flow analysis is simplified for AutoDocs.

        Returns basic dependency relationships instead of complex data flows.
        For detailed data flow analysis, use specialized tools.
        """
        try:
            with self.connection.get_session() as session:
                if source_module:
                    # Use DEPENDS_ON relationships as proxy for data flows
                    query = """
                    MATCH (source:Module {name: $source_module, repo_url: $repo_url})
                    MATCH (source)-[:DEPENDS_ON]->(target)
                    WHERE target.repo_url = $repo_url
                    RETURN source.name as source_name,
                        target.name as target_name,
                        'dependency' as flow_type
                    LIMIT 20
                    """
                    result = session.run(query, source_module=source_module, repo_url=repo_url)
                else:
                    # Basic dependency-based flow overview
                    query = """
                    MATCH (source:Module)-[:DEPENDS_ON]->(target:Module)
                    WHERE source.repo_url = $repo_url AND target.repo_url = $repo_url
                    RETURN source.name as source_name,
                        target.name as target_name,
                        'dependency' as flow_type
                    LIMIT 50
                    """
                    result = session.run(query, repo_url=repo_url)

                flows = []
                for record in result:
                    flows.append({
                        'source': record['source_name'],
                        'target': record['target_name'],
                        'flow_type': record['flow_type'],
                    })

                logger.debug(f'Found {len(flows)} dependency flows for {repo_url}')
                return flows

        except Exception as e:
            logger.warning(f'Error getting data flows (using fallback): {e}')
            return []

    def get_cluster_analysis(self, repo_url: str, cluster_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cluster analysis information."""
        try:
            with self.connection.get_session() as session:
                if cluster_name:
                    query = """
                    MATCH (c:Cluster {name: $cluster_name, repo_url: $repo_url})
                    OPTIONAL MATCH (c)-[:CONTAINS]->(m:Module)
                    WHERE m.repo_url = $repo_url
                    OPTIONAL MATCH (m)-[ext:DEPENDS_ON]->(external)
                    WHERE NOT exists((c)-[:CONTAINS]->(external)) AND external.repo_url = $repo_url
                    WITH c, collect(DISTINCT m.name) as modules, collect(DISTINCT external.name) as external_deps
                    RETURN c,
                        modules,
                        external_deps,
                        size(modules) as module_count
                    """
                    result = session.run(query, cluster_name=cluster_name, repo_url=repo_url)
                else:
                    query = """
                    MATCH (c:Cluster {repo_url: $repo_url})
                    OPTIONAL MATCH (c)-[:CONTAINS]->(m:Module)
                    WHERE m.repo_url = $repo_url
                    OPTIONAL MATCH (m)-[ext:DEPENDS_ON]->(external)
                    WHERE NOT exists((c)-[:CONTAINS]->(external)) AND external.repo_url = $repo_url
                    WITH c, collect(DISTINCT m.name) as modules, collect(DISTINCT external.name) as external_deps
                    RETURN c,
                        modules,
                        external_deps,
                        size(modules) as module_count
                    """
                    result = session.run(query, repo_url=repo_url)

                clusters = []
                has_results = False

                for record in result:
                    has_results = True
                    cluster = record['c']
                    if cluster:
                        clusters.append({
                            'id': cluster.get('id', cluster.get('name', 'unknown')),
                            'name': cluster.get('name', 'Unknown Cluster'),
                            'purpose': cluster.get('purpose', 'No description available'),
                            'size': record['module_count'],
                            'modules': record['modules'] or [],
                            'external_dependencies': record['external_deps'] or [],
                            'cohesion': cluster.get('cohesion', 0.0),
                            'coupling': cluster.get('coupling', 0.0),
                        })

                # Fallback: If no clusters or relationships found, create virtual cluster
                if not has_results or not clusters:
                    fallback_query = """
                    MATCH (m:Module {repo_url: $repo_url})
                    OPTIONAL MATCH (m)-[:DEPENDS_ON]->(dep:Module {repo_url: $repo_url})
                    RETURN count(DISTINCT m) as total_modules,
                           collect(DISTINCT m.name) as all_modules,
                           count(DISTINCT dep) as total_dependencies
                    """
                    fallback_result = session.run(fallback_query, repo_url=repo_url)
                    fallback_record = fallback_result.single()

                    if fallback_record and fallback_record['total_modules'] > 0:
                        clusters = [{
                            'id': 'system-overview',
                            'name': 'System Overview',
                            'purpose': 'All modules in the system (clustering not available)',
                            'size': fallback_record['total_modules'],
                            'modules': fallback_record['all_modules'] or [],
                            'external_dependencies': [],
                            'cohesion': 0.0,
                            'coupling': 0.0,
                        }]
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
        """
        Communication pattern analysis is simplified for AutoDocs.

        Returns basic dependency relationships as communication patterns.
        For detailed communication analysis, use specialized tools.
        """
        try:
            with self.connection.get_session() as session:
                if module_name:
                    # Use DEPENDS_ON relationships as proxy for communication
                    query = """
                    MATCH (source:Module {repo_url: $repo_url})-[:DEPENDS_ON]->(target:Module {repo_url: $repo_url})
                    WHERE source.name = $module_name OR target.name = $module_name
                    RETURN source.name as source_module,
                        target.name as target_module,
                        'dependency' as communication_type,
                        'internal' as protocol,
                        'static' as frequency
                    LIMIT 20
                    """
                    result = session.run(query, repo_url=repo_url, module_name=module_name)
                else:
                    # Basic dependency overview as communication patterns
                    query = """
                    MATCH (source:Module {repo_url: $repo_url})-[:DEPENDS_ON]->(target:Module {repo_url: $repo_url})
                    RETURN source.name as source_module,
                        target.name as target_module,
                        'dependency' as communication_type,
                        'internal' as protocol,
                        'static' as frequency
                    LIMIT 30
                    """
                    result = session.run(query, repo_url=repo_url)

                patterns = []
                for record in result:
                    patterns.append({
                        'source_module': record['source_module'],
                        'target_module': record['target_module'],
                        'communication_type': record['communication_type'],
                        'protocol': record['protocol'],
                        'frequency': record['frequency'],
                    })

                logger.debug(f'Found {len(patterns)} dependency patterns for {repo_url}')
                return patterns

        except Exception as e:
            logger.warning(f'Error getting communication patterns (using fallback): {e}')
            return []

    def get_circular_dependencies(self, repo_url: str, max_depth: int = 5) -> List[Tuple[str, str]]:
        """
        Circular dependency detection is disabled for AutoDocs.

        This feature is not relevant for documentation generation and
        significantly impacts performance. For code quality analysis,
        use dedicated tools like pylint, flake8, or SonarQube.
        """
        logger.debug(f'Circular dependency check not applicable for documentation system (repo: {repo_url})')
        return []

    def get_high_coupling_modules(self, repo_url: str, threshold: int = 5) -> List[Dict[str, Any]]:
        """
        High coupling analysis is disabled for AutoDocs.

        This feature is more suitable for code quality tools rather than
        documentation generation systems.
        """
        logger.debug(f'High coupling analysis not applicable for documentation system (repo: {repo_url})')
        return []

    def get_all_modules(self, repo_url: str) -> List[Dict[str, Any]]:
        """Get all modules in the repository."""
        query = """
        MATCH (m:Module {repo_url: $repo_url})
        OPTIONAL MATCH (m)-[:BELONGS_TO]->(c:Cluster)
        WHERE c.repo_url = $repo_url
        RETURN m.name as name,
               m.file_path as file_path,
               m.layer as layer,
               m.domain as domain,
               m.module_type as module_type,
               m.complexity_score as complexity_score,
               m.lines_of_code as lines_of_code,
               c.name as cluster_name
        ORDER BY m.name
        """

        try:
            with self.connection.get_session() as session:
                result = session.run(query, repo_url=repo_url)
                modules = []

                for record in result:
                    modules.append({
                        'name': record['name'],
                        'file_path': record['file_path'],
                        'layer': record['layer'] or 'unknown',
                        'domain': record['domain'] or 'unknown',
                        'module_type': record['module_type'] or 'unknown',
                        'complexity_score': record['complexity_score'] or 0.0,
                        'lines_of_code': record['lines_of_code'] or 0,
                        'cluster_name': record['cluster_name'],  # Có thể null nếu không có relationship
                    })
                return modules

        except Exception as e:
            logger.error(f'Error getting all modules: {e}')
            return []
