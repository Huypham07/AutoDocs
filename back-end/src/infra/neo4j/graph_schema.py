from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional


# Node Types
class NodeType(Enum):
    MODULE = 'Module'
    CLASS = 'Class'
    FUNCTION = 'Function'
    CLUSTER = 'Cluster'
    LAYER = 'Layer'
    DOMAIN = 'Domain'
    TECHNOLOGY = 'Technology'
    DATA_FLOW = 'DataFlow'


# Relationship Types
class RelationshipType(Enum):
    IMPORTS = 'IMPORTS'
    CALLS = 'CALLS'
    INHERITS = 'INHERITS'
    CONTAINS = 'CONTAINS'
    DEPENDS_ON = 'DEPENDS_ON'
    BELONGS_TO = 'BELONGS_TO'
    COMMUNICATES = 'COMMUNICATES'
    DATA_FLOW = 'DATA_FLOW'
    IMPLEMENTS = 'IMPLEMENTS'
    USES = 'USES'


# Specific Node Models
@dataclass
class ModuleNode:
    """Module node with code-specific properties."""
    name: str
    file_path: str
    lines_of_code: int = 0
    complexity_score: float = 0.0
    layer: str = 'unknown'
    domain: str = 'unknown'
    module_type: str = 'unknown'
    repo_url: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name


@dataclass
class ClusterNode:
    """Cluster node representing logical groupings."""
    name: str
    purpose: str = ''
    size: int = 0
    cohesion: float = 0.0
    coupling: float = 0.0
    repo_url: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name


@dataclass
class TechnologyNode:
    """Technology node representing tech stack."""
    name: str
    tech_type: str = 'unknown'
    version: Optional[str] = None
    repo_url: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.name


class CypherQueries:
    """Collection of Cypher query templates."""

    # System Overview
    SYSTEM_OVERVIEW = """
    MATCH (m:Module)
    WHERE m.repo_url = $repo_url
    OPTIONAL MATCH (c:Cluster)
    WHERE c.repo_url = $repo_url
    OPTIONAL MATCH (t:Technology)
    WHERE t.repo_url = $repo_url
    RETURN COUNT(DISTINCT m) as total_modules,
           COUNT(DISTINCT c) as total_clusters,
           COUNT(DISTINCT t) as total_technologies,
           COLLECT(DISTINCT m.layer) as layers,
           COLLECT(DISTINCT m.domain) as domains,
           COLLECT(DISTINCT m.module_type) as module_types
    """

    # Module Details
    MODULE_DETAILS = """
    MATCH (m:Module {name: $module_name, repo_url: $repo_url})
    OPTIONAL MATCH (m)-[r1:DEPENDS_ON]->(dep:Module)
    WHERE dep.repo_url = $repo_url
    OPTIONAL MATCH (dependent:Module)-[r2:DEPENDS_ON]->(m)
    WHERE dependent.repo_url = $repo_url
    OPTIONAL MATCH (m)-[:BELONGS_TO]->(c:Cluster)
    WHERE c.repo_url = $repo_url
    RETURN {
        name: m.name,
        file_path: m.file_path,
        layer: m.layer,
        domain: m.domain,
        module_type: m.module_type,
        complexity_score: m.complexity_score,
        lines_of_code: m.lines_of_code,
        dependencies: COLLECT(DISTINCT dep.name),
        dependents: COLLECT(DISTINCT dependent.name),
        cluster_name: c.name
    } as module_details
    """

    # Data Flows
    DATA_FLOWS = """
    MATCH (source:Module {repo_url: $repo_url})-[r:DATA_FLOW]->(target:Module {repo_url: $repo_url})
    WHERE ($target_module IS NULL OR source.name = $target_module OR target.name = $target_module)
    RETURN {
        source: source.name,
        target: target.name,
        flow_type: r.flow_type
    } as flow
    """

    # Cluster Analysis
    CLUSTER_ANALYSIS = """
    MATCH (c:Cluster {repo_url: $repo_url})
    WHERE ($cluster_name IS NULL OR c.name = $cluster_name)
    OPTIONAL MATCH (c)-[:CONTAINS]->(m:Module)
    WHERE m.repo_url = $repo_url
    OPTIONAL MATCH (m)-[:DEPENDS_ON]->(external:Module)
    WHERE NOT exists((c)-[:CONTAINS]->(external)) AND external.repo_url = $repo_url
    RETURN {
        name: c.name,
        purpose: c.purpose,
        size: c.size,
        cohesion: c.cohesion,
        coupling: c.coupling,
        modules: COLLECT(DISTINCT m.name),
        external_dependencies: COLLECT(DISTINCT external.name)
    } as cluster
    """

    # Technology Stack
    TECHNOLOGY_STACK = """
    MATCH (t:Technology {repo_url: $repo_url})
    RETURN t.tech_type as category, COLLECT(t.name) as technologies
    """

    # Communication Patterns
    COMMUNICATION_PATTERNS = """
    MATCH (source:Module {repo_url: $repo_url})-[r:COMMUNICATES]->(target:Module {repo_url: $repo_url})
    WHERE ($target_module IS NULL OR source.name = $target_module OR target.name = $target_module)
    RETURN {
        source_module: source.name,
        target_module: target.name,
        communication_type: r.communication_type,
        protocol: r.protocol,
        frequency: r.frequency
    } as pattern
    """

    # Circular Dependencies
    CIRCULAR_DEPENDENCIES = """
    MATCH (m1:Module {repo_url: $repo_url})-[:DEPENDS_ON*2..]->(m2:Module {repo_url: $repo_url})
    WHERE m1.name = m2.name
    RETURN DISTINCT m1.name as module1, m2.name as module2
    """

    # High Coupling Modules
    HIGH_COUPLING_MODULES = """
    MATCH (m:Module {repo_url: $repo_url})
    OPTIONAL MATCH (m)-[r1:DEPENDS_ON]->(out:Module {repo_url: $repo_url})
    OPTIONAL MATCH (in:Module {repo_url: $repo_url})-[r2:DEPENDS_ON]->(m)
    WITH m,
         COUNT(DISTINCT out) as outgoing_dependencies,
         COUNT(DISTINCT in) as incoming_dependencies,
         COUNT(DISTINCT out) + COUNT(DISTINCT in) as total_coupling
    WHERE total_coupling > 5
    RETURN {
        module_name: m.name,
        outgoing_dependencies: outgoing_dependencies,
        incoming_dependencies: incoming_dependencies,
        total_coupling: total_coupling
    } as high_coupling_module
    ORDER BY total_coupling DESC
    """

    # Delete Repository Data
    DELETE_REPOSITORY_DATA = """
    MATCH (n {repo_url: $repo_url})
    DETACH DELETE n
    """
