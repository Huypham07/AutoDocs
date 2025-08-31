from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from domain.preparator.architecture_knowledge import ArchitectureKnowledge
from domain.preparator.architecture_knowledge import CommunicationPattern
from domain.preparator.architecture_knowledge import DataFlowPath
from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArchitectureQuery:
    """Query structure for architecture information."""
    query_type: str  # "overview", "module_details", "data_flow", "communication", "dependencies"
    target: Optional[str] = None  # specific module/cluster name
    filters: Optional[Dict[str, Any]] = None
    context: str = ''  # additional context for the query


@dataclass
class QueryResult:
    """Result of an architecture query."""
    query_type: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    related_items: Optional[List[str]] = None


class ArchitectureQueryService:
    """Service for querying architecture knowledge graph."""

    def __init__(self, knowledge: ArchitectureKnowledge):
        self.knowledge = knowledge

    def query(self, query: ArchitectureQuery) -> QueryResult:
        """Execute an architecture query."""
        if query.query_type == 'overview':
            return self._get_overview_info(query)
        elif query.query_type == 'module_details':
            return self._get_module_details(query)
        elif query.query_type == 'data_flow':
            return self._get_data_flow_info(query)
        elif query.query_type == 'communication':
            return self._get_communication_info(query)
        elif query.query_type == 'dependencies':
            return self._get_dependency_info(query)
        elif query.query_type == 'technology_stack':
            return self._get_technology_stack_info(query)
        elif query.query_type == 'cluster_analysis':
            return self._get_cluster_analysis(query)
        else:
            return QueryResult(
                query_type=query.query_type,
                content='Unknown query type',
                metadata={},
            )

    def _get_overview_info(self, query: ArchitectureQuery) -> QueryResult:
        """Get high-level architecture overview."""
        overview = self.knowledge.overview

        content = f"""
## System Overview: {overview.system_name}

### Architecture Summary
{overview.executive_summary}

### System Statistics
- **Total Modules**: {overview.total_modules}
- **Logical Clusters**: {overview.total_clusters}
- **Overall Modularity**: {overview.overall_modularity:.2f}

### Layer Distribution
{self._format_distribution(overview.layer_distribution)}

### Domain Distribution
{self._format_distribution(overview.domain_distribution)}

### Architectural Patterns
{self._format_list(overview.architectural_patterns)}

### Design Principles
{self._format_list(overview.design_principles)}

### Technology Stack
{self._format_tech_stack()}

### Critical Dependencies
{self._format_list(overview.critical_dependencies[:5])}

### Architecture Description
{overview.architecture_description}
        """.strip()

        return QueryResult(
            query_type='overview',
            content=content,
            metadata={
                'total_modules': overview.total_modules,
                'total_clusters': overview.total_clusters,
                'modularity': overview.overall_modularity,
            },
        )

    def _get_module_details(self, query: ArchitectureQuery) -> QueryResult:
        """Get detailed information about a specific module."""
        target_module = query.target
        if not target_module:
            return QueryResult(
                query_type='module_details',
                content='No target module specified',
                metadata={},
            )

        # Find the module
        module = None
        for m in self.knowledge.modules:
            if m.id == target_module or m.name == target_module:
                module = m
                break

        if not module:
            return QueryResult(
                query_type='module_details',
                content=f"Module '{target_module}' not found",
                metadata={},
            )

        content = f"""
## Module: {module.name}

### Basic Information
- **File Path**: `{module.file_path}`
- **Module Type**: {module.module_type}
- **Layer**: {module.layer}
- **Domain**: {module.domain}
- **Lines of Code**: {module.lines_of_code}
- **Complexity Level**: {module.complexity_level}
- **Centrality Score**: {module.centrality_score:.3f}

### Purpose
{module.purpose}

### Key Functions
{self._format_list(module.key_functions)}

### Dependencies
{self._format_list(module.dependencies)}

### Dependents
{self._format_list(module.dependents)}

### Keywords
{', '.join(module.keywords)}
        """.strip()

        return QueryResult(
            query_type='module_details',
            content=content,
            metadata={
                'module_id': module.id,
                'layer': module.layer,
                'domain': module.domain,
                'complexity': module.complexity_level,
            },
            related_items=module.dependencies + module.dependents,
        )

    def _get_data_flow_info(self, query: ArchitectureQuery) -> QueryResult:
        """Get data flow information."""
        target = query.target

        if target:
            # Get flows involving specific module
            relevant_flows = [
                flow for flow in self.knowledge.data_flows
                if target in flow.source_module or target in flow.target_module
            ]
        else:
            # Get all major data flows
            relevant_flows = self.knowledge.data_flows[:10]  # Limit to top 10

        content = '## Data Flow Analysis\n\n'

        if not relevant_flows:
            content += 'No significant data flows identified.'
        else:
            for flow in relevant_flows:
                content += f"""
### {flow.path_id}
- **Source**: {flow.source_module}
- **Target**: {flow.target_module}
- **Description**: {flow.flow_description}
"""
                if flow.intermediate_modules:
                    content += f"- **Intermediate Modules**: {', '.join(flow.intermediate_modules)}\n"
                if flow.data_types:
                    content += f"- **Data Types**: {', '.join(flow.data_types)}\n"
                content += '\n'

        return QueryResult(
            query_type='data_flow',
            content=content.strip(),
            metadata={'flow_count': len(relevant_flows)},
        )

    def _get_communication_info(self, query: ArchitectureQuery) -> QueryResult:
        """Get communication pattern information."""
        target = query.target

        if target:
            relevant_patterns = [
                pattern for pattern in self.knowledge.communication_patterns
                if target in pattern.source_module or target in pattern.target_module
            ]
        else:
            relevant_patterns = self.knowledge.communication_patterns[:10]

        content = '## Communication Patterns\n\n'

        if not relevant_patterns:
            content += 'No communication patterns identified.'
        else:
            # Group by pattern type
            pattern_groups: Dict[str, List[Dict[str, Any]]] = {}
            for pattern in relevant_patterns:
                pattern_type = pattern.pattern_type
                if pattern_type not in pattern_groups:
                    pattern_groups[pattern_type] = []
                pattern_groups[pattern_type].append(pattern)

            for pattern_type, patterns in pattern_groups.items():
                content += f'### {pattern_type.title()} Communication\n\n'
                for pattern in patterns:
                    content += f"""
- **{pattern.source_module}** → **{pattern.target_module}**
  - Protocol: {pattern.protocol}
  - Frequency: {pattern.frequency}
  - Description: {pattern.description}
"""
                content += '\n'

        return QueryResult(
            query_type='communication',
            content=content.strip(),
            metadata={'pattern_count': len(relevant_patterns)},
        )

    def _get_dependency_info(self, query: ArchitectureQuery) -> QueryResult:
        """Get dependency information."""
        target = query.target

        if target:
            # Find module and get its dependencies
            module = next((m for m in self.knowledge.modules if m.id == target or m.name == target), None)
            if not module:
                return QueryResult(
                    query_type='dependencies',
                    content=f"Module '{target}' not found",
                    metadata={},
                )

            content = f"""
## Dependencies for {module.name}

### Direct Dependencies
{self._format_list(module.dependencies)}

### Modules Dependent on This
{self._format_list(module.dependents)}

### Dependency Analysis
- **Dependency Count**: {len(module.dependencies)}
- **Dependent Count**: {len(module.dependents)}
- **Coupling Level**: {'High' if len(module.dependencies) > 5 else 'Medium' if len(module.dependencies) > 2 else 'Low'}
            """.strip()
        else:
            # General dependency analysis
            content = f"""
## System Dependency Analysis

### Critical Dependencies
{self._format_list(self.knowledge.overview.critical_dependencies)}

### Circular Dependencies
{self._format_list(self.knowledge.overview.circular_dependencies)}

### High-Coupling Modules
"""
            high_coupling_modules = [
                m for m in self.knowledge.modules
                if len(m.dependencies) > 5
            ]
            content += self._format_list([f'{m.name} ({len(m.dependencies)} dependencies)' for m in high_coupling_modules])

        return QueryResult(
            query_type='dependencies',
            content=content,
            metadata={},
        )

    def _get_technology_stack_info(self, query: ArchitectureQuery) -> QueryResult:
        """Get technology stack information."""
        tech_stack = self.knowledge.technology_stack

        if not tech_stack:
            return QueryResult(
                query_type='technology_stack',
                content='No technology stack information available',
                metadata={},
            )

        content = f"""
## Technology Stack

### Programming Languages
{self._format_list(tech_stack.languages)}

### Frameworks & Libraries
{self._format_list(tech_stack.frameworks)}

### Databases
{self._format_list(tech_stack.databases)}

### External Services
{self._format_list(tech_stack.external_services)}

### Deployment Information
{self._format_dict(tech_stack.deployment_info)}
        """.strip()

        return QueryResult(
            query_type='technology_stack',
            content=content,
            metadata={
                'languages': tech_stack.languages,
                'frameworks': tech_stack.frameworks,
                'databases': tech_stack.databases,
            },
        )

    def _get_cluster_analysis(self, query: ArchitectureQuery) -> QueryResult:
        """Get cluster analysis information."""
        target = query.target

        if target:
            # Get specific cluster
            cluster = next((c for c in self.knowledge.clusters if c.id == target or c.name == target), None)
            if not cluster:
                return QueryResult(
                    query_type='cluster_analysis',
                    content=f"Cluster '{target}' not found",
                    metadata={},
                )

            content = f"""
## Cluster Analysis: {cluster.name}

### Overview
- **Purpose**: {cluster.purpose}
- **Size**: {cluster.size} modules
- **Layer**: {cluster.layer}
- **Domain**: {cluster.domain}
- **Complexity Level**: {cluster.complexity_level}

### Quality Metrics
- **Internal Cohesion**: {cluster.cohesion:.2f}
- **External Coupling**: {cluster.coupling:.2f}

### Modules in Cluster
{self._format_list(cluster.modules)}

### Interface Modules
{self._format_list(cluster.interfaces)}

### External Dependencies
{self._format_list(cluster.external_dependencies)}

### Interaction Patterns
{self._format_list(cluster.interaction_patterns)}
            """.strip()
        else:
            # General cluster overview
            content = f"""
## Cluster Analysis Overview

### Total Clusters: {len(self.knowledge.clusters)}

### Cluster Summary
"""
            for cluster in self.knowledge.clusters:
                content += f"""
#### {cluster.name}
- **Purpose**: {cluster.purpose}
- **Size**: {cluster.size} modules
- **Layer**: {cluster.layer}
- **Domain**: {cluster.domain}
- **Cohesion/Coupling**: {cluster.cohesion:.2f}/{cluster.coupling:.2f}

"""

        return QueryResult(
            query_type='cluster_analysis',
            content=content,
            metadata={},
        )

    # Helper formatting methods
    def _format_list(self, items: List[str]) -> str:
        """Format list as markdown bullets."""
        if not items:
            return '- None'
        return '\n'.join(f'- {item}' for item in items)

    def _format_distribution(self, dist: Dict[str, int]) -> str:
        """Format distribution dictionary."""
        if not dist:
            return '- No data available'
        total = sum(dist.values())
        return '\n'.join(
            f'- **{key}**: {value} ({value/total*100:.1f}%)'
            for key, value in sorted(dist.items(), key=lambda x: x[1], reverse=True)
        )

    def _format_dict(self, d: Dict[str, str]) -> str:
        """Format dictionary as key-value pairs."""
        if not d:
            return '- No information available'
        return '\n'.join(f'- **{k}**: {v}' for k, v in d.items())

    def _format_tech_stack(self) -> str:
        """Format technology stack summary."""
        if not self.knowledge.technology_stack:
            return '- Technology stack information not available'

        tech = self.knowledge.technology_stack
        items = []
        if tech.languages:
            items.append(f"Languages: {', '.join(tech.languages)}")
        if tech.frameworks:
            items.append(f"Frameworks: {', '.join(tech.frameworks)}")
        if tech.databases:
            items.append(f"Databases: {', '.join(tech.databases)}")

        return '\n'.join(f'- {item}' for item in items) if items else '- No technology information available'


def create_outline_queries() -> List[ArchitectureQuery]:
    """Create standard queries for outline generation."""
    return [
        ArchitectureQuery(query_type='overview', context='high-level system overview'),
        ArchitectureQuery(query_type='technology_stack', context='technology choices and rationale'),
        ArchitectureQuery(query_type='cluster_analysis', context='architectural components and modules'),
        ArchitectureQuery(query_type='data_flow', context='data flow and processing pipelines'),
        ArchitectureQuery(query_type='communication', context='inter-component communication'),
        ArchitectureQuery(query_type='dependencies', context='system dependencies and coupling'),
    ]


def create_content_queries(target_module: str) -> List[ArchitectureQuery]:
    """Create queries for detailed content generation about a specific component."""
    return [
        ArchitectureQuery(query_type='module_details', target=target_module, context='detailed module analysis'),
        ArchitectureQuery(query_type='dependencies', target=target_module, context='module dependencies'),
        ArchitectureQuery(query_type='data_flow', target=target_module, context='data flows involving this module'),
        ArchitectureQuery(query_type='communication', target=target_module, context='communication patterns'),
    ]
