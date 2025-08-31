from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from domain.preparator.rag_serializer import ClusterSummary
from domain.preparator.rag_serializer import ModuleSummary
from domain.preparator.rag_serializer import RAGOptimizedData


@dataclass
class DataFlowPath:
    """Represents a data flow path through the architecture."""
    path_id: str
    source_module: str
    target_module: str
    intermediate_modules: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    flow_description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path_id': self.path_id,
            'source_module': self.source_module,
            'target_module': self.target_module,
            'intermediate_modules': self.intermediate_modules,
            'data_types': self.data_types,
            'flow_description': self.flow_description,
        }


@dataclass
class CommunicationPattern:
    """Represents communication pattern between modules/clusters."""
    pattern_id: str
    pattern_type: str  # "synchronous", "asynchronous", "event-driven", "request-response"
    source_module: str
    target_module: str
    protocol: str = 'unknown'  # "HTTP", "message_queue", "direct_call", "database"
    frequency: str = 'low'  # "low", "medium", "high"
    description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'source_module': self.source_module,
            'target_module': self.target_module,
            'protocol': self.protocol,
            'frequency': self.frequency,
            'description': self.description,
        }


@dataclass
class TechnologyStack:
    """Technology stack information."""
    frameworks: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    external_services: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    deployment_info: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frameworks': self.frameworks,
            'databases': self.databases,
            'external_services': self.external_services,
            'languages': self.languages,
            'deployment_info': self.deployment_info,
        }


@dataclass
class ArchitectureKnowledge(RAGOptimizedData):
    """Enhanced RAG data with architecture-specific knowledge."""

    # Architecture-specific additions
    data_flows: List[DataFlowPath] = field(default_factory=list)
    communication_patterns: List[CommunicationPattern] = field(default_factory=list)
    technology_stack: Optional[TechnologyStack] = None

    # Architecture insights
    scalability_bottlenecks: List[str] = field(default_factory=list)
    security_considerations: List[str] = field(default_factory=list)
    performance_hotspots: List[str] = field(default_factory=list)

    # Documentation metadata
    documentation_version: str = '1.0'
    last_analysis_date: str = ''

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'data_flows': [flow.to_dict() for flow in self.data_flows],
            'communication_patterns': [pattern.to_dict() for pattern in self.communication_patterns],
            'technology_stack': self.technology_stack.to_dict() if self.technology_stack else None,
            'scalability_bottlenecks': self.scalability_bottlenecks,
            'security_considerations': self.security_considerations,
            'performance_hotspots': self.performance_hotspots,
            'documentation_version': self.documentation_version,
            'last_analysis_date': self.last_analysis_date,
        })
        return base_dict


class ArchitectureKnowledgeExtractor:
    """Extracts architecture-specific knowledge from RAG data."""

    def __init__(self, rag_data: RAGOptimizedData):
        self.rag_data = rag_data

    def extract_data_flows(self) -> List[DataFlowPath]:
        """Extract data flow paths from the module dependencies."""
        data_flows = []

        # Analyze dependency chains to identify data flows
        for module in self.rag_data.modules:
            if module.layer in ['data', 'infrastructure']:
                # Find modules that depend on this data module
                dependents = module.dependents
                for dependent in dependents:
                    if dependent in self.rag_data.dependency_index:
                        flow = DataFlowPath(
                            path_id=f'{module.id}_to_{dependent}',
                            source_module=module.id,
                            target_module=dependent,
                            flow_description=f'Data flow from {module.name} to {dependent}',
                        )
                        data_flows.append(flow)

        return data_flows

    def extract_communication_patterns(self) -> List[CommunicationPattern]:
        """Extract communication patterns between modules."""
        patterns = []

        # Analyze adjacency lists for communication patterns
        for module_id, neighbors in self.rag_data.adjacency_lists.items():
            module = next((m for m in self.rag_data.modules if m.id == module_id), None)
            if not module:
                continue

            for neighbor_id in neighbors:
                neighbor = next((m for m in self.rag_data.modules if m.id == neighbor_id), None)
                if not neighbor:
                    continue

                # Determine communication pattern based on layers
                pattern_type = self._determine_pattern_type(module, neighbor)
                protocol = self._determine_protocol(module, neighbor)

                pattern = CommunicationPattern(
                    pattern_id=f'{module_id}_to_{neighbor_id}',
                    pattern_type=pattern_type,
                    source_module=module_id,
                    target_module=neighbor_id,
                    protocol=protocol,
                    description=f'{pattern_type} communication from {module.name} to {neighbor.name}',
                )
                patterns.append(pattern)

        return patterns

    def extract_technology_stack(self) -> TechnologyStack:
        """Extract technology stack information."""
        frameworks = set()
        databases = set()
        languages = set()

        # Analyze module keywords and content for tech stack
        for module in self.rag_data.modules:
            # Look for common framework keywords
            for keyword in module.keywords:
                if keyword.lower() in ['fastapi', 'django', 'flask', 'express', 'react', 'vue', 'angular']:
                    frameworks.add(keyword)
                elif keyword.lower() in ['mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch']:
                    databases.add(keyword)
                elif keyword.lower() in ['python', 'javascript', 'typescript', 'java', 'go']:
                    languages.add(keyword)

        return TechnologyStack(
            frameworks=list(frameworks),
            databases=list(databases),
            languages=list(languages),
        )

    def _determine_pattern_type(self, source: ModuleSummary, target: ModuleSummary) -> str:
        """Determine communication pattern type based on module characteristics."""
        if source.layer == 'presentation' and target.layer == 'business':
            return 'request-response'
        elif 'async' in source.keywords or 'queue' in source.keywords:
            return 'asynchronous'
        elif 'event' in source.keywords:
            return 'event-driven'
        else:
            return 'synchronous'

    def _determine_protocol(self, source: ModuleSummary, target: ModuleSummary) -> str:
        """Determine communication protocol."""
        if 'api' in source.keywords or 'http' in source.keywords:
            return 'HTTP'
        elif 'rabbit' in source.keywords or 'queue' in source.keywords:
            return 'message_queue'
        elif target.layer == 'data':
            return 'database'
        else:
            return 'direct_call'

    def create_enhanced_knowledge(self) -> ArchitectureKnowledge:
        """Create enhanced architecture knowledge."""
        # Extract architecture-specific information
        data_flows = self.extract_data_flows()
        communication_patterns = self.extract_communication_patterns()
        technology_stack = self.extract_technology_stack()

        # Analyze for bottlenecks and considerations
        scalability_bottlenecks = self._identify_scalability_bottlenecks()
        security_considerations = self._identify_security_considerations()
        performance_hotspots = self._identify_performance_hotspots()

        return ArchitectureKnowledge(
            overview=self.rag_data.overview,
            modules=self.rag_data.modules,
            clusters=self.rag_data.clusters,
            content_index=self.rag_data.content_index,
            dependency_index=self.rag_data.dependency_index,
            keyword_index=self.rag_data.keyword_index,
            adjacency_lists=self.rag_data.adjacency_lists,
            reverse_adjacency_lists=self.rag_data.reverse_adjacency_lists,
            generation_timestamp=self.rag_data.generation_timestamp,
            version=self.rag_data.version,
            data_flows=data_flows,
            communication_patterns=communication_patterns,
            technology_stack=technology_stack,
            scalability_bottlenecks=scalability_bottlenecks,
            security_considerations=security_considerations,
            performance_hotspots=performance_hotspots,
        )

    def _identify_scalability_bottlenecks(self) -> List[str]:
        """Identify potential scalability bottlenecks."""
        bottlenecks = []

        # High centrality modules could be bottlenecks
        for module in self.rag_data.modules:
            if module.centrality_score > 0.1:
                bottlenecks.append(f'{module.name} has high centrality score ({module.centrality_score:.3f})')

        return bottlenecks

    def _identify_security_considerations(self) -> List[str]:
        """Identify security considerations."""
        considerations = []

        # Look for authentication/authorization modules
        for module in self.rag_data.modules:
            if any(keyword in ['auth', 'security', 'token', 'password'] for keyword in module.keywords):
                considerations.append(f'Security-critical module: {module.name}')

        return considerations

    def _identify_performance_hotspots(self) -> List[str]:
        """Identify performance hotspots."""
        hotspots = []

        # High complexity modules could be performance hotspots
        for module in self.rag_data.modules:
            if module.complexity_level == 'high':
                hotspots.append(f'High complexity module: {module.name}')

        return hotspots
