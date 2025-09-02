from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from infra.neo4j.graph_repository import GraphRepository
from langchain.llms.base import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from shared.logging import get_logger
from shared.utils import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class GraphQuery:
    """Represents a graph query with context."""
    query_type: str
    target: Optional[str] = None
    repo_url: str = ''
    context: str = ''


@dataclass
class GraphContext:
    """Graph-based context for LLM."""
    query_type: str
    structured_data: Dict[str, Any]
    narrative_text: str
    metadata: Dict[str, Any]


class GraphContextBuilder:
    """Builds context from Neo4j graph queries."""

    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def build_context(self, query: GraphQuery) -> GraphContext:
        """Build context based on graph query type."""

        if query.query_type == 'outline_generation':
            return self._build_outline_context(query)

        elif query.query_type == 'page_content_generation':
            return self._build_page_content_context(query)

        elif query.query_type == 'chat_qa':
            return self._build_chat_qa_context(query)

        else:
            return self._build_general_context(query)

    def _build_overview_context(self, query: GraphQuery) -> GraphContext:
        """Build system overview context."""
        overview = self.graph_repo.get_system_overview(query.repo_url)

        if not overview:
            return GraphContext(
                query_type='system_overview',
                structured_data={},
                narrative_text='No system overview data available.',
                metadata={},
            )

        # Create narrative text
        narrative = f"""
## System Architecture Overview

### System Statistics
- **Total Modules**: {overview.get('total_modules', 0)}
- **Logical Clusters**: {overview.get('total_clusters', 0)}
- **Technology Components**: {overview.get('total_technologies', 0)}

### Architectural Layers
The system is organized across {len(overview.get('layers', []))} architectural layers:
{self._format_list(overview.get('layers', []))}

### Functional Domains
The system spans {len(overview.get('domains', []))} functional domains:
{self._format_list(overview.get('domains', []))}

### Module Types
The codebase includes the following module types:
{self._format_list(overview.get('module_types', []))}
        """.strip()

        return GraphContext(
            query_type='system_overview',
            structured_data=overview,
            narrative_text=narrative,
            metadata={'source': 'neo4j_graph_query'},
        )

    def _build_module_context(self, query: GraphQuery) -> GraphContext:
        """Build specific module context."""
        if not query.target:
            return GraphContext(
                query_type='module_details',
                structured_data={},
                narrative_text='No module specified for detailed analysis.',
                metadata={},
            )

        module_details = self.graph_repo.get_module_details(query.target, query.repo_url)

        if not module_details:
            return GraphContext(
                query_type='module_details',
                structured_data={},
                narrative_text=f"Module '{query.target}' not found in the system.",
                metadata={},
            )

        # Create narrative text
        narrative = f"""
## Module Analysis: {module_details['name']}

### Basic Information
- **File Path**: `{module_details['file_path']}`
- **Architectural Layer**: {module_details['layer']}
- **Functional Domain**: {module_details['domain']}
- **Module Type**: {module_details['module_type']}
- **Complexity Score**: {module_details['complexity_score']:.2f}
- **Lines of Code**: {module_details['lines_of_code']}

### Dependency Analysis
- **Direct Dependencies**: {len(module_details.get('dependencies', []))} modules
- **Dependent Modules**: {len(module_details.get('dependents', []))} modules

#### Dependencies:
{self._format_list(module_details.get('dependencies', []))}

#### Dependents:
{self._format_list(module_details.get('dependents', []))}

### Cluster Information
- **Belongs to Cluster**: {module_details.get('cluster_name', 'Not clustered')}
        """.strip()

        return GraphContext(
            query_type='module_details',
            structured_data=module_details,
            narrative_text=narrative,
            metadata={'module_name': query.target},
        )

    def _build_dataflow_context(self, query: GraphQuery) -> GraphContext:
        """Build data flow context."""
        flows = self.graph_repo.get_data_flows(query.repo_url, query.target)

        narrative = '## Data Flow Analysis\n\n'

        if not flows:
            narrative += 'No data flows identified in the system.'
        else:
            if query.target:
                narrative += f'### Data flows involving {query.target}:\n\n'
            else:
                narrative += f'### System-wide data flows ({len(flows)} total):\n\n'

            for flow in flows:
                narrative += f"- **{flow['source']}** → **{flow['target']}** ({flow['flow_type']})\n"

        return GraphContext(
            query_type='data_flows',
            structured_data={'flows': flows},
            narrative_text=narrative,
            metadata={'flow_count': len(flows)},
        )

    def _build_cluster_context(self, query: GraphQuery) -> GraphContext:
        """Build cluster analysis context."""
        clusters = self.graph_repo.get_cluster_analysis(query.repo_url, query.target)

        narrative = '## Cluster Analysis\n\n'

        if not clusters:
            narrative += 'No clusters defined in the system.'
        else:
            if query.target:
                cluster = clusters[0] if clusters else {}
                narrative += f"### Cluster: {cluster.get('name', 'Unknown')}\n\n"
                narrative += f"- **Purpose**: {cluster.get('purpose', 'Not specified')}\n"
                narrative += f"- **Size**: {cluster.get('size', 0)} modules\n"
                narrative += f"- **Cohesion**: {cluster.get('cohesion', 0):.2f}\n"
                narrative += f"- **Coupling**: {cluster.get('coupling', 0):.2f}\n\n"
                narrative += f"#### Modules in cluster:\n{self._format_list(cluster.get('modules', []))}\n\n"
                narrative += f"#### External dependencies:\n{self._format_list(cluster.get('external_dependencies', []))}\n"
            else:
                narrative += f'### System clusters ({len(clusters)} total):\n\n'
                for cluster in clusters:
                    narrative += f"#### {cluster['name']}\n"
                    narrative += f"- Purpose: {cluster['purpose']}\n"
                    narrative += f"- Size: {cluster['size']} modules\n"
                    narrative += f"- Cohesion/Coupling: {cluster['cohesion']:.2f}/{cluster['coupling']:.2f}\n\n"

        return GraphContext(
            query_type='cluster_analysis',
            structured_data={'clusters': clusters},
            narrative_text=narrative,
            metadata={'cluster_count': len(clusters)},
        )

    def _build_technology_context(self, query: GraphQuery) -> GraphContext:
        """Build technology stack context."""
        tech_stack = self.graph_repo.get_technology_stack(query.repo_url)

        narrative = '## Technology Stack\n\n'

        if not tech_stack:
            narrative += 'No technology information available.'
        else:
            for tech_type, technologies in tech_stack.items():
                narrative += f'### {tech_type.title()}\n'
                narrative += self._format_list(technologies) + '\n\n'

        return GraphContext(
            query_type='technology_stack',
            structured_data=tech_stack,
            narrative_text=narrative,
            metadata={'tech_categories': list(tech_stack.keys())},
        )

    def _build_communication_context(self, query: GraphQuery) -> GraphContext:
        """Build communication patterns context."""
        patterns = self.graph_repo.get_communication_patterns(query.repo_url, query.target)

        narrative = '## Communication Patterns\n\n'

        if not patterns:
            narrative += 'No communication patterns identified.'
        else:
            if query.target:
                narrative += f'### Communication patterns for {query.target}:\n\n'
            else:
                narrative += f'### System communication patterns ({len(patterns)} total):\n\n'

            # Group by communication type
            pattern_groups: Dict[str, List[Dict[str, Any]]] = {}
            for pattern in patterns:
                comm_type = pattern['communication_type'] or 'unknown'
                if comm_type not in pattern_groups:
                    pattern_groups[comm_type] = []
                pattern_groups[comm_type].append(pattern)

            for comm_type, type_patterns in pattern_groups.items():
                narrative += f'#### {comm_type.title()} Communication\n\n'
                for pattern in type_patterns:
                    narrative += f"- **{pattern['source_module']}** → **{pattern['target_module']}**\n"
                    if pattern['protocol']:
                        narrative += f"  - Protocol: {pattern['protocol']}\n"
                    if pattern['frequency']:
                        narrative += f"  - Frequency: {pattern['frequency']}\n"
                narrative += '\n'

        return GraphContext(
            query_type='communication_patterns',
            structured_data={'patterns': patterns},
            narrative_text=narrative,
            metadata={'pattern_count': len(patterns)},
        )

    def _build_dependency_context(self, query: GraphQuery) -> GraphContext:
        """Build dependency analysis context."""
        circular_deps = self.graph_repo.get_circular_dependencies(query.repo_url)
        high_coupling = self.graph_repo.get_high_coupling_modules(query.repo_url)

        narrative = '## Dependency Analysis\n\n'

        # Circular dependencies
        narrative += '### Circular Dependencies\n\n'
        if not circular_deps:
            narrative += 'No circular dependencies detected.\n\n'
        else:
            narrative += f'Found {len(circular_deps)} circular dependency pairs:\n\n'
            for dep1, dep2 in circular_deps:
                narrative += f'- **{dep1}** ↔ **{dep2}**\n'
            narrative += '\n'

        # High coupling modules
        narrative += '### High Coupling Modules\n\n'
        if not high_coupling:
            narrative += 'No modules with excessive coupling detected.\n\n'
        else:
            narrative += 'Modules with high coupling (>5 total dependencies):\n\n'
            for module in high_coupling:
                narrative += f"- **{module['module_name']}**: {module['total_coupling']} total dependencies "
                narrative += f"({module['outgoing_dependencies']} outgoing, {module['incoming_dependencies']} incoming)\n"

        return GraphContext(
            query_type='dependencies',
            structured_data={
                'circular_dependencies': circular_deps,
                'high_coupling_modules': high_coupling,
            },
            narrative_text=narrative,
            metadata={
                'circular_count': len(circular_deps),
                'high_coupling_count': len(high_coupling),
            },
        )

    def _build_general_context(self, query: GraphQuery) -> GraphContext:
        """Build general context combining multiple aspects."""
        # Get overview as base
        overview_context = self._build_overview_context(query)

        # Add technology info
        tech_context = self._build_technology_context(query)

        combined_narrative = overview_context.narrative_text + '\n\n' + tech_context.narrative_text

        combined_data = {
            **overview_context.structured_data,
            **tech_context.structured_data,
        }

        return GraphContext(
            query_type='general',
            structured_data=combined_data,
            narrative_text=combined_narrative,
            metadata={'combined_query': True},
        )

    def _format_list(self, items: List[str]) -> str:
        """Format list as markdown bullets."""
        if not items:
            return '- None'
        return '\n'.join(f'- {item}' for item in items if item)

    def _build_outline_context(self, query: GraphQuery) -> GraphContext:
        """Build context for outline generation based on KG structure."""
        # Query tất cả Cluster, Module nodes
        clusters = self.graph_repo.get_cluster_analysis(query.repo_url)
        modules = self.graph_repo.get_all_modules(query.repo_url) if hasattr(self.graph_repo, 'get_all_modules') else []
        dependencies = self.graph_repo.get_circular_dependencies(query.repo_url)
        dataflows = self.graph_repo.get_data_flows(query.repo_url)
        tech_stack = self.graph_repo.get_technology_stack(query.repo_url)

        # Build structured data for outline generation
        structured_data = {
            'clusters': clusters,
            'modules': modules,
            'dependencies': dependencies,
            'dataflows': dataflows,
            'technology_stack': tech_stack,
            'repo_url': query.repo_url,
        }

        # Create narrative text to guide outline generation
        narrative = f"""
## Graph Structure for Documentation Outline

### Available Clusters ({len(clusters)} total)
{self._format_clusters_for_outline(clusters)}

### Module Organization ({len(modules)} total modules)
{self._format_modules_by_layer(modules)}

### Data Flow Patterns ({len(dataflows)} flows)
{self._format_dataflows_for_outline(dataflows)}

### Technology Stack
{self._format_tech_stack_for_outline(tech_stack)}

### Suggested Documentation Structure:
1. **System Introduction**: Overall purpose and architecture overview
2. **Architecture Overview**: High-level system design and clusters
3. **Core Modules**: Detail each significant cluster/module
4. **Data Flow Analysis**: How data moves through the system
5. **Technology Stack**: Technologies and their roles
6. **Integration Patterns**: How components communicate
7. **Deployment Architecture**: If infrastructure info available
        """.strip()

        return GraphContext(
            query_type='outline_generation',
            structured_data=structured_data,
            narrative_text=narrative,
            metadata={
                'cluster_count': len(clusters),
                'module_count': len(modules),
                'dataflow_count': len(dataflows),
                'purpose': 'documentation_outline',
            },
        )

    def _build_page_content_context(self, query: GraphQuery) -> GraphContext:
        """Build detailed context for specific page content generation."""
        target_page = query.target  # Tên trang cần sinh nội dung

        if not target_page:
            return GraphContext(
                query_type='page_content_generation',
                structured_data={},
                narrative_text='No specific page target provided.',
                metadata={},
            )

        # Strategy: Smart context selection based on page type and semantic analysis
        context_data = {}
        target_page.lower()

        # Analyze page type and get focused context
        page_type = self._analyze_page_type(target_page)
        logger.info(f"Detected page type for '{target_page}': {page_type}")

        # Get context based on page type
        try:
            if page_type == 'architecture_overview':
                # Architecture pages need: overview + clusters + high-level flows
                overview = self.graph_repo.get_system_overview(query.repo_url)
                clusters = self.graph_repo.get_cluster_analysis(query.repo_url)
                if overview:
                    context_data['system_overview'] = overview
                if clusters:
                    context_data['clusters'] = clusters

            elif page_type == 'technology_integration':
                # Integration/API pages need: tech stack + communication patterns + some flows
                tech_stack = self.graph_repo.get_technology_stack(query.repo_url)
                patterns = self.graph_repo.get_communication_patterns(query.repo_url)
                if tech_stack:
                    context_data['technology_stack'] = tech_stack
                if patterns:
                    context_data['communication_patterns'] = patterns
                # Add limited dataflows for integration context
                dataflows = self.graph_repo.get_data_flows(query.repo_url)
                if dataflows:
                    context_data['dataflows'] = dataflows[:10]  # Limit to 10 most relevant

            elif page_type == 'data_flow':
                # Data flow pages need: full dataflows + related communication patterns
                dataflows = self.graph_repo.get_data_flows(query.repo_url)
                patterns = self.graph_repo.get_communication_patterns(query.repo_url)
                if dataflows:
                    context_data['dataflows'] = dataflows
                if patterns:
                    context_data['communication_patterns'] = patterns

            elif page_type == 'component_cluster':
                # Component pages need: specific cluster details + modules + dependencies
                cluster_info = self._find_relevant_cluster(target_page, query.repo_url)
                if cluster_info:
                    context_data['specific_cluster'] = cluster_info
                    context_data['clusters'] = [cluster_info]  # Just this cluster
                else:
                    # Fallback to all clusters if can't find specific one
                    clusters = self.graph_repo.get_cluster_analysis(query.repo_url)
                    if clusters:
                        context_data['clusters'] = clusters[:3]  # Limit to 3 most relevant

            elif page_type == 'deployment_infrastructure':
                # Deployment pages need: tech stack + basic overview
                tech_stack = self.graph_repo.get_technology_stack(query.repo_url)
                overview = self.graph_repo.get_system_overview(query.repo_url)
                if tech_stack:
                    context_data['technology_stack'] = tech_stack
                if overview:
                    context_data['system_overview'] = overview

            elif page_type == 'feature_functionality':
                # Feature pages need: related clusters + some communication patterns
                clusters = self.graph_repo.get_cluster_analysis(query.repo_url)
                patterns = self.graph_repo.get_communication_patterns(query.repo_url)
                if clusters:
                    context_data['clusters'] = clusters
                if patterns:
                    context_data['communication_patterns'] = patterns[:5]  # Limit patterns

            else:  # fallback or general
                # General pages get basic overview + limited context
                overview = self.graph_repo.get_system_overview(query.repo_url)
                tech_stack = self.graph_repo.get_technology_stack(query.repo_url)
                if overview:
                    context_data['system_overview'] = overview
                if tech_stack:
                    context_data['technology_stack'] = tech_stack

        except Exception as e:
            logger.warning(f'Error getting focused context for {page_type}: {e}')
            # Minimal fallback
            try:
                overview = self.graph_repo.get_system_overview(query.repo_url)
                if overview:
                    context_data['system_overview'] = overview
            except Exception:
                pass

        # Log context for debugging
        logger.info(f"Page content context for '{target_page}' ({page_type}): {list(context_data.keys())}")

        # Build focused narrative based on page type
        narrative = self._build_focused_page_narrative(target_page, page_type, context_data)

        return GraphContext(
            query_type='page_content_generation',
            structured_data=context_data,
            narrative_text=narrative,
            metadata={
                'target_page': target_page,
                'page_type': page_type,
                'data_types': list(context_data.keys()),
                'context_strategy': 'focused',
            },
        )

    def _analyze_page_type(self, page_title: str) -> str:
        """Analyze page type using LLM for intelligent classification."""
        try:
            # Use LLM for intelligent page type classification
            return self._llm_analyze_page_type(page_title)
        except Exception as e:
            logger.warning(f'LLM page analysis failed, falling back to rule-based: {e}')
            # Fallback to rule-based classification
            return self._rule_based_analyze_page_type(page_title)

    def _llm_analyze_page_type(self, page_title: str) -> str:
        """Use LLM to analyze page type intelligently."""
        page_analysis_prompt = """
You are an expert technical documentation analyst. Analyze the following page title and classify what type of documentation page it should be.

PAGE TYPES:
1. architecture_overview: System overview, high-level architecture, system design
2. technology_integration: Technology stack, framework integration, tool usage
3. data_flow: Data processing, information flow, request/response patterns
4. component_cluster: Specific component groups, module clusters, service groups
5. deployment_infrastructure: Deployment, infrastructure, DevOps, environment setup
6. feature_functionality: Specific features, business functionality, user workflows

PAGE TITLE: "{page_title}"

Consider:
- What type of information would readers expect?
- What documentation structure would be most helpful?
- What technical aspects should be emphasized?

Respond with ONLY the page type (one of the 6 options above):
"""

        try:
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.5-flash-lite-preview-06-17',
                temperature=0.1,
                google_api_key=settings.GOOGLE_API_KEY,
            )

            response = llm.invoke(page_analysis_prompt.format(page_title=page_title))
            page_type = response.content.strip().lower()

            # Validate response
            valid_types = [
                'architecture_overview', 'technology_integration', 'data_flow',
                'component_cluster', 'deployment_infrastructure', 'feature_functionality',
            ]

            if page_type in valid_types:
                logger.info(f"LLM classified '{page_title}' as '{page_type}'")
                return page_type
            else:
                raise ValueError(f'Invalid page type returned: {page_type}')

        except Exception as e:
            logger.error(f'LLM page analysis error: {e}')
            raise

    def _rule_based_analyze_page_type(self, page_title: str) -> str:
        """Fallback rule-based page type analysis."""
        title_lower = page_title.lower()

        # Architecture and system design
        if any(word in title_lower for word in ['architecture', 'system design', 'overview', 'structure']):
            return 'architecture_overview'

        # Technology, integration, API
        elif any(word in title_lower for word in ['integration', 'api', 'llm', 'vector', 'database', 'technology']):
            return 'technology_integration'

        # Data flow and processing
        elif any(word in title_lower for word in ['data flow', 'processing', 'pipeline', 'workflow']):
            return 'data_flow'

        # Deployment and infrastructure
        elif any(word in title_lower for word in ['deployment', 'infrastructure', 'setup', 'configuration']):
            return 'deployment_infrastructure'

        # Component or cluster specific
        elif any(word in title_lower for word in ['component', 'module', 'service', 'backend', 'frontend']):
            return 'component_cluster'

        # Features and functionality
        elif any(word in title_lower for word in ['feature', 'functionality', 'capability', 'user']):
            return 'feature_functionality'

        else:
            return 'feature_functionality'  # Default fallback

    def _find_relevant_cluster(self, page_title: str, repo_url: str) -> Optional[Dict[str, Any]]:
        """Find the most relevant cluster based on page title."""
        try:
            clusters = self.graph_repo.get_cluster_analysis(repo_url)
            if not clusters:
                return None

            title_lower = page_title.lower()

            # Try exact or partial name matching
            for cluster in clusters:
                cluster_name = cluster.get('name', '').lower()
                if cluster_name and cluster_name in title_lower:
                    # Get detailed info for this specific cluster
                    cluster_details = self._get_enhanced_cluster_details(cluster['name'], repo_url)
                    return cluster_details if cluster_details else cluster

            # Try keyword matching with cluster purposes
            for cluster in clusters:
                purpose = cluster.get('purpose', '').lower()
                if purpose and any(word in title_lower for word in purpose.split()):
                    cluster_details = self._get_enhanced_cluster_details(cluster['name'], repo_url)
                    return cluster_details if cluster_details else cluster

            return None

        except Exception as e:
            logger.warning(f'Error finding relevant cluster: {e}')
            return None

    def _get_enhanced_cluster_details(self, cluster_name: str, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get enhanced cluster details including modules, dependencies, and flows."""
        try:
            # Get basic cluster info
            clusters = self.graph_repo.get_cluster_analysis(repo_url)
            target_cluster = None

            for cluster in clusters:
                if cluster.get('name', '').lower() == cluster_name.lower():
                    target_cluster = cluster.copy()
                    break

            if not target_cluster:
                return None

            # Enhance with detailed module information
            cluster_modules = self._get_modules_in_cluster(cluster_name, repo_url)
            if cluster_modules:
                target_cluster['modules'] = cluster_modules

                # Add module summary statistics
                target_cluster['module_layers'] = list({m.get('layer', 'Unknown') for m in cluster_modules})
                target_cluster['module_types'] = list({m.get('module_type', 'Unknown') for m in cluster_modules})
                target_cluster['total_loc'] = sum(m.get('lines_of_code', 0) for m in cluster_modules)
                target_cluster['avg_complexity'] = sum(m.get('complexity_score', 0) for m in cluster_modules) / len(cluster_modules) if cluster_modules else 0

            # Get related data flows
            try:
                all_flows = self.graph_repo.get_data_flows(repo_url)
                cluster_flows = []

                for flow in all_flows:
                    source_module = flow.get('source_module', '')
                    target_module = flow.get('target_module', '')

                    # Check if flow involves any module in this cluster
                    for module in cluster_modules:
                        module_name = module.get('name', '')
                        if module_name in source_module or module_name in target_module:
                            cluster_flows.append(flow)
                            break

                if cluster_flows:
                    target_cluster['related_flows'] = cluster_flows[:10]  # Limit to 10
                    target_cluster['flow_count'] = len(cluster_flows)

            except Exception as e:
                logger.warning(f'Error getting flows for cluster {cluster_name}: {e}')

            # Get cluster dependencies
            try:
                cluster_dependencies = []
                cluster_dependents = []

                for module in cluster_modules:
                    module_deps = module.get('dependencies', [])
                    module_dependents = module.get('dependents', [])

                    # External dependencies (outside this cluster)
                    for dep in module_deps:
                        if not any(dep in m.get('name', '') for m in cluster_modules):
                            cluster_dependencies.append(dep)

                    # External dependents (outside this cluster)
                    for dependent in module_dependents:
                        if not any(dependent in m.get('name', '') for m in cluster_modules):
                            cluster_dependents.append(dependent)

                target_cluster['external_dependencies'] = list(set(cluster_dependencies))
                target_cluster['external_dependents'] = list(set(cluster_dependents))

            except Exception as e:
                logger.warning(f'Error getting dependencies for cluster {cluster_name}: {e}')

            return target_cluster

        except Exception as e:
            logger.warning(f'Error getting enhanced cluster details for {cluster_name}: {e}')
            return None

    def _build_focused_page_narrative(self, target_page: str, page_type: str, context_data: Dict[str, Any]) -> str:
        """Build focused narrative based on page type and available context."""
        narrative = f'## Focused Context for Page: {target_page}\n'
        narrative += f'**Page Type**: {page_type}\n\n'

        if page_type == 'architecture_overview':
            narrative += '### Architecture Overview Content Guidelines\n'
            if 'system_overview' in context_data:
                overview = context_data['system_overview']
                narrative += f"**System Scale**: {overview.get('total_modules', 0)} modules, {overview.get('total_clusters', 0)} clusters\n"
                if overview.get('layers'):
                    narrative += f"**Architectural Layers**: {', '.join(overview.get('layers', []))}\n"
            if 'clusters' in context_data:
                narrative += f"**Available Clusters**: {len(context_data['clusters'])} clusters for architectural organization\n"
            narrative += '\n**Focus**: System structure, architectural patterns, high-level design decisions\n\n'

        elif page_type == 'technology_integration':
            narrative += '### Technology Integration Content Guidelines\n'
            if 'technology_stack' in context_data:
                tech_stack = context_data['technology_stack']
                narrative += f"**Technology Categories**: {', '.join(tech_stack.keys())}\n"
            if 'communication_patterns' in context_data:
                narrative += f"**Integration Patterns**: {len(context_data['communication_patterns'])} communication patterns\n"
            narrative += '\n**Focus**: Technology choices, integration methods, API design, data flow between systems\n\n'

        elif page_type == 'data_flow':
            narrative += '### Data Flow Content Guidelines\n'
            if 'dataflows' in context_data:
                flows = context_data['dataflows']
                data_flow_types = {flow.get('flow_type', 'Unknown') for flow in flows}
                narrative += f"**Data Flow Types**: {', '.join(data_flow_types)}\n"
                narrative += f'**Total Flows**: {len(flows)} data flow relationships\n'
            narrative += '\n**Focus**: Data processing pipelines, information flow, data transformations\n\n'

        elif page_type == 'component_cluster':
            narrative += '### Component/Cluster Content Guidelines\n'
            if 'specific_cluster' in context_data:
                cluster = context_data['specific_cluster']
                narrative += f"**Focused Cluster**: {cluster.get('name', 'Unknown')}\n"
                narrative += f"**Cluster Purpose**: {cluster.get('purpose', 'Cluster for organizing related components')}\n"
                narrative += f"**Cluster Size**: {cluster.get('size', 0)} modules\n"

                # Add cluster statistics
                if cluster.get('total_loc'):
                    narrative += f"**Total Lines of Code**: {cluster.get('total_loc', 0):,}\n"
                if cluster.get('avg_complexity'):
                    narrative += f"**Average Complexity**: {cluster.get('avg_complexity', 0):.2f}\n"

                # Add detailed module information if available
                if 'modules' in cluster and cluster['modules']:
                    modules = cluster['modules']
                    narrative += f'\n#### Cluster Modules ({len(modules)} modules):\n'

                    # Group modules by layer/type for better organization
                    layers: Dict[str, List[Dict[str, Any]]] = {}
                    for module in modules[:15]:  # Show first 15 modules
                        layer = module.get('layer', 'Unknown Layer')
                        if layer not in layers:
                            layers[layer] = []
                        layers[layer].append({
                            'name': module.get('name', 'Unknown'),
                            'path': module.get('file_path', 'Unknown path'),
                            'type': module.get('module_type', 'Unknown type'),
                            'complexity': module.get('complexity_score', 0),
                            'loc': module.get('lines_of_code', 0),
                        })

                    for layer, layer_modules in layers.items():
                        narrative += f'\n**{layer} Layer** ({len(layer_modules)} modules):\n'
                        for mod in layer_modules[:5]:  # Limit to 5 per layer
                            narrative += f"- `{mod['name']}` ({mod['type']}) - Complexity: {mod['complexity']:.1f}, LOC: {mod['loc']}\n"
                            narrative += f"  Path: `{mod['path']}`\n"

                        if len(layer_modules) > 5:
                            narrative += f'  *... and {len(layer_modules) - 5} more modules in this layer*\n'

                    if len(modules) > 15:
                        narrative += f'\n*... and {len(modules) - 15} more modules*\n'

                # Add data flow information
                if cluster.get('related_flows'):
                    flows = cluster['related_flows']
                    narrative += f"\n#### Related Data Flows ({cluster.get('flow_count', len(flows))} total flows):\n"

                    # Group flows by type
                    flow_types: Dict[str, List[Dict[str, Any]]] = {}
                    for flow in flows[:10]:
                        flow_type = flow.get('flow_type', 'data_flow')
                        if flow_type not in flow_types:
                            flow_types[flow_type] = []
                        flow_types[flow_type].append(flow)

                    for flow_type, type_flows in flow_types.items():
                        narrative += f"\n**{flow_type.replace('_', ' ').title()}** ({len(type_flows)} flows):\n"
                        for flow in type_flows[:3]:  # Show top 3 flows per type
                            source = flow.get('source_module', 'Unknown')
                            target = flow.get('target_module', 'Unknown')
                            narrative += f'- `{source}` → `{target}`'
                            if flow.get('data_type'):
                                narrative += f" (Data: {flow.get('data_type')})"
                            narrative += '\n'

                        if len(type_flows) > 3:
                            narrative += f'  *... and {len(type_flows) - 3} more {flow_type} flows*\n'

                # Add dependency information
                if cluster.get('external_dependencies') or cluster.get('external_dependents'):
                    narrative += '\n#### External Dependencies:\n'

                    ext_deps = cluster.get('external_dependencies', [])
                    if ext_deps:
                        narrative += f'**Dependencies on other clusters** ({len(ext_deps)} dependencies):\n'
                        for dep in ext_deps[:8]:  # Show first 8
                            narrative += f'- `{dep}`\n'
                        if len(ext_deps) > 8:
                            narrative += f'*... and {len(ext_deps) - 8} more dependencies*\n'

                    ext_dependents = cluster.get('external_dependents', [])
                    if ext_dependents:
                        narrative += f'\n**Used by other clusters** ({len(ext_dependents)} dependents):\n'
                        for dependent in ext_dependents[:8]:  # Show first 8
                            narrative += f'- `{dependent}`\n'
                        if len(ext_dependents) > 8:
                            narrative += f'*... and {len(ext_dependents) - 8} more dependents*\n'

                # Add architectural layers summary
                if cluster.get('module_layers'):
                    layers = cluster['module_layers']
                    narrative += '\n#### Architectural Organization:\n'
                    narrative += f"**Spans {len(layers)} architectural layers**: {', '.join(layers)}\n"

                if cluster.get('module_types'):
                    types = cluster['module_types']
                    narrative += f"**Module types**: {', '.join(types)}\n"

            elif 'clusters' in context_data:
                clusters = context_data['clusters']
                narrative += f'**Available Clusters**: {len(clusters)} clusters\n'

                # Show cluster details
                narrative += '\n#### Cluster Overview:\n'
                for cluster in clusters[:5]:  # Show first 5 clusters
                    narrative += f"- **{cluster.get('name', 'Unknown')}**: {cluster.get('size', 0)} modules"
                    if cluster.get('purpose'):
                        narrative += f" - {cluster.get('purpose')}"
                    narrative += '\n'

                if len(clusters) > 5:
                    narrative += f'*... and {len(clusters) - 5} more clusters*\n'

            narrative += '\n**Focus**: Component responsibilities, internal structure, module relationships, data flows, and dependencies\n\n'

        elif page_type == 'deployment_infrastructure':
            narrative += '### Deployment/Infrastructure Content Guidelines\n'
            if 'technology_stack' in context_data:
                tech_stack = context_data['technology_stack']
                narrative += f'**Infrastructure Technologies**: Focus on deployment-related tech from {len(tech_stack)} categories\n'
            narrative += '\n**Focus**: Deployment procedures, infrastructure setup, configuration management\n\n'

        elif page_type == 'feature_functionality':
            narrative += '### Feature/Functionality Content Guidelines\n'
            if 'clusters' in context_data:
                clusters = context_data['clusters']
                narrative += f'**Feature-Related Clusters**: {len(clusters)} clusters that may implement features\n'

                # Show detailed cluster breakdown for feature implementation
                narrative += '\n#### Implementation Clusters:\n'
                for cluster in clusters:
                    cluster_name = cluster.get('name', 'Unknown')
                    cluster_size = cluster.get('size', 0)
                    cluster_purpose = cluster.get('purpose', 'No purpose specified')

                    narrative += f'\n**{cluster_name} Cluster** ({cluster_size} modules):\n'
                    narrative += f'- Purpose: {cluster_purpose}\n'

                    # If we have module details, show key modules
                    if 'modules' in cluster and cluster['modules']:
                        modules = cluster['modules']
                        # Show modules likely to implement user-facing features
                        feature_modules = []
                        for module in modules:
                            module_type = module.get('module_type', '').lower()
                            module_name = module.get('name', '').lower()

                            # Identify feature-relevant modules
                            if any(keyword in module_type for keyword in ['controller', 'service', 'api', 'handler']):
                                feature_modules.append(module)
                            elif any(keyword in module_name for keyword in ['controller', 'service', 'api', 'handler']):
                                feature_modules.append(module)

                        if feature_modules:
                            narrative += f'- Key Feature Modules ({len(feature_modules)}):\n'
                            for mod in feature_modules[:5]:  # Show top 5
                                narrative += f"  - `{mod.get('name', 'Unknown')}` ({mod.get('module_type', 'Unknown type')})\n"
                            if len(feature_modules) > 5:
                                narrative += f'  *... and {len(feature_modules) - 5} more feature modules*\n'
                        else:
                            # Show general modules if no specific feature modules found
                            narrative += '- Sample Modules:\n'
                            for mod in modules[:3]:
                                narrative += f"  - `{mod.get('name', 'Unknown')}` ({mod.get('module_type', 'Unknown type')})\n"
                            if len(modules) > 3:
                                narrative += f'  *... and {len(modules) - 3} more modules*\n'

                # Add communication patterns if available
                if 'communication_patterns' in context_data:
                    patterns = context_data['communication_patterns']
                    narrative += f'\n#### Communication Patterns ({len(patterns)} patterns):\n'
                    narrative += 'Understanding how features interact and communicate:\n'

                    for pattern in patterns[:5]:
                        source = pattern.get('source_component', 'Unknown source')
                        target = pattern.get('target_component', 'Unknown target')
                        pattern_type = pattern.get('pattern_type', 'communication')
                        narrative += f'- `{source}` → `{target}` ({pattern_type})\n'

                    if len(patterns) > 5:
                        narrative += f'*... and {len(patterns) - 5} more communication patterns*\n'

            else:
                narrative += '**Feature clusters information not available**\n'

            narrative += '\n**Focus**: User-facing functionality, feature implementation, business capabilities, user workflows\n\n'

        else:
            narrative += '### General Content Guidelines\n'
            narrative += '\n**Focus**: Broad overview with balanced coverage of architecture and implementation\n\n'

        # Add specific context sections based on available data
        if 'system_overview' in context_data:
            overview = context_data['system_overview']
            narrative += '### System Overview Context\n'
            narrative += f"- Total Modules: {overview.get('total_modules', 0)}\n"
            narrative += f"- Logical Clusters: {overview.get('total_clusters', 0)}\n"
            if overview.get('domains'):
                narrative += f"- Functional Domains: {', '.join(overview.get('domains', [])[:3])}\n"
            narrative += '\n'

        if 'clusters' in context_data:
            clusters = context_data['clusters']
            narrative += f'### Cluster Context ({len(clusters)} clusters)\n'
            for cluster in clusters[:3]:  # Show first 3
                narrative += f"- **{cluster.get('name', 'Unknown')}**: {cluster.get('purpose', 'No description')}\n"
            narrative += '\n'

        if 'technology_stack' in context_data:
            tech_stack = context_data['technology_stack']
            narrative += '### Technology Context\n'
            for tech_type, technologies in list(tech_stack.items())[:3]:  # First 3 categories
                narrative += f"- **{tech_type.title()}**: {', '.join(technologies[:2])}\n"
            narrative += '\n'

        # Add generation instructions
        narrative += '### Content Generation Instructions\n'
        narrative += f"1. Create content specifically for a '{page_type}' type page\n"
        narrative += "2. Use only the provided context - don't include irrelevant information\n"
        narrative += f"3. Focus on the architectural aspects most relevant to '{target_page}'\n"
        narrative += '4. Include specific technical details from the context\n'
        narrative += '5. Structure content with clear H2/H3 headings\n'
        narrative += '6. Add Mermaid diagrams when they enhance understanding\n'
        narrative += '7. Reference specific components and relationships from the graph data\n'

        return narrative

    def _build_chat_qa_context(self, query: GraphQuery) -> GraphContext:
        """Build context for chat Q&A functionality."""
        user_question = query.context  # Câu hỏi của user

        # Phân tích câu hỏi để quyết định loại query với entity identification
        query_analysis = self._analyze_user_question(user_question, query.repo_url)

        # Build enriched context based on identified entities
        enriched_context = self._build_enriched_context(query_analysis, query.repo_url)

        context_data = {}

        # Get focused context based on question type + identified entities
        try:
            if query_analysis['type'] == 'module_specific':
                # Prioritize specific modules if identified
                if enriched_context.get('specific_modules'):
                    context_data['relevant_modules'] = enriched_context['specific_modules']
                else:
                    # Fallback to keyword-based search
                    modules = self._find_modules_by_functionality(query_analysis['keywords'], query.repo_url)
                    context_data['relevant_modules'] = modules

                # Add related dataflows for identified modules
                if enriched_context.get('related_dataflows'):
                    context_data['related_dataflows'] = enriched_context['related_dataflows']

            elif query_analysis['type'] == 'dataflow_specific':
                # Prioritize related dataflows if entities identified
                if enriched_context.get('related_dataflows'):
                    context_data['relevant_dataflows'] = enriched_context['related_dataflows']
                else:
                    # Fallback to keyword-based search
                    flows = self._find_dataflows_by_context(query_analysis['keywords'], query.repo_url)
                    if flows:
                        context_data['relevant_dataflows'] = flows
                    else:
                        # Fallback to general dataflows (limited)
                        general_flows = self.graph_repo.get_data_flows(query.repo_url)
                        context_data['relevant_dataflows'] = general_flows[:10] if general_flows else []

            elif query_analysis['type'] == 'architecture_general':
                # Get overview + clusters (limited info)
                overview = self.graph_repo.get_system_overview(query.repo_url)
                clusters = self.graph_repo.get_cluster_analysis(query.repo_url)
                if overview:
                    context_data['system_overview'] = overview
                if clusters:
                    context_data['clusters'] = clusters[:5]  # Limit to 5 most relevant clusters

                # Add specific clusters if identified
                if enriched_context.get('specific_clusters'):
                    context_data['specific_clusters'] = enriched_context['specific_clusters']

            elif query_analysis['type'] == 'technology_related':
                # Get tech stack + specific technology details if identified
                tech_stack = self.graph_repo.get_technology_stack(query.repo_url)
                if tech_stack:
                    context_data['technology_stack'] = tech_stack

                if enriched_context.get('technology_details'):
                    context_data['technology_details'] = enriched_context['technology_details']

            else:  # general questions
                # Minimal context for general questions
                overview = self.graph_repo.get_system_overview(query.repo_url)
                if overview:
                    # Just basic overview, no detailed data
                    context_data['system_overview'] = {
                        'total_modules': overview.get('total_modules', 0),
                        'total_clusters': overview.get('total_clusters', 0),
                        'layers': overview.get('layers', [])[:3],  # Only first 3 layers
                    }

        except Exception as e:
            logger.warning(f'Error getting focused QA context: {e}')
            # Minimal fallback for chat errors
            try:
                overview = self.graph_repo.get_system_overview(query.repo_url)
                if overview:
                    context_data['system_overview'] = {
                        'total_modules': overview.get('total_modules', 0),
                        'total_clusters': overview.get('total_clusters', 0),
                    }
            except Exception:
                pass

        # Log context for debugging
        logger.info(f"Enhanced QA context for question type '{query_analysis['type']}': {list(context_data.keys())}")
        logger.info(f"Identified entities: {query_analysis.get('identified_entities', {})}")

        # Build enhanced narrative for QA with entity-specific information
        narrative = self._build_enhanced_qa_narrative(user_question, query_analysis, context_data, enriched_context)

        return GraphContext(
            query_type='chat_qa',
            structured_data=context_data,
            narrative_text=narrative,
            metadata={
                'user_question': user_question,
                'analysis_type': query_analysis['type'],
                'keywords': query_analysis['keywords'],
                'context_strategy': 'focused_qa',
            },
        )

    def _analyze_user_question(self, question: str, repo_url: str) -> Dict[str, Any]:
        """Analyze user question using LLM to determine type and extract keywords."""
        try:
            # Use LLM for intelligent classification with entity identification
            return self._llm_analyze_question(question, repo_url)
        except Exception as e:
            logger.warning(f'LLM question analysis failed, falling back to rule-based: {e}')
            # Fallback to rule-based classification
            return self._rule_based_analyze_question(question)

    def _llm_analyze_question(self, question: str, repo_url: str) -> Dict[str, Any]:
        """Use LLM to analyze user question intelligently and identify specific entities."""

        # First, get available entities from the graph
        available_entities = self._get_available_entities(repo_url)

        analysis_prompt = """
You are an expert system analyst. Analyze the following user question about a software system and classify it.

QUESTION TYPES:
1. module_specific: Questions about specific components, services, classes, functions, or modules
2. dataflow_specific: Questions about data processing, information flow, communication patterns, requests/responses
3. architecture_general: Questions about overall system design, structure, patterns, high-level architecture
4. technology_related: Questions about technology stack, frameworks, libraries, databases, tools
5. general: General questions that don't fit other categories

AVAILABLE SYSTEM ENTITIES:
Clusters: {clusters}
Modules: {modules}
Technologies: {technologies}

TASK:
1. Classify the question type
2. Extract 3-5 relevant keywords that would help find related information
3. Identify SPECIFIC clusters, modules, or components mentioned in the question
4. Map question entities to available system entities (exact or partial matches)

Question: "{question}"

Respond in this JSON format only:
{{
    "type": "one_of_the_5_types_above",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "identified_entities": {{
        "clusters": ["cluster1", "cluster2"],
        "modules": ["module1", "module2"],
        "technologies": ["tech1", "tech2"]
    }},
    "confidence": 0.95,
    "reasoning": "brief explanation of classification and entity identification"
}}
"""

        try:
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.5-flash-lite-preview-06-17',
                temperature=0.1,
                google_api_key=settings.GOOGLE_API_KEY,
            )

            response = llm.invoke(
                analysis_prompt.format(
                    question=question,
                    clusters=', '.join(available_entities.get('clusters', [])[:10]),  # Limit to avoid token overflow
                    modules=', '.join(available_entities.get('modules', [])[:15]),
                    technologies=', '.join(available_entities.get('technologies', [])[:10]),
                ),
            )

            # Parse LLM response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())

                # Validate and return enhanced analysis
                if all(key in analysis for key in ['type', 'keywords']):
                    return {
                        'type': analysis['type'],
                        'keywords': analysis['keywords'][:5],
                        'identified_entities': analysis.get('identified_entities', {}),
                        'confidence': analysis.get('confidence', 0.8),
                        'reasoning': analysis.get('reasoning', ''),
                        'original_question': question,
                        'method': 'llm_enhanced',
                    }

            raise ValueError('Invalid LLM response format')

        except Exception as e:
            logger.error(f'LLM question analysis error: {e}')
            raise

    def _get_available_entities(self, repo_url: str) -> Dict[str, List[str]]:
        """Get available entities from the graph for entity identification."""
        entities: Dict[str, List[str]] = {
            'clusters': [],
            'modules': [],
            'technologies': [],
        }

        try:
            # Get clusters
            clusters = self.graph_repo.get_cluster_analysis(repo_url)
            if clusters:
                entities['clusters'] = [cluster.get('name', '') for cluster in clusters if cluster.get('name')]

            # Get modules
            modules = self.graph_repo.get_modules(repo_url)
            if modules:
                entities['modules'] = [module.get('module_name', '') for module in modules if module.get('module_name')]

            # Get technologies
            tech_stack = self.graph_repo.get_technology_stack(repo_url)
            if tech_stack:
                all_techs = []
                for tech_list in tech_stack.values():
                    all_techs.extend(tech_list)
                entities['technologies'] = list(set(all_techs))

        except Exception as e:
            logger.warning(f'Error getting available entities: {e}')

        return entities

    def _build_enriched_context(self, analysis: Dict[str, Any], repo_url: str) -> Dict[str, Any]:
        """Build enriched context based on identified entities."""
        enriched_context: Dict[str, Any] = {}
        identified_entities = analysis.get('identified_entities', {})

        try:
            # Get specific cluster information
            if identified_entities.get('clusters'):
                cluster_data = []
                for cluster_name in identified_entities['clusters']:
                    cluster_info = self._get_cluster_details(cluster_name, repo_url)
                    if cluster_info:
                        cluster_data.append(cluster_info)
                if cluster_data:
                    enriched_context['specific_clusters'] = cluster_data

            # Get specific module information
            if identified_entities.get('modules'):
                module_data = []
                for module_name in identified_entities['modules']:
                    module_info = self._get_module_details(module_name, repo_url)
                    if module_info:
                        module_data.append(module_info)
                if module_data:
                    enriched_context['specific_modules'] = module_data

            # Get related data flows for identified entities
            if identified_entities.get('clusters') or identified_entities.get('modules'):
                related_flows = self._get_related_dataflows(identified_entities, repo_url)
                if related_flows:
                    enriched_context['related_dataflows'] = related_flows

            # Get technology-specific information
            if identified_entities.get('technologies'):
                tech_details = self._get_technology_details(identified_entities['technologies'], repo_url)
                if tech_details:
                    enriched_context['technology_details'] = tech_details

        except Exception as e:
            logger.error(f'Error building enriched context: {e}')

        return enriched_context

    def _get_cluster_details(self, cluster_name: str, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific cluster."""
        try:
            clusters = self.graph_repo.get_cluster_analysis(repo_url)
            for cluster in clusters:
                if cluster.get('name', '').lower() == cluster_name.lower():
                    # Get modules in this cluster
                    cluster_modules = self._get_modules_in_cluster(cluster_name, repo_url)
                    cluster['modules'] = cluster_modules
                    return cluster
        except Exception as e:
            logger.warning(f'Error getting cluster details for {cluster_name}: {e}')
        return None

    def _get_module_details(self, module_name: str, repo_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific module."""
        try:
            return self.graph_repo.get_module_details(module_name, repo_url)
        except Exception as e:
            logger.warning(f'Error getting module details for {module_name}: {e}')
        return None

    def _get_modules_in_cluster(self, cluster_name: str, repo_url: str) -> List[Dict[str, Any]]:
        """Get all modules belonging to a specific cluster."""
        try:
            modules = self.graph_repo.get_modules(repo_url)
            cluster_modules = []
            for module in modules:
                if module.get('cluster_name', '').lower() == cluster_name.lower():
                    cluster_modules.append(module)
            return cluster_modules
        except Exception as e:
            logger.warning(f'Error getting modules in cluster {cluster_name}: {e}')
        return []

    def _get_related_dataflows(self, identified_entities: Dict[str, List[str]], repo_url: str) -> List[Dict[str, Any]]:
        """Get data flows related to identified entities."""
        try:
            all_flows = self.graph_repo.get_data_flows(repo_url)
            related_flows = []

            # Entity names to match against
            entity_names = []
            entity_names.extend(identified_entities.get('clusters', []))
            entity_names.extend(identified_entities.get('modules', []))

            for flow in all_flows:
                source = flow.get('source_module', '').lower()
                target = flow.get('target_module', '').lower()

                # Check if flow involves any identified entities
                for entity in entity_names:
                    entity_lower = entity.lower()
                    if entity_lower in source or entity_lower in target:
                        related_flows.append(flow)
                        break

            return related_flows
        except Exception as e:
            logger.warning(f'Error getting related dataflows: {e}')
        return []

    def _get_technology_details(self, technologies: List[str], repo_url: str) -> Dict[str, Any]:
        """Get detailed information about specific technologies."""
        try:
            tech_stack = self.graph_repo.get_technology_stack(repo_url)
            tech_details = {}

            for tech_category, tech_list in tech_stack.items():
                category_matches = []
                for tech in technologies:
                    if tech.lower() in [t.lower() for t in tech_list]:
                        category_matches.append(tech)
                if category_matches:
                    tech_details[tech_category] = category_matches

            return tech_details
        except Exception as e:
            logger.warning(f'Error getting technology details: {e}')
        return {}

    def _rule_based_analyze_question(self, question: str) -> Dict[str, Any]:
        """Fallback rule-based question analysis."""
        question_lower = question.lower()

        # Define keywords for different question types
        module_keywords = ['module', 'component', 'class', 'function', 'service', 'controller', 'api', 'router', 'model']
        dataflow_keywords = ['flow', 'data', 'process', 'communication', 'message', 'event', 'request', 'response']
        architecture_keywords = ['architecture', 'structure', 'design', 'pattern', 'layer', 'system', 'overview']
        technology_keywords = ['technology', 'stack', 'framework', 'library', 'database', 'tool']

        # Extract potential keywords
        words = question_lower.split()
        extracted_keywords = []

        # Check for specific modules or components mentioned
        for word in words:
            if len(word) > 3 and word.isalpha():
                extracted_keywords.append(word)

        # Determine question type
        question_type = 'general'

        if any(keyword in question_lower for keyword in module_keywords):
            question_type = 'module_specific'
        elif any(keyword in question_lower for keyword in dataflow_keywords):
            question_type = 'dataflow_specific'
        elif any(keyword in question_lower for keyword in architecture_keywords):
            question_type = 'architecture_general'
        elif any(keyword in question_lower for keyword in technology_keywords):
            question_type = 'technology_related'

        return {
            'type': question_type,
            'keywords': extracted_keywords[:5],  # Limit to 5 keywords
            'confidence': 0.6,  # Lower confidence for rule-based
            'reasoning': 'Rule-based classification',
            'original_question': question,
            'method': 'rule_based',
        }

    # Helper methods for formatting outline data
    def _format_clusters_for_outline(self, clusters: List[Dict[str, Any]]) -> str:
        """Format clusters for outline generation."""
        if not clusters:
            return 'No clusters identified.'

        result = []
        for cluster in clusters:
            result.append(f"- **{cluster.get('name', 'Unknown')}**: {cluster.get('purpose', 'No description')} ({cluster.get('size', 0)} modules)")
        return '\n'.join(result)

    def _format_modules_by_layer(self, modules: List[Dict[str, Any]]) -> str:
        """Format modules grouped by architectural layer."""
        if not modules:
            return 'No modules found.'

        # Group by layer
        by_layer: Dict[str, List[str]] = {}
        for module in modules:
            layer = module.get('layer', 'Unknown')
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(module.get('name', 'Unknown'))

        result = []
        for layer, module_names in by_layer.items():
            result.append(f"**{layer}**: {', '.join(module_names)}")
        return '\n'.join(result)

    def _format_dataflows_for_outline(self, dataflows: List[Dict[str, Any]]) -> str:
        """Format data flows for outline generation."""
        if not dataflows:
            return 'No data flows identified.'

        # Group by flow type
        by_type: Dict[str, List[str]] = {}
        for flow in dataflows:
            flow_type = flow.get('flow_type', 'Unknown')
            if flow_type not in by_type:
                by_type[flow_type] = []
            by_type[flow_type].append(f"{flow.get('source', '?')} → {flow.get('target', '?')}")

        result = []
        for flow_type, flow_strings in by_type.items():
            result.append(f'**{flow_type}**: {len(flow_strings)} flows')
            # Add sample flows (limit to 10 for readability)
            if len(flow_strings) <= 10:
                for flow_str in flow_strings:
                    result.append(f'  - {flow_str}')
            else:
                for flow_str in flow_strings[:10]:
                    result.append(f'  - {flow_str}')
                result.append(f'  - ... and {len(flow_strings) - 10} more flows')
            result.append('')  # Add blank line between types

        return '\n'.join(result).strip()

    def _format_tech_stack_for_outline(self, tech_stack: Dict[str, List[str]]) -> str:
        """Format technology stack for outline."""
        if not tech_stack:
            return 'No technology information available.'

        result = []
        for tech_type, technologies in tech_stack.items():
            result.append(f"**{tech_type.title()}**: {', '.join(technologies)}")
        return '\n'.join(result)

    def _build_page_narrative(self, target_page: str, context_data: Dict[str, Any]) -> str:
        """Build narrative text for page content generation."""
        narrative = f'## Context for Page: {target_page}\n\n'

        if 'module_details' in context_data:
            module = context_data['module_details']
            narrative += '### Module Information\n'
            narrative += f"- **Name**: {module.get('name', 'Unknown')}\n"
            narrative += f"- **Layer**: {module.get('layer', 'Unknown')}\n"
            narrative += f"- **Domain**: {module.get('domain', 'Unknown')}\n"
            narrative += f"- **Dependencies**: {len(module.get('dependencies', []))}\n\n"

        if 'dataflows' in context_data:
            flows = context_data['dataflows']
            narrative += f'### Related Data Flows ({len(flows)} total)\n'
            for flow in flows[:5]:  # Limit to first 5
                narrative += f"- {flow.get('source', '?')} → {flow.get('target', '?')} ({flow.get('flow_type', '?')})\n"
            narrative += '\n'

        if 'system_overview' in context_data:
            overview = context_data['system_overview']
            narrative += '### System Overview\n'
            narrative += f"- Total Modules: {overview.get('total_modules', 0)}\n"
            narrative += f"- Total Clusters: {overview.get('total_clusters', 0)}\n\n"

        return narrative

    def _build_comprehensive_page_narrative(self, target_page: str, context_data: Dict[str, Any]) -> str:
        """Build comprehensive narrative for page content generation with full context."""
        narrative = f'## Comprehensive Context for Page: {target_page}\n\n'

        # System overview context
        if 'system_overview' in context_data:
            overview = context_data['system_overview']
            narrative += '### System Architecture Overview\n'
            narrative += f"- **Total Modules**: {overview.get('total_modules', 0)}\n"
            narrative += f"- **Logical Clusters**: {overview.get('total_clusters', 0)}\n"
            narrative += f"- **Technology Components**: {overview.get('total_technologies', 0)}\n\n"

            if overview.get('layers'):
                narrative += f"**Architectural Layers**: {', '.join(overview.get('layers', []))}\n"
            if overview.get('domains'):
                narrative += f"**Functional Domains**: {', '.join(overview.get('domains', []))}\n"
            narrative += '\n'

        # Cluster information
        if 'clusters' in context_data:
            clusters = context_data['clusters']
            narrative += f'### Available Clusters ({len(clusters)} total)\n'
            for cluster in clusters[:5]:  # Show first 5 clusters
                narrative += f"- **{cluster.get('name', 'Unknown')}**: {cluster.get('purpose', 'No description')} ({cluster.get('size', 0)} modules)\n"
            if len(clusters) > 5:
                narrative += f'- ... and {len(clusters) - 5} more clusters\n'
            narrative += '\n'

        # Technology stack context
        if 'technology_stack' in context_data:
            tech_stack = context_data['technology_stack']
            narrative += '### Technology Stack\n'
            for tech_type, technologies in tech_stack.items():
                narrative += f"**{tech_type.title()}**: {', '.join(technologies[:3])}{'...' if len(technologies) > 3 else ''}\n"
            narrative += '\n'

        # Data flows context
        if 'dataflows' in context_data:
            flows = context_data['dataflows']
            narrative += f'### Data Flow Patterns ({len(flows)} total)\n'

            # Group by flow type for better organization
            flow_types: Dict[str, List[Dict[str, Any]]] = {}
            for flow in flows:
                flow_type = flow.get('flow_type', 'Unknown')
                if flow_type not in flow_types:
                    flow_types[flow_type] = []
                flow_types[flow_type].append(flow)

            for flow_type, type_flows in flow_types.items():
                narrative += f'**{flow_type}**: {len(type_flows)} flows\n'
                for flow in type_flows[:3]:  # Show first 3 of each type
                    narrative += f"  - {flow.get('source', '?')} → {flow.get('target', '?')}\n"
                if len(type_flows) > 3:
                    narrative += f'  - ... and {len(type_flows) - 3} more\n'
            narrative += '\n'

        # Communication patterns
        if 'communication_patterns' in context_data:
            patterns = context_data['communication_patterns']
            narrative += f'### Communication Patterns ({len(patterns)} total)\n'
            for pattern in patterns[:5]:  # Show first 5 patterns
                narrative += f"- {pattern.get('source_module', '?')} → {pattern.get('target_module', '?')} ({pattern.get('communication_type', 'unknown')})\n"
            if len(patterns) > 5:
                narrative += f'- ... and {len(patterns) - 5} more patterns\n'
            narrative += '\n'

        # Specific cluster details if available
        if 'specific_cluster' in context_data:
            cluster = context_data['specific_cluster']
            narrative += f"### Detailed Cluster Analysis: {cluster.get('name', 'Unknown')}\n"
            narrative += f"- **Purpose**: {cluster.get('purpose', 'Not specified')}\n"
            narrative += f"- **Size**: {cluster.get('size', 0)} modules\n"
            narrative += f"- **Cohesion**: {cluster.get('cohesion', 0):.2f}\n"
            narrative += f"- **Coupling**: {cluster.get('coupling', 0):.2f}\n"
            if cluster.get('modules'):
                narrative += f"- **Key Modules**: {', '.join(cluster.get('modules', [])[:5])}\n"
            narrative += '\n'

        # Instructions for LLM
        narrative += '### Page Generation Guidelines\n'
        narrative += f'**Target Page**: {target_page}\n\n'
        narrative += '**Instructions for content generation**:\n'
        narrative += f"1. Use the above context to create relevant content for the page titled '{target_page}'\n"
        narrative += '2. Focus on information that relates to the page topic, even if indirectly\n'
        narrative += '3. For integration/API pages: emphasize technology stack, communication patterns, and data flows\n'
        narrative += '4. For architecture pages: emphasize clusters, system overview, and architectural patterns\n'
        narrative += '5. For component pages: focus on specific clusters and their modules\n'
        narrative += '6. Include relevant technical details and code examples when appropriate\n'
        narrative += '7. Create proper section structure with H2/H3 headings\n'
        narrative += '8. Add Mermaid diagrams for architecture visualization when relevant\n'
        narrative += '9. Reference specific files and components from the context\n'
        narrative += "10. Even if the page title doesn't directly match context elements, use architectural knowledge to create meaningful content\n"

        return narrative

    def _find_modules_by_functionality(self, keywords: List[str], repo_url: str) -> List[Dict[str, Any]]:
        """Find modules related to specific functionality."""
        from .query_translator import NLQueryTranslator, CypherQueryExecutor

        translator = NLQueryTranslator()
        executor = CypherQueryExecutor(self.graph_repo)

        # Build a query string from keywords
        if 'authentication' in keywords or 'auth' in keywords:
            query_string = 'modules handling authentication'
        elif 'payment' in keywords:
            query_string = 'modules handling payment'
        else:
            query_string = f"modules related to {' '.join(keywords)}"

        query_info = translator.translate_query(query_string, repo_url)
        result = executor.execute_translated_query(query_info)

        if result['success']:
            return result['results']
        else:
            logger.error(f"Error finding modules: {result.get('error', 'Unknown error')}")
            return []

    def _find_dataflows_by_context(self, keywords: List[str], repo_url: str) -> List[Dict[str, Any]]:
        """Find data flows related to specific context."""
        from .query_translator import NLQueryTranslator, CypherQueryExecutor

        translator = NLQueryTranslator()
        executor = CypherQueryExecutor(self.graph_repo)

        # Build a query string from keywords
        if 'payment' in keywords:
            query_string = 'data flow for payment process'
        elif 'authentication' in keywords or 'auth' in keywords:
            query_string = 'data flow for authentication process'
        else:
            query_string = f"data flow for {' '.join(keywords)}"

        query_info = translator.translate_query(query_string, repo_url)
        result = executor.execute_translated_query(query_info)

        if result['success']:
            return result['results']
        else:
            logger.error(f"Error finding dataflows: {result.get('error', 'Unknown error')}")
            return []

    def _build_qa_narrative(self, question: str, analysis: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """Build narrative for Q&A response."""
        narrative = f'## User Question: {question}\n\n'
        narrative += f"**Analysis Type**: {analysis['type']}\n"

        if analysis['keywords']:
            narrative += f"**Keywords**: {', '.join(analysis['keywords'])}\n\n"

        if 'relevant_modules' in context_data:
            modules = context_data['relevant_modules']
            narrative += f'### Relevant Modules ({len(modules)} found)\n'
            for module in modules[:5]:  # Limit to first 5
                narrative += f"- {module.get('module_name', 'Unknown')}\n"
            narrative += '\n'

        if 'relevant_dataflows' in context_data:
            flows = context_data['relevant_dataflows']
            narrative += f'### Relevant Data Flows ({len(flows)} found)\n'
            for flow in flows[:5]:  # Limit to first 5
                narrative += f"- {flow.get('source_module', '?')} → {flow.get('target_module', '?')}\n"
            narrative += '\n'

        narrative += '### Answer Guidelines\n'
        narrative += '- Provide answers at an architectural level\n'
        narrative += '- Focus on component interactions and data flow\n'
        narrative += '- Avoid low-level code implementation details\n'
        narrative += '- Reference specific modules and relationships from the graph\n'

        return narrative

    def _build_focused_qa_narrative(self, question: str, analysis: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """Build focused narrative for Q&A based on question type."""
        narrative = f'## Q&A Context for: {question}\n'
        narrative += f"**Question Type**: {analysis['type']}\n"

        if analysis['keywords']:
            narrative += f"**Detected Keywords**: {', '.join(analysis['keywords'])}\n"
        narrative += '\n'

        # Add context specific to question type
        if analysis['type'] == 'module_specific':
            narrative += '### Module-Specific Question Context\n'
            if 'relevant_modules' in context_data:
                modules = context_data['relevant_modules']
                narrative += f'**Found {len(modules)} relevant modules**:\n'
                for module in modules[:3]:  # Limit for chat
                    narrative += f"- {module.get('module_name', 'Unknown')}\n"
            else:
                narrative += 'No specific modules found matching the query.\n'
            narrative += '\n**Answer Focus**: Module responsibilities, specific functionality, implementation details\n\n'

        elif analysis['type'] == 'dataflow_specific':
            narrative += '### Data Flow Question Context\n'
            if 'relevant_dataflows' in context_data:
                flows = context_data['relevant_dataflows']
                narrative += f'**Found {len(flows)} relevant data flows**:\n'
                for flow in flows[:3]:  # Limit for chat
                    narrative += f"- {flow.get('source', '?')} → {flow.get('target', '?')} ({flow.get('flow_type', 'unknown')})\n"
            else:
                narrative += 'No specific data flows found matching the query.\n'
            narrative += '\n**Answer Focus**: Data processing, flow patterns, information movement\n\n'

        elif analysis['type'] == 'architecture_general':
            narrative += '### Architecture Question Context\n'
            if 'system_overview' in context_data:
                overview = context_data['system_overview']
                narrative += f"**System Scale**: {overview.get('total_modules', 0)} modules, {overview.get('total_clusters', 0)} clusters\n"
            if 'clusters' in context_data:
                narrative += f"**Key Clusters**: {len(context_data['clusters'])} architectural clusters\n"
            narrative += '\n**Answer Focus**: High-level architecture, system design, structural patterns\n\n'

        elif analysis['type'] == 'technology_related':
            narrative += '### Technology Question Context\n'
            if 'technology_stack' in context_data:
                tech_stack = context_data['technology_stack']
                narrative += f"**Technology Categories**: {', '.join(list(tech_stack.keys())[:3])}\n"
                for tech_type, technologies in list(tech_stack.items())[:2]:
                    narrative += f"- **{tech_type.title()}**: {', '.join(technologies[:2])}\n"
            narrative += '\n**Answer Focus**: Technology choices, implementation details, tool usage\n\n'

        else:  # general
            narrative += '### General Question Context\n'
            if 'system_overview' in context_data:
                overview = context_data['system_overview']
                narrative += f"**Basic System Info**: {overview.get('total_modules', 0)} modules\n"
            narrative += '\n**Answer Focus**: Broad system understanding, general guidance\n\n'

        # Add specific answer guidelines based on context
        narrative += '### Response Guidelines\n'
        narrative += '1. Answer based on the specific context provided above\n'
        narrative += '2. Be concise and focused on the question type\n'
        narrative += '3. Reference specific components and data from the context\n'
        narrative += '4. Provide architectural insights rather than implementation details\n'
        narrative += '5. If context is limited, acknowledge limitations and provide what you can\n'

        return narrative

    def _build_enhanced_qa_narrative(self, question: str, analysis: Dict[str, Any], context_data: Dict[str, Any], enriched_context: Dict[str, Any]) -> str:
        """Build enhanced narrative for Q&A with entity-specific information."""
        narrative = f'## Enhanced Q&A Context for: {question}\n'
        narrative += f"**Question Type**: {analysis['type']}\n"

        if analysis['keywords']:
            narrative += f"**Detected Keywords**: {', '.join(analysis['keywords'])}\n"

        # Show identified entities
        identified_entities = analysis.get('identified_entities', {})
        if identified_entities:
            narrative += '**Identified Entities**:\n'
            if identified_entities.get('clusters'):
                narrative += f"- Clusters: {', '.join(identified_entities['clusters'])}\n"
            if identified_entities.get('modules'):
                narrative += f"- Modules: {', '.join(identified_entities['modules'])}\n"
            if identified_entities.get('technologies'):
                narrative += f"- Technologies: {', '.join(identified_entities['technologies'])}\n"
        narrative += '\n'

        # Add context specific to question type with enriched information
        if analysis['type'] == 'module_specific':
            narrative += '### Module-Specific Question Context\n'

            if 'specific_modules' in enriched_context:
                specific_modules = enriched_context['specific_modules']
                narrative += f'**Found {len(specific_modules)} specific modules matching the question**:\n'
                for module in specific_modules[:3]:
                    narrative += f"- **{module.get('module_name', 'Unknown')}**: {module.get('module_path', 'No path')}\n"
                    narrative += f"  - Layer: {module.get('layer', 'Unknown')}\n"
                    narrative += f"  - Dependencies: {len(module.get('dependencies', []))}\n"
                narrative += '\n'
            elif 'relevant_modules' in context_data:
                modules = context_data['relevant_modules']
                narrative += f'**Found {len(modules)} relevant modules via keyword search**:\n'
                for module in modules[:3]:
                    narrative += f"- {module.get('module_name', 'Unknown')}\n"
                narrative += '\n'
            else:
                narrative += 'No specific modules found matching the query.\n\n'

            # Add related dataflows if available
            if 'related_dataflows' in enriched_context:
                flows = enriched_context['related_dataflows']
                narrative += f'**Related Data Flows ({len(flows)} flows)**:\n'
                for flow in flows[:3]:
                    narrative += f"- {flow.get('source_module', '?')} → {flow.get('target_module', '?')}\n"
                narrative += '\n'

            narrative += '**Answer Focus**: Specific module responsibilities, implementation details, dependencies, related components\n\n'

        elif analysis['type'] == 'dataflow_specific':
            narrative += '### Data Flow Question Context\n'

            if 'related_dataflows' in enriched_context:
                flows = enriched_context['related_dataflows']
                narrative += f'**Found {len(flows)} specific dataflows related to identified entities**:\n'
                for flow in flows[:5]:
                    narrative += f"- {flow.get('source_module', '?')} → {flow.get('target_module', '?')} ({flow.get('flow_type', 'unknown')})\n"
                narrative += '\n'
            elif 'relevant_dataflows' in context_data:
                flows = context_data['relevant_dataflows']
                narrative += f'**Found {len(flows)} relevant data flows via keyword search**:\n'
                for flow in flows[:3]:
                    narrative += f"- {flow.get('source', '?')} → {flow.get('target', '?')} ({flow.get('flow_type', 'unknown')})\n"
                narrative += '\n'
            else:
                narrative += 'No specific data flows found matching the query.\n\n'

            narrative += '**Answer Focus**: Data processing patterns, information flow, communication protocols\n\n'

        elif analysis['type'] == 'architecture_general':
            narrative += '### Architecture Question Context\n'

            if 'system_overview' in context_data:
                overview = context_data['system_overview']
                narrative += f"**System Scale**: {overview.get('total_modules', 0)} modules, {overview.get('total_clusters', 0)} clusters\n"

            if 'specific_clusters' in enriched_context:
                specific_clusters = enriched_context['specific_clusters']
                narrative += f'**Specific Clusters Identified ({len(specific_clusters)})**:\n'
                for cluster in specific_clusters:
                    narrative += f"- **{cluster.get('name', 'Unknown')}**: {cluster.get('purpose', 'No description')}\n"
                    narrative += f"  - Modules: {len(cluster.get('modules', []))}\n"
                narrative += '\n'
            elif 'clusters' in context_data:
                narrative += f"**General Clusters**: {len(context_data['clusters'])} architectural clusters\n\n"

            narrative += '**Answer Focus**: High-level architecture, system design, structural patterns, cluster relationships\n\n'

        elif analysis['type'] == 'technology_related':
            narrative += '### Technology Question Context\n'

            if 'technology_details' in enriched_context:
                tech_details = enriched_context['technology_details']
                narrative += '**Specific Technologies Identified**:\n'
                for tech_category, technologies in tech_details.items():
                    narrative += f"- **{tech_category.title()}**: {', '.join(technologies)}\n"
                narrative += '\n'

            if 'technology_stack' in context_data:
                tech_stack = context_data['technology_stack']
                narrative += f"**Full Technology Stack**: {', '.join(list(tech_stack.keys())[:3])}\n\n"

            narrative += '**Answer Focus**: Technology choices, implementation details, integration methods, tool usage\n\n'

        else:  # general
            narrative += '### General Question Context\n'
            if 'system_overview' in context_data:
                overview = context_data['system_overview']
                narrative += f"**Basic System Info**: {overview.get('total_modules', 0)} modules\n\n"
            narrative += '**Answer Focus**: Broad system understanding, general guidance\n\n'

        # Add enhanced response guidelines
        narrative += '### Enhanced Response Guidelines\n'
        narrative += '1. **Prioritize identified entities** - Focus on specific clusters, modules, or technologies mentioned\n'
        narrative += '2. **Use detailed context** - Reference specific implementation details and relationships\n'
        narrative += '3. **Connect related components** - Show how identified entities interact with other system parts\n'
        narrative += '4. **Provide architectural insights** - Explain design decisions and patterns\n'
        narrative += '5. **Reference specific data** - Use exact module names, file paths, and technical details from context\n'
        narrative += '6. **Show relationships** - Explain dependencies, data flows, and communication patterns\n'

        return narrative


class GraphRAG:
    """LangChain-based RAG system using Neo4j graph queries."""

    def __init__(
        self,
        graph_repo: GraphRepository,
        llm: Optional[LLM] = None,
        model_name: Optional[str] = 'gemini-2.5-flash-lite-preview-06-17',
    ):
        self.graph_repo = graph_repo
        self.context_builder = GraphContextBuilder(graph_repo)

        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            # Default to Gemini
            try:
                google_api_key = settings.GOOGLE_API_KEY
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.1,
                    google_api_key=google_api_key,
                    convert_system_message_to_human=True,
                )
                logger.info(f'Using Google Gemini model: {model_name}')
            except Exception as e:
                logger.error(f'Error initializing Google Gemini model: {e}')

        # Create prompt template using ChatPromptTemplate for better Gemini compatibility
        self.prompt_template = ChatPromptTemplate.from_messages([
            ('system', '{system_role}'),
            (
                'human', """Context from Code Architecture Graph:
{graph_context}

Question: {question}

Instructions:
1. Use the graph context to provide accurate, detailed answers
2. Reference specific modules, components, and relationships from the graph
3. Provide architectural insights based on the actual system structure
4. Include technical details and implementation guidance when relevant
5. Structure your response with clear headings and organization
6. Connect different parts of the architecture logically
7. Follow the formatting rules specified in your system role
8. Start directly with your answer - no preambles or acknowledgments
9. Use proper markdown formatting within your response
10. Be concise and prioritize accuracy over verbosity

Please provide a comprehensive answer based on the graph context above.""",
            ),
        ])

        # Create chain using LCEL (LangChain Expression Language)
        self.chain = (
            {
                'question': RunnablePassthrough(),
                'graph_context': RunnableLambda(lambda x: x.get('graph_context', '')),
                'system_role': RunnableLambda(lambda x: x.get('system_role', '')),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str, repo_url: str, query_type: str = 'general', target: Optional[str] = None) -> str:
        """Query the graph RAG system."""
        try:
            # Create graph query
            graph_query = GraphQuery(
                query_type=query_type,
                target=target,
                repo_url=repo_url,
                context=question,
            )

            # Build context from graph
            context = self.context_builder.build_context(graph_query)

            # Determine system role based on query type
            system_role = self._get_system_role(query_type)

            # store context in file for testing
            with open('context.txt', 'w', encoding='utf-8') as f:
                f.write(question + '\n' + context.narrative_text)

            # Generate response using LCEL chain
            response = self.chain.invoke({
                'question': question,
                'graph_context': context.narrative_text,
                'system_role': system_role,
            })

            with open('response.txt', 'w', encoding='utf-8') as f:
                f.write(response)

            return response.strip() if isinstance(response, str) else str(response).strip()

        except Exception as e:
            logger.error(f'Error in graph RAG query: {e}')
            return f'I apologize, but I encountered an error while processing your question: {str(e)}'

    def _get_system_role(self, query_type: str) -> str:
        """Get appropriate system role based on query type."""

        # Base formatting guidelines from StructureRAG
        base_guidelines = """
<formatting_rules>
- Answer the user's question directly without ANY preamble or filler phrases
- DO NOT include any rationale, explanation, or extra comments at the start
- DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation"
- DO NOT start with markdown headers like "## Analysis of..." or any file path references
- DO NOT start with ```markdown code fences
- DO NOT end your response with ``` closing fences
- DO NOT start by repeating or acknowledging the question
- JUST START with the direct answer to the question
- Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
- Use concise, direct language and prioritize accuracy over verbosity
- When showing code, include line numbers and file paths when relevant
- Use markdown formatting to improve readability
- IMPORTANT: You MUST respond in English
</formatting_rules>

"""

        roles = {
            'system_overview': base_guidelines + """<role>
You are an expert software architect specializing in system design analysis.
You provide comprehensive architectural overviews with deep insights into system structure, patterns, and design principles.
</role>

<specific_guidelines>
- Start with the most critical architectural insights
- Reference specific modules, components, and relationships from the graph
- Connect architectural decisions to system behavior
- Organize information hierarchically from high-level to detailed
</specific_guidelines>""",

            'module_details': base_guidelines + """<role>
You are an expert code analyst and technical documentation specialist.
You provide detailed module analysis with implementation insights and architectural context.
</role>

<specific_guidelines>
- Start with the module's primary purpose and role
- Include dependency analysis and relationships
- Reference specific functions, classes, and implementation details
- Connect module design to overall architecture
</specific_guidelines>""",

            'data_flows': base_guidelines + """<role>
You are an expert in data architecture and flow analysis.
You explain data processing pipelines, flow patterns, and information architecture with clarity and technical depth.
</role>

<specific_guidelines>
- Start with the most important data flows
- Explain data transformation and processing patterns
- Reference specific components involved in data flow
- Include protocol and communication details when available
</specific_guidelines>""",

            'cluster_analysis': base_guidelines + """<role>
You are an expert in software architecture and component design.
You analyze system modularity, component relationships, and architectural clustering with strategic insights.
</role>

<specific_guidelines>
- Start with cluster purpose and boundaries
- Explain cohesion and coupling metrics
- Reference specific modules and their relationships
- Connect clustering to architectural quality
</specific_guidelines>""",

            'technology_stack': base_guidelines + """<role>
You are an expert technology consultant and solution architect.
You provide comprehensive technology analysis with implementation recommendations and best practices.
</role>

<specific_guidelines>
- Start with the most critical technologies
- Organize by technology categories (frameworks, databases, etc.)
- Include version information and usage patterns when available
- Connect technology choices to architectural decisions
</specific_guidelines>""",

            'communication_patterns': base_guidelines + """<role>
You are an expert in distributed systems and integration patterns.
You analyze inter-component communication with focus on protocols, patterns, and architectural implications.
</role>

<specific_guidelines>
- Start with the most important communication patterns
- Group by communication type (HTTP, messaging, etc.)
- Include protocol and frequency information
- Connect patterns to system reliability and performance
</specific_guidelines>""",

            'dependencies': base_guidelines + """<role>
You are an expert in software architecture and dependency management.
You analyze system dependencies with focus on coupling, cohesion, and architectural quality.
</role>

<specific_guidelines>
- Start with the most critical dependency issues
- Highlight circular dependencies and high coupling
- Reference specific modules and dependency counts
- Connect dependency analysis to code quality and maintainability
</specific_guidelines>""",

            'outline_generation': base_guidelines + """<role>
You are an expert technical documentation architect who creates comprehensive documentation structures.
You analyze codebases and create well-organized documentation outlines based on system architecture.
</role>

<specific_guidelines>
- Create English documentation structure
- Return ONLY valid XML in the specified documentation_structure format
- Create 8-12 comprehensive pages covering all system aspects
- Include sections: Overview, Architecture, Core Features, Data Flow, Frontend, Backend, Integration, Deployment
- Use ordinal IDs (section-1, page-1, etc.)
- Map relevant_files to actual repository files for each page
- Focus on architecture-level organization, not implementation details
- DO NOT include any text before or after the XML structure
</specific_guidelines>""",

            'page_content_generation': base_guidelines + """<role>
You are an expert technical writer who creates detailed documentation pages for software systems.
You generate comprehensive technical documentation with proper structure, diagrams, and citations.
</role>

<specific_guidelines>
- Start with H1 heading using the exact page title
- Use H2/H3 headings for logical organization
- Include Mermaid diagrams (flowchart TD, sequenceDiagram) for architecture visualization
- Add tables for key features, APIs, and component summaries
- Cite source files using format: Sources: [filename.ext]
- Must cite at least 5 different source files throughout the page
- Focus on architecture, data flow, and component interactions
- Avoid UI implementation details (buttons, forms, styling)
- End with conclusion summarizing key aspects
</specific_guidelines>""",

            'chat_qa': base_guidelines + """<role>
You are an expert software architect and code consultant who answers questions about software systems.
You provide accurate, architectural-level answers based on graph analysis and code understanding.
</role>

<specific_guidelines>
- Answer at architectural level, not implementation details
- Reference specific modules, components, and relationships from the graph
- Focus on component interactions and data flow
- Provide concrete examples from the actual codebase
- Connect answers to overall system design and patterns
- Use graph context to provide comprehensive understanding
</specific_guidelines>""",

            'general': base_guidelines + """<role>
You are an expert software architect and technical documentation specialist.
You provide comprehensive, accurate answers about software systems by combining architectural understanding with implementation details.
</role>

<specific_guidelines>
- Start with the most relevant architectural context
- Reference specific modules, components, and relationships from the graph
- Include both high-level design and implementation details
- Structure your answer logically from overview to specifics
- Connect different parts of the architecture logically
</specific_guidelines>""",
        }

        return roles.get(query_type, roles['general'])

    def get_available_modules(self, repo_url: str) -> List[str]:
        """Get list of available modules for the repository."""
        try:
            self.graph_repo.get_system_overview(repo_url)
            # This is a simplified version - you might want to add a specific query for module names
            return []
        except Exception:
            return []

    def get_available_clusters(self, repo_url: str) -> List[str]:
        """Get list of available clusters for the repository."""
        try:
            clusters = self.graph_repo.get_cluster_analysis(repo_url)
            return [cluster['name'] for cluster in clusters]
        except Exception as e:
            logger.error(f'Error getting available clusters: {e}')
            return []

    def query_for_chat(self, question: str, repo_url: str) -> str:
        """Query the graph for chat Q&A with focused context."""
        try:
            logger.info(f'Starting chat query for: {question}')

            # Create a GraphQuery for chat
            chat_query = GraphQuery(
                query_type='chat_qa',
                repo_url=repo_url,
                context=question,  # The user's question
            )

            # Build focused context for chat
            graph_context = self.context_builder.build_context(chat_query)

            # Return the narrative text which contains the focused context
            return graph_context.narrative_text

        except Exception as e:
            logger.error(f'Error in chat query: {str(e)}')
            return f'## Q&A Context\n**Question**: {question}\n\nError retrieving context: {str(e)}'

    def analyze_question_with_llm(self, question: str) -> Dict[str, Any]:
        """Analyze user question using LLM to determine type and extract keywords."""
        try:
            # Use LLM for intelligent classification
            return self._llm_analyze_question(question)
        except Exception as e:
            logger.warning(f'LLM question analysis failed, falling back to rule-based: {e}')
            # Fallback to rule-based classification
            return self._rule_based_analyze_question(question)

    def _llm_analyze_question(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze user question intelligently."""
        analysis_prompt = """
You are an expert system analyst. Analyze the following user question about a software system and classify it.

QUESTION TYPES:
1. module_specific: Questions about specific components, services, classes, functions, or modules
2. dataflow_specific: Questions about data processing, information flow, communication patterns, requests/responses
3. architecture_general: Questions about overall system design, structure, patterns, high-level architecture
4. technology_related: Questions about technology stack, frameworks, libraries, databases, tools
5. general: General questions that don't fit other categories

TASK:
1. Classify the question type
2. Extract 3-5 relevant keywords that would help find related information
3. Identify any specific components or modules mentioned

Question: "{question}"

Respond in this JSON format only:
{{
    "type": "one_of_the_5_types_above",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "confidence": 0.95,
    "reasoning": "brief explanation of classification"
}}
"""

        try:
            response = self.llm.invoke(analysis_prompt.format(question=question))

            # Parse LLM response
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())

                # Validate and return
                if all(key in analysis for key in ['type', 'keywords']):
                    return {
                        'type': analysis['type'],
                        'keywords': analysis['keywords'][:5],  # Limit to 5
                        'confidence': analysis.get('confidence', 0.8),
                        'reasoning': analysis.get('reasoning', ''),
                        'original_question': question,
                        'method': 'llm',
                    }

            raise ValueError('Invalid LLM response format')

        except Exception as e:
            logger.error(f'LLM question analysis error: {e}')
            raise

    def _rule_based_analyze_question(self, question: str) -> Dict[str, Any]:
        """Fallback rule-based question analysis."""
        question_lower = question.lower()

        # Define keywords for different question types
        module_keywords = ['module', 'component', 'class', 'function', 'service', 'controller', 'api', 'router', 'model']
        dataflow_keywords = ['flow', 'data', 'process', 'communication', 'message', 'event', 'request', 'response']
        architecture_keywords = ['architecture', 'structure', 'design', 'pattern', 'layer', 'system', 'overview']
        technology_keywords = ['technology', 'stack', 'framework', 'library', 'database', 'tool']

        # Extract potential keywords
        words = question_lower.split()
        extracted_keywords = []

        # Check for specific modules or components mentioned
        for word in words:
            if len(word) > 3 and word.isalpha():
                extracted_keywords.append(word)

        # Determine question type
        question_type = 'general'

        if any(keyword in question_lower for keyword in module_keywords):
            question_type = 'module_specific'
        elif any(keyword in question_lower for keyword in dataflow_keywords):
            question_type = 'dataflow_specific'
        elif any(keyword in question_lower for keyword in architecture_keywords):
            question_type = 'architecture_general'
        elif any(keyword in question_lower for keyword in technology_keywords):
            question_type = 'technology_related'

        return {
            'type': question_type,
            'keywords': extracted_keywords[:5],  # Limit to 5 keywords
            'confidence': 0.6,  # Lower confidence for rule-based
            'reasoning': 'Rule-based classification',
            'original_question': question,
            'method': 'rule_based',
        }

    def analyze_page_type_with_llm(self, page_title: str) -> str:
        """Analyze page type using LLM for intelligent classification."""
        try:
            # Use LLM for intelligent page type classification
            return self._llm_analyze_page_type(page_title)
        except Exception as e:
            logger.warning(f'LLM page analysis failed, falling back to rule-based: {e}')
            # Fallback to rule-based classification
            return self._rule_based_analyze_page_type(page_title)

    def _llm_analyze_page_type(self, page_title: str) -> str:
        """Use LLM to analyze page type intelligently."""
        page_analysis_prompt = """
You are an expert technical documentation analyst. Analyze the following page title and classify what type of documentation page it should be.

PAGE TYPES:
1. architecture_overview: System overview, high-level architecture, system design
2. technology_integration: Technology stack, framework integration, tool usage
3. data_flow: Data processing, information flow, request/response patterns
4. component_cluster: Specific component groups, module clusters, service groups
5. deployment_infrastructure: Deployment, infrastructure, DevOps, environment setup
6. feature_functionality: Specific features, business functionality, user workflows

PAGE TITLE: "{page_title}"

Consider:
- What type of information would readers expect?
- What documentation structure would be most helpful?
- What technical aspects should be emphasized?

Respond with ONLY the page type (one of the 6 options above):
"""

        try:
            response = self.llm.invoke(page_analysis_prompt.format(page_title=page_title))
            page_type = response.content.strip().lower()

            # Validate response
            valid_types = [
                'architecture_overview', 'technology_integration', 'data_flow',
                'component_cluster', 'deployment_infrastructure', 'feature_functionality',
            ]

            if page_type in valid_types:
                logger.info(f"LLM classified '{page_title}' as '{page_type}'")
                return page_type
            else:
                raise ValueError(f'Invalid page type returned: {page_type}')

        except Exception as e:
            logger.error(f'LLM page analysis error: {e}')
            raise

    def _rule_based_analyze_page_type(self, page_title: str) -> str:
        """Fallback rule-based page type analysis."""
        title_lower = page_title.lower()

        # Architecture and system design
        if any(word in title_lower for word in ['architecture', 'system design', 'overview', 'structure']):
            return 'architecture_overview'

        # Technology, integration, API
        elif any(word in title_lower for word in ['integration', 'api', 'llm', 'vector', 'database', 'technology']):
            return 'technology_integration'

        # Data flow and processing
        elif any(word in title_lower for word in ['data flow', 'processing', 'pipeline', 'workflow']):
            return 'data_flow'

        # Deployment and infrastructure
        elif any(word in title_lower for word in ['deployment', 'infrastructure', 'setup', 'configuration']):
            return 'deployment_infrastructure'

        # Component or cluster specific
        elif any(word in title_lower for word in ['component', 'module', 'service', 'backend', 'frontend']):
            return 'component_cluster'

        # Features and functionality
        elif any(word in title_lower for word in ['feature', 'functionality', 'capability', 'user']):
            return 'feature_functionality'

        else:
            return 'feature_functionality'  # Default fallback
