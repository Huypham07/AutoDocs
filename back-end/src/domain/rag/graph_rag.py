from __future__ import annotations

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

        if query.query_type == 'system_overview':
            return self._build_overview_context(query)
        elif query.query_type == 'module_details':
            return self._build_module_context(query)
        elif query.query_type == 'data_flows':
            return self._build_dataflow_context(query)
        elif query.query_type == 'cluster_analysis':
            return self._build_cluster_context(query)
        elif query.query_type == 'technology_stack':
            return self._build_technology_context(query)
        elif query.query_type == 'communication_patterns':
            return self._build_communication_context(query)
        elif query.query_type == 'dependencies':
            return self._build_dependency_context(query)
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
            with open('context.txt', 'w') as f:
                f.write(question + '\n' + context.narrative_text)

            # Generate response using LCEL chain
            response = self.chain.invoke({
                'question': question,
                'graph_context': context.narrative_text,
                'system_role': system_role,
            })

            with open('response.txt', 'w') as f:
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
        except Exception as e:
            logger.error(f'Error getting available modules: {e}')
            return []

    def get_available_clusters(self, repo_url: str) -> List[str]:
        """Get list of available clusters for the repository."""
        try:
            clusters = self.graph_repo.get_cluster_analysis(repo_url)
            return [cluster['name'] for cluster in clusters]
        except Exception as e:
            logger.error(f'Error getting available clusters: {e}')
            return []
