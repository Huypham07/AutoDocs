from __future__ import annotations

import re
from typing import Any
from typing import Dict
from typing import List

from shared.logging import get_logger

logger = get_logger(__name__)


class NLQueryTranslator:
    """Translates natural language queries to Cypher queries."""

    def __init__(self):
        self.module_keywords = [
            'module', 'component', 'service', 'class', 'file',
            'xử lý', 'handle', 'process', 'manage', 'quản lý',
        ]

        self.dataflow_keywords = [
            'flow', 'luồng', 'dữ liệu', 'data', 'process',
            'workflow', 'pipeline', 'stream',
        ]

        self.technology_keywords = [
            'technology', 'tech', 'framework', 'library',
            'database', 'công nghệ', 'framework', 'thư viện',
        ]

        self.authentication_terms = [
            'authentication', 'auth', 'login', 'signin',
            'xác thực', 'đăng nhập', 'security', 'bảo mật',
        ]

        self.payment_terms = [
            'payment', 'pay', 'thanh toán', 'billing',
            'transaction', 'giao dịch', 'money', 'tiền',
        ]

    def translate_query(self, question: str, repo_url: str) -> Dict[str, Any]:
        """
        Translate natural language question to structured query information.

        Returns:
            Dict containing:
            - query_type: Type of query (module_search, dataflow_search, etc.)
            - cypher_query: Generated Cypher query
            - parameters: Query parameters
            - intent: Parsed intent information
        """
        question_lower = question.lower()

        # Phân tích intent
        intent = self._analyze_intent(question_lower)

        # Generate Cypher query based on intent
        if intent['type'] == 'module_search':
            return self._generate_module_query(intent, repo_url)
        elif intent['type'] == 'dataflow_search':
            return self._generate_dataflow_query(intent, repo_url)
        elif intent['type'] == 'technology_search':
            return self._generate_technology_query(intent, repo_url)
        elif intent['type'] == 'architecture_overview':
            return self._generate_overview_query(repo_url)
        else:
            return self._generate_general_query(question, repo_url)

    def _analyze_intent(self, question: str) -> Dict[str, Any]:
        """Analyze the intent of the natural language question."""

        # Tìm kiếm module
        if any(keyword in question for keyword in self.module_keywords):
            # Xác định chức năng cụ thể
            functionality = None
            if any(term in question for term in self.authentication_terms):
                functionality = 'authentication'
            elif any(term in question for term in self.payment_terms):
                functionality = 'payment'

            return {
                'type': 'module_search',
                'functionality': functionality,
                'keywords': self._extract_relevant_keywords(question),
            }

        # Tìm kiếm data flow
        elif any(keyword in question for keyword in self.dataflow_keywords):
            context = None
            if any(term in question for term in self.payment_terms):
                context = 'payment'
            elif any(term in question for term in self.authentication_terms):
                context = 'authentication'

            return {
                'type': 'dataflow_search',
                'context': context,
                'keywords': self._extract_relevant_keywords(question),
            }

        # Tìm kiếm technology
        elif any(keyword in question for keyword in self.technology_keywords):
            return {
                'type': 'technology_search',
                'keywords': self._extract_relevant_keywords(question),
            }

        # Architecture overview
        elif any(word in question for word in ['architecture', 'kiến trúc', 'overview', 'tổng quan', 'structure', 'cấu trúc']):
            return {
                'type': 'architecture_overview',
                'keywords': [],
            }

        else:
            return {
                'type': 'general',
                'keywords': self._extract_relevant_keywords(question),
            }

    def _extract_relevant_keywords(self, question: str) -> List[str]:
        """Extract relevant keywords from the question."""
        # Simple extraction - có thể cải thiện với NLP
        all_terms = self.authentication_terms + self.payment_terms + ['user', 'database', 'api', 'service']
        found_keywords = []

        for term in all_terms:
            if term in question:
                found_keywords.append(term)

        return found_keywords

    def _generate_module_query(self, intent: Dict, repo_url: str) -> Dict[str, Any]:
        """Generate Cypher query for module search."""

        if intent['functionality'] == 'authentication':
            # Tìm module xử lý authentication
            cypher = """
            MATCH (m:Module {repo_url: $repo_url})
            WHERE m.name =~ '.*[Aa]uth.*' OR
                  m.file_path =~ '.*auth.*' OR
                  m.name =~ '.*[Ss]ecurity.*' OR
                  m.name =~ '.*[Ll]ogin.*'
            OPTIONAL MATCH (m)-[:DEPENDS_ON]->(dep:Module {repo_url: $repo_url})
            OPTIONAL MATCH (m)<-[:DEPENDS_ON]-(dependent:Module {repo_url: $repo_url})
            OPTIONAL MATCH (m)-[:BELONGS_TO]->(c:Cluster {repo_url: $repo_url})
            RETURN m.name as module_name,
                   m.file_path as file_path,
                   m.layer as layer,
                   m.domain as domain,
                   collect(DISTINCT dep.name) as dependencies,
                   collect(DISTINCT dependent.name) as dependents,
                   c.name as cluster_name
            """

        elif intent['functionality'] == 'payment':
            # Tìm module xử lý payment
            cypher = """
            MATCH (m:Module {repo_url: $repo_url})
            WHERE m.name =~ '.*[Pp]ayment.*' OR
                  m.file_path =~ '.*payment.*' OR
                  m.name =~ '.*[Bb]illing.*' OR
                  m.name =~ '.*[Tt]ransaction.*'
            OPTIONAL MATCH (m)-[:DEPENDS_ON]->(dep:Module {repo_url: $repo_url})
            OPTIONAL MATCH (m)<-[:DEPENDS_ON]-(dependent:Module {repo_url: $repo_url})
            OPTIONAL MATCH (m)-[:BELONGS_TO]->(c:Cluster {repo_url: $repo_url})
            RETURN m.name as module_name,
                   m.file_path as file_path,
                   m.layer as layer,
                   m.domain as domain,
                   collect(DISTINCT dep.name) as dependencies,
                   collect(DISTINCT dependent.name) as dependents,
                   c.name as cluster_name
            """
        else:
            # General module search
            cypher = """
            MATCH (m:Module {repo_url: $repo_url})
            OPTIONAL MATCH (m)-[:BELONGS_TO]->(c:Cluster {repo_url: $repo_url})
            RETURN m.name as module_name,
                   m.file_path as file_path,
                   m.layer as layer,
                   m.domain as domain,
                   c.name as cluster_name
            LIMIT 10
            """

        return {
            'query_type': 'module_search',
            'cypher_query': cypher,
            'parameters': {'repo_url': repo_url},
            'intent': intent,
        }

    def _generate_dataflow_query(self, intent: Dict, repo_url: str) -> Dict[str, Any]:
        """Generate Cypher query for data flow search."""

        if intent['context'] == 'payment':
            # Tìm data flow liên quan đến payment
            cypher = """
            MATCH (source:Module {repo_url: $repo_url})-[df:DATA_FLOW]->(target:Module {repo_url: $repo_url})
            WHERE source.name =~ '.*[Pp]ayment.*' OR
                  target.name =~ '.*[Pp]ayment.*' OR
                  df.data_type =~ '.*payment.*'
            RETURN source.name as source_module,
                   target.name as target_module,
                   df.flow_type as flow_type,
                   df.data_type as data_type,
                   df.frequency as frequency
            """

        elif intent['context'] == 'authentication':
            # Tìm data flow liên quan đến authentication
            cypher = """
            MATCH (source:Module {repo_url: $repo_url})-[df:DATA_FLOW]->(target:Module {repo_url: $repo_url})
            WHERE source.name =~ '.*[Aa]uth.*' OR
                  target.name =~ '.*[Aa]uth.*' OR
                  df.data_type =~ '.*auth.*'
            RETURN source.name as source_module,
                   target.name as target_module,
                   df.flow_type as flow_type,
                   df.data_type as data_type,
                   df.frequency as frequency
            """
        else:
            # General data flow search
            cypher = """
            MATCH (source:Module {repo_url: $repo_url})-[df:DATA_FLOW]->(target:Module {repo_url: $repo_url})
            RETURN source.name as source_module,
                   target.name as target_module,
                   df.flow_type as flow_type,
                   df.data_type as data_type,
                   df.frequency as frequency
            LIMIT 20
            """

        return {
            'query_type': 'dataflow_search',
            'cypher_query': cypher,
            'parameters': {'repo_url': repo_url},
            'intent': intent,
        }

    def _generate_technology_query(self, intent: Dict, repo_url: str) -> Dict[str, Any]:
        """Generate Cypher query for technology search."""

        cypher = """
        MATCH (t:Technology {repo_url: $repo_url})
        OPTIONAL MATCH (m:Module {repo_url: $repo_url})-[:USES]->(t)
        RETURN t.name as technology_name,
               t.category as category,
               t.version as version,
               collect(DISTINCT m.name) as used_by_modules
        """

        return {
            'query_type': 'technology_search',
            'cypher_query': cypher,
            'parameters': {'repo_url': repo_url},
            'intent': intent,
        }

    def _generate_overview_query(self, repo_url: str) -> Dict[str, Any]:
        """Generate Cypher query for architecture overview."""

        cypher = """
        MATCH (m:Module {repo_url: $repo_url})
        OPTIONAL MATCH (c:Cluster {repo_url: $repo_url})
        OPTIONAL MATCH (t:Technology {repo_url: $repo_url})
        RETURN count(DISTINCT m) as total_modules,
               count(DISTINCT c) as total_clusters,
               count(DISTINCT t) as total_technologies,
               collect(DISTINCT m.layer) as layers,
               collect(DISTINCT m.domain) as domains
        """

        return {
            'query_type': 'architecture_overview',
            'cypher_query': cypher,
            'parameters': {'repo_url': repo_url},
            'intent': {'type': 'architecture_overview'},
        }

    def _generate_general_query(self, question: str, repo_url: str) -> Dict[str, Any]:
        """Generate general query for unspecific questions."""

        # Fallback to basic system overview
        return self._generate_overview_query(repo_url)


class CypherQueryExecutor:
    """Executes Cypher queries and formats results."""

    def __init__(self, graph_repository):
        self.graph_repo = graph_repository

    def execute_translated_query(self, query_info: Dict) -> Dict[str, Any]:
        """Execute the translated query and return formatted results."""

        try:
            with self.graph_repo.connection.get_session() as session:
                result = session.run(
                    query_info['cypher_query'],
                    **query_info['parameters'],
                )

                records = [record.data() for record in result]

                return {
                    'success': True,
                    'query_type': query_info['query_type'],
                    'results': records,
                    'result_count': len(records),
                    'intent': query_info.get('intent', {}),
                    'formatted_answer': self._format_results(query_info['query_type'], records),
                }

        except Exception as e:
            logger.error(f'Error executing Cypher query: {e}')
            return {
                'success': False,
                'error': str(e),
                'query_type': query_info['query_type'],
                'results': [],
                'formatted_answer': f'Không thể truy vấn dữ liệu: {str(e)}',
            }

    def _format_results(self, query_type: str, results: List[Dict]) -> str:
        """Format query results into human-readable text."""

        if not results:
            return 'Không tìm thấy kết quả phù hợp trong hệ thống.'

        if query_type == 'module_search':
            return self._format_module_results(results)
        elif query_type == 'dataflow_search':
            return self._format_dataflow_results(results)
        elif query_type == 'technology_search':
            return self._format_technology_results(results)
        elif query_type == 'architecture_overview':
            return self._format_overview_results(results)
        else:
            return f'Tìm thấy {len(results)} kết quả trong hệ thống.'

    def _format_module_results(self, results: List[Dict]) -> str:
        """Format module search results."""
        if len(results) == 1:
            module = results[0]
            return f"""
Tìm thấy module: **{module.get('module_name', 'Unknown')}**
- Đường dẫn: `{module.get('file_path', 'Unknown')}`
- Layer: {module.get('layer', 'Unknown')}
- Domain: {module.get('domain', 'Unknown')}
- Cluster: {module.get('cluster_name', 'None')}
- Dependencies: {len(module.get('dependencies', []))} modules
- Dependents: {len(module.get('dependents', []))} modules
            """.strip()
        else:
            formatted = f'Tìm thấy {len(results)} modules:\n\n'
            for module in results[:5]:  # Limit to top 5
                formatted += f"- **{module.get('module_name', 'Unknown')}** ({module.get('layer', 'Unknown')} layer)\n"
            if len(results) > 5:
                formatted += f'\n... và {len(results) - 5} modules khác.'
            return formatted

    def _format_dataflow_results(self, results: List[Dict]) -> str:
        """Format data flow search results."""
        if not results:
            return 'Không tìm thấy data flow nào trong hệ thống.'

        formatted = f'Tìm thấy {len(results)} data flows:\n\n'
        for flow in results[:10]:  # Limit to top 10
            formatted += f"- **{flow.get('source_module', '?')}** → **{flow.get('target_module', '?')}**"
            if flow.get('flow_type'):
                formatted += f" ({flow['flow_type']})"
            formatted += '\n'

        if len(results) > 10:
            formatted += f'\n... và {len(results) - 10} flows khác.'

        return formatted

    def _format_technology_results(self, results: List[Dict]) -> str:
        """Format technology search results."""
        if not results:
            return 'Không tìm thấy thông tin technology nào.'

        formatted = f'Technology stack ({len(results)} components):\n\n'
        for tech in results:
            formatted += f"- **{tech.get('technology_name', 'Unknown')}**"
            if tech.get('category'):
                formatted += f" ({tech['category']})"
            if tech.get('version'):
                formatted += f" v{tech['version']}"
            formatted += '\n'

        return formatted

    def _format_overview_results(self, results: List[Dict]) -> str:
        """Format architecture overview results."""
        if not results:
            return 'Không thể lấy thông tin tổng quan về hệ thống.'

        overview = results[0]
        return f"""
## Tổng quan kiến trúc hệ thống

- **Tổng số modules**: {overview.get('total_modules', 0)}
- **Tổng số clusters**: {overview.get('total_clusters', 0)}
- **Tổng số technologies**: {overview.get('total_technologies', 0)}
- **Layers**: {', '.join(overview.get('layers', []))}
- **Domains**: {', '.join(overview.get('domains', []))}
        """.strip()
