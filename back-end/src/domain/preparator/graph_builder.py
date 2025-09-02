from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from shared.logging import get_logger

from .ast_parser import DependencyParser

logger = get_logger(__name__)

# Try to import multi-language parser, fallback to Python-only
try:
    from .tree_sitter_parser import MultiLanguageDependencyParser
    MULTI_LANGUAGE_SUPPORT = True
except ImportError:
    MULTI_LANGUAGE_SUPPORT = False
    logger.warning('Multi-language support not available. Install tree-sitter for full language support.')

logger = get_logger(__name__)


class RelationshipType(Enum):
    """Types of relationships between code modules."""
    IMPORTS = 'imports'
    CALLS = 'calls'
    INHERITS = 'inherits'
    COMPOSES = 'composes'
    SEMANTIC = 'semantic'
    STRUCTURAL = 'structural'
    DEPENDENCY = 'dependency'


@dataclass
class GraphNode:
    """Represents a node in the multi-modal graph."""
    id: str
    module_path: str
    file_path: str
    relative_path: str
    node_type: str  # 'module', 'class', 'function'

    # Code metrics
    lines_of_code: int = 0
    complexity_metrics: Dict[str, float] = field(default_factory=dict)

    # Graph metrics (will be computed later)
    centrality_measures: Dict[str, float] = field(default_factory=dict)

    # Semantic information
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = field(default_factory=list)

    # Classification
    module_type: str = 'unknown'  # 'core', 'utility', 'interface', 'config'
    domain: str = 'unknown'
    layer: str = 'unknown'  # 'presentation', 'business', 'data', 'infrastructure'

    # Content for embedding
    content: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'module_path': self.module_path,
            'file_path': self.file_path,
            'relative_path': self.relative_path,
            'node_type': self.node_type,
            'lines_of_code': self.lines_of_code,
            'complexity_metrics': self.complexity_metrics,
            'centrality_measures': self.centrality_measures,
            'embedding': self.embedding,
            'semantic_tags': self.semantic_tags,
            'module_type': self.module_type,
            'domain': self.domain,
            'layer': self.layer,
            'content': self.content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphNode:
        """Create from dictionary."""
        return cls(
            id=data['id'],
            module_path=data['module_path'],
            file_path=data['file_path'],
            relative_path=data['relative_path'],
            node_type=data['node_type'],
            lines_of_code=data.get('lines_of_code', 0),
            complexity_metrics=data.get('complexity_metrics', {}),
            centrality_measures=data.get('centrality_measures', {}),
            embedding=data.get('embedding'),
            semantic_tags=data.get('semantic_tags', []),
            module_type=data.get('module_type', 'unknown'),
            domain=data.get('domain', 'unknown'),
            layer=data.get('layer', 'unknown'),
            content=data.get('content', ''),
        )


@dataclass
class GraphEdge:
    """Represents an edge in the multi-modal graph."""
    source: str
    target: str
    relationship_types: List[RelationshipType] = field(default_factory=list)

    # Relationship strength
    interaction_weight: float = 0.0
    call_frequency: int = 0
    dependency_strength: str = 'weak'  # 'weak', 'medium', 'strong'

    # Semantic similarity
    semantic_similarity: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'relationship_types': [rt.value for rt in self.relationship_types],
            'interaction_weight': self.interaction_weight,
            'call_frequency': self.call_frequency,
            'dependency_strength': self.dependency_strength,
            'semantic_similarity': self.semantic_similarity,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphEdge:
        """Create from dictionary."""
        return cls(
            source=data['source'],
            target=data['target'],
            relationship_types=[RelationshipType(rt) for rt in data.get('relationship_types', [])],
            interaction_weight=data.get('interaction_weight', 0.0),
            call_frequency=data.get('call_frequency', 0),
            dependency_strength=data.get('dependency_strength', 'weak'),
            semantic_similarity=data.get('semantic_similarity', 0.0),
            metadata=data.get('metadata', {}),
        )


class MultiModalGraph:
    """
    Multi-modal graph that captures different types of relationships between code components.
    """

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency_list = defaultdict(list)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency_list:
            self.adjacency_list[node.id] = []
        if node.id not in self.reverse_adjacency_list:
            self.reverse_adjacency_list[node.id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        edge_key = (edge.source, edge.target)

        if edge_key in self.edges:
            # Merge with existing edge
            existing_edge = self.edges[edge_key]
            existing_edge.relationship_types.extend(edge.relationship_types)
            existing_edge.relationship_types = list(set(existing_edge.relationship_types))
            existing_edge.interaction_weight = max(existing_edge.interaction_weight, edge.interaction_weight)
            existing_edge.call_frequency += edge.call_frequency
            existing_edge.semantic_similarity = max(existing_edge.semantic_similarity, edge.semantic_similarity)
        else:
            self.edges[edge_key] = edge
            self.adjacency_list[edge.source].append(edge.target)
            self.reverse_adjacency_list[edge.target].append(edge.source)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_edge(self, source: str, target: str) -> Optional[GraphEdge]:
        """Get edge between two nodes."""
        return self.edges.get((source, target))

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node."""
        return self.adjacency_list.get(node_id, [])

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all predecessors of a node."""
        return self.reverse_adjacency_list.get(node_id, [])

    def get_subgraph(self, node_ids: List[str]) -> MultiModalGraph:
        """Extract a subgraph containing only the specified nodes."""
        subgraph = MultiModalGraph()

        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])

        # Add edges between the nodes
        for (source, target), edge in self.edges.items():
            if source in node_ids and target in node_ids:
                subgraph.add_edge(edge)

        return subgraph

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {
                f'{source}->{target}': edge.to_dict()
                for (source, target), edge in self.edges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiModalGraph:
        """Create from dictionary."""
        graph = cls()

        # Load nodes
        for node_id, node_data in data.get('nodes', {}).items():
            node = GraphNode.from_dict(node_data)
            graph.add_node(node)

        # Load edges
        for edge_key, edge_data in data.get('edges', {}).items():
            edge = GraphEdge.from_dict(edge_data)
            graph.add_edge(edge)

        return graph


class GraphBuilder:
    """
    Builds multi-modal graphs from code repositories by extracting different types of relationships.
    Supports multiple programming languages via tree-sitter when available.
    """

    def __init__(self, repo_path: str, use_multi_language: bool = True):
        self.repo_path = os.path.abspath(repo_path)
        self.graph = MultiModalGraph()
        self.use_multi_language = use_multi_language and MULTI_LANGUAGE_SUPPORT

        # Initialize appropriate parser
        if self.use_multi_language:
            logger.info('Using multi-language parser with tree-sitter')
            self.dependency_parser: Union[MultiLanguageDependencyParser, DependencyParser] = MultiLanguageDependencyParser(repo_path)
        else:
            logger.info('Using Python-only AST parser')
            self.dependency_parser = DependencyParser(repo_path)

        self._file_contents: Dict[str, str] = {}
        self._import_graph: Dict[str, Set[str]] = defaultdict(set)

    def build_graph(self) -> MultiModalGraph:
        """
        Build simplified graph focusing on essential relationships.
        Supports multiple programming languages when tree-sitter is available.

        Returns:
            MultiModalGraph: The constructed graph
        """
        if self.use_multi_language:
            logger.info('Building multi-language graph...')
        else:
            logger.info('Building Python-only graph...')

        # Step 1: Parse code structure using appropriate parser
        self._parse_code_structure()

        # Step 2: Extract only essential relationships
        self._extract_import_relationships()  # Keep imports (essential for dependencies)

        # Step 3: Basic classification (essential for clustering)
        self._classify_modules()

        # Step 4: Log language statistics if multi-language
        if self.use_multi_language and hasattr(self.dependency_parser, 'get_language_statistics'):
            stats = self.dependency_parser.get_language_statistics()
            logger.info(f'Language distribution: {stats}')

        logger.info(f'Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges')
        return self.graph

    def _parse_code_structure(self):
        """Parse code structure using appropriate parser (multi-language or Python-only)."""
        logger.info('Parsing code structure...')

        # Use appropriate parser
        components = self.dependency_parser.parse_repository()

        # Create graph nodes from components
        for component_id, component in components.items():
            # Read file content for embedding
            content = self._get_file_content(component.file_path)

            # Determine file language for better classification
            file_ext = os.path.splitext(component.file_path)[1]
            language = self._get_language_from_extension(file_ext)

            # Create graph node
            node = GraphNode(
                id=component_id,
                module_path=component_id,
                file_path=component.file_path,
                relative_path=component.relative_path,
                node_type=component.component_type,
                content=component.source_code or content[:1000],  # First 1000 chars for embedding
            )

            # Add language-specific metadata
            node.metadata = {'language': language, 'file_extension': file_ext}

            self.graph.add_node(node)

            # Add edges for dependencies
            for dep_id in component.depends_on:
                if dep_id in components:
                    edge = GraphEdge(
                        source=component_id,
                        target=dep_id,
                        relationship_types=[RelationshipType.DEPENDENCY],
                        interaction_weight=0.5,
                        metadata={'language': language},
                    )
                    self.graph.add_edge(edge)

    def _get_file_content(self, file_path: str) -> str:
        """Get content of a file."""
        if file_path not in self._file_contents:
            try:
                with open(file_path, encoding='utf-8') as f:
                    self._file_contents[file_path] = f.read()
            except Exception as e:
                logger.warning(f'Could not read file {file_path}: {e}')
                self._file_contents[file_path] = ''

        return self._file_contents[file_path]

    def _extract_import_relationships(self):
        """Extract import-based relationships for supported file types."""
        logger.info('Extracting import relationships...')

        if self.use_multi_language:
            # Multi-language approach - relationships already extracted in parse_repository
            logger.info('Import relationships extracted during multi-language parsing')
            return

        # Python-only approach (existing logic)
        for file_path in self._get_python_files():
            content = self._get_file_content(file_path)
            relative_path = os.path.relpath(file_path, self.repo_path)
            module_path = self._file_to_module_path(relative_path)

            try:
                tree = ast.parse(content)

                # Find import statements
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_module = alias.name
                            self._add_import_edge(module_path, imported_module)

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported_module = node.module
                            self._add_import_edge(module_path, imported_module)

            except SyntaxError as e:
                logger.warning(f'Syntax error in {file_path}: {e}')

    # The following methods are disabled for simplified pipeline
    # def _extract_call_relationships(self):
    #     """Extract function/method call relationships - Disabled."""
    #     pass

    # def _extract_inheritance_relationships(self):
    #     """Extract class inheritance relationships - Disabled."""
    #     pass

    # def _extract_structural_relationships(self):
    #     """Extract structural relationships - Disabled."""
    #     pass

    # def _compute_code_metrics(self):
    #     """Compute code complexity metrics - Disabled."""
    #     pass

    def _classify_modules(self):
        """Simplified module classification with multi-language support."""
        logger.info('Classifying modules...')

        for node_id, node in self.graph.nodes.items():
            # Simple module type classification based on path
            node.module_type = self._classify_module_type(node)

            # Simple layer classification
            node.layer = self._classify_layer(node)

            # Simple domain classification based on directory
            node.domain = self._classify_domain(node)

    def _get_language_from_extension(self, file_ext: str) -> str:
        """Get language name from file extension."""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.go': 'go',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.rs': 'rust',
            '.kt': 'kotlin',
            '.scala': 'scala',
        }
        return language_map.get(file_ext.lower(), 'unknown')

    def get_supported_languages(self) -> List[str]:
        """Get list of languages detected in the repository."""
        languages = set()
        for node in self.graph.nodes.values():
            if hasattr(node, 'metadata') and 'language' in node.metadata:
                languages.add(node.metadata['language'])
        return list(languages)

    def get_language_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get detailed statistics about languages in the repository."""
        stats = {}

        for node in self.graph.nodes.values():
            language = 'unknown'
            if hasattr(node, 'metadata') and 'language' in node.metadata:
                language = node.metadata['language']

            if language not in stats:
                stats[language] = {'classes': 0, 'functions': 0, 'methods': 0, 'total': 0}

            stats[language][node.node_type] = stats[language].get(node.node_type, 0) + 1
            stats[language]['total'] += 1

        return stats

    def _get_python_files(self) -> List[str]:
        """Get all Python files in the repository."""
        python_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files

    def _file_to_module_path(self, relative_path: str) -> str:
        """Convert file path to module path."""
        path = relative_path[:-3] if relative_path.endswith('.py') else relative_path
        return path.replace(os.path.sep, '.')

    def _add_import_edge(self, source_module: str, imported_module: str):
        """Add an import relationship edge."""
        # First, try to find existing nodes for the source module
        source_nodes = [
            node_id for node_id in self.graph.nodes.keys()
            if node_id.startswith(source_module)
        ]

        # If no specific nodes found, create a module-level node
        if not source_nodes:
            # Create a module-level node if it doesn't exist
            if source_module not in self.graph.nodes:
                module_node = GraphNode(
                    id=source_module,
                    module_path=source_module,
                    file_path='',
                    relative_path=source_module.replace('.', '/') + '.py',
                    node_type='module',
                )
                self.graph.add_node(module_node)
            source_nodes = [source_module]

        # Check if the imported module exists in our graph
        target_nodes = [
            node_id for node_id in self.graph.nodes.keys()
            if node_id.startswith(imported_module)
        ]

        # If target module not found, create a module-level node
        if not target_nodes:
            if imported_module not in self.graph.nodes:
                target_module_node = GraphNode(
                    id=imported_module,
                    module_path=imported_module,
                    file_path='',
                    relative_path=imported_module.replace('.', '/') + '.py',
                    node_type='module',
                )
                self.graph.add_node(target_module_node)
            target_nodes = [imported_module]

        # Create edges between all source and target combinations
        for source_node in source_nodes[:3]:  # Limit to avoid too many edges
            for target_node in target_nodes[:3]:
                if source_node != target_node:  # Avoid self-loops
                    edge = GraphEdge(
                        source=source_node,
                        target=target_node,
                        relationship_types=[RelationshipType.IMPORTS],
                        interaction_weight=0.4,
                    )
                    self.graph.add_edge(edge)

    def _analyze_function_call(self, caller_module: str, function_name: str):
        """Analyze a function call to extract relationships."""
        # Look for functions with this name in other modules
        for node_id, node in self.graph.nodes.items():
            if node.node_type == 'function' and node_id.endswith(f'.{function_name}'):
                edge = GraphEdge(
                    source=caller_module,
                    target=node_id,
                    relationship_types=[RelationshipType.DEPENDENCY],  # Use DEPENDENCY instead of DEPENDS_ON
                    interaction_weight=0.6,
                    call_frequency=1,
                )
                self.graph.add_edge(edge)

    def _extract_base_class_name(self, base_node: ast.AST) -> Optional[str]:
        """Extract base class name from AST node."""
        if isinstance(base_node, ast.Name):
            return base_node.id
        elif isinstance(base_node, ast.Attribute):
            # Handle qualified names like module.ClassName
            parts: List[str] = []
            current: Union[ast.Attribute, ast.expr] = base_node
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
                return '.'.join(parts)
        return None

    def _add_inheritance_edge(self, child_class: str, base_class: str):
        """Add an inheritance relationship edge."""
        # Find the child class in our graph
        child_nodes = [
            node_id for node_id in self.graph.nodes.keys()
            if node_id.endswith(f'.{child_class}') or node_id == child_class
        ]

        # Find the base class in our graph
        base_nodes = [
            node_id for node_id in self.graph.nodes.keys()
            if node_id.endswith(f'.{base_class}') or node_id == base_class
        ]

        # If either class is not found, create module-level placeholders
        if not child_nodes:
            if child_class not in self.graph.nodes:
                child_node = GraphNode(
                    id=child_class,
                    module_path=child_class,
                    file_path='',
                    relative_path='',
                    node_type='class',
                )
                self.graph.add_node(child_node)
            child_nodes = [child_class]

        if not base_nodes:
            if base_class not in self.graph.nodes:
                base_node = GraphNode(
                    id=base_class,
                    module_path=base_class,
                    file_path='',
                    relative_path='',
                    node_type='class',
                )
                self.graph.add_node(base_node)
            base_nodes = [base_class]

        # Create inheritance edges
        for child in child_nodes:
            for base in base_nodes:
                if child != base:
                    edge = GraphEdge(
                        source=child,
                        target=base,
                        relationship_types=[RelationshipType.INHERITS],
                        interaction_weight=0.8,
                    )
                    self.graph.add_edge(edge)

    def _calculate_complexity_metrics(self, content: str) -> Dict[str, float]:
        """Calculate complexity metrics for code content."""
        metrics: Dict[str, float] = {}

        try:
            tree = ast.parse(content)

            # Cyclomatic complexity (simplified)
            complexity = 1  # Base complexity
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            metrics['cyclomatic_complexity'] = float(complexity)

            # Function count
            function_count = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            metrics['function_count'] = float(function_count)

            # Class count
            class_count = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            )
            metrics['class_count'] = float(class_count)

            # Import count
            import_count = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            metrics['import_count'] = float(import_count)

        except SyntaxError:
            # Default values for unparseable files
            metrics = {
                'cyclomatic_complexity': 1.0,
                'function_count': 0.0,
                'class_count': 0.0,
                'import_count': 0.0,
            }

        return metrics

    def _classify_module_type(self, node: GraphNode) -> str:
        """Classify module type based on patterns, with multi-language support."""
        path_lower = node.relative_path.lower()

        # Get language for language-specific classification
        language = 'python'  # default
        if hasattr(node, 'metadata') and 'language' in node.metadata:
            language = node.metadata['language']

        # Common patterns across languages
        if 'test' in path_lower or 'spec' in path_lower:
            return 'test'
        elif 'config' in path_lower or 'setting' in path_lower:
            return 'config'
        elif 'util' in path_lower or 'helper' in path_lower:
            return 'utility'
        elif 'model' in path_lower or 'schema' in path_lower or 'entity' in path_lower:
            return 'model'
        elif 'service' in path_lower or 'manager' in path_lower:
            return 'core'
        elif 'infra' in path_lower or 'db' in path_lower or 'database' in path_lower:
            return 'infrastructure'

        # Language-specific patterns
        if language in ['javascript', 'typescript']:
            if 'component' in path_lower or 'page' in path_lower or 'view' in path_lower:
                return 'interface'
            elif 'api' in path_lower or 'router' in path_lower or 'endpoint' in path_lower or 'controller' in path_lower:
                return 'interface'
            elif 'hook' in path_lower or 'context' in path_lower:
                return 'utility'
        elif language == 'java':
            if 'controller' in path_lower or 'rest' in path_lower:
                return 'interface'
            elif 'repository' in path_lower or 'dao' in path_lower:
                return 'infrastructure'
            elif 'dto' in path_lower or 'vo' in path_lower:
                return 'model'
        elif language == 'python':
            if 'api' in path_lower or 'router' in path_lower or 'endpoint' in path_lower:
                return 'interface'
        elif language in ['go']:
            if 'handler' in path_lower or 'controller' in path_lower:
                return 'interface'
            elif 'repo' in path_lower or 'store' in path_lower:
                return 'infrastructure'

        return 'core'

    def _classify_domain(self, node: GraphNode) -> str:
        """Classify domain based on directory structure and content."""
        path_parts = node.relative_path.lower().split(os.path.sep)

        # Domain keywords mapping
        domain_keywords = {
            'auth': 'authentication',
            'user': 'user_management',
            'db': 'data_access',
            'api': 'api_layer',
            'chat': 'communication',
            'doc': 'documentation',
            'rag': 'retrieval',
            'llm': 'ai_services',
            'embed': 'embedding_services',
            'mongo': 'database',
            'rabbit': 'messaging',
        }

        for part in path_parts:
            for keyword, domain in domain_keywords.items():
                if keyword in part:
                    return domain

        return 'core'

    def _classify_layer(self, node: GraphNode) -> str:
        """Classify architectural layer."""
        path_lower = node.relative_path.lower()

        if 'api' in path_lower or 'router' in path_lower:
            return 'presentation'
        elif 'application' in path_lower or 'service' in path_lower:
            return 'business'
        elif 'domain' in path_lower:
            return 'business'
        elif 'infra' in path_lower or 'db' in path_lower or 'mongo' in path_lower:
            return 'data'
        elif 'shared' in path_lower:
            return 'infrastructure'
        else:
            return 'business'

    def _extract_semantic_tags(self, node: GraphNode) -> List[str]:
        """Extract semantic tags from module content and structure."""
        tags = []

        # Add tags based on file path
        path_parts = node.relative_path.lower().split(os.path.sep)
        tags.extend(path_parts)

        # Add tags based on module type
        tags.append(node.module_type)
        tags.append(node.domain)
        tags.append(node.layer)

        # Add tags based on complexity
        if node.complexity_metrics.get('cyclomatic_complexity', 0) > 10:
            tags.append('complex')

        if node.lines_of_code > 200:
            tags.append('large')

        return list(set(tags))  # Remove duplicates


def save_graph(graph: MultiModalGraph, output_path: str):
    """Save graph to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph.to_dict(), f, indent=2)

    logger.info(f'Saved graph to {output_path}')


def load_graph(input_path: str) -> MultiModalGraph:
    """Load graph from JSON file."""
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    graph = MultiModalGraph.from_dict(data)
    logger.info(f'Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges')
    return graph
