"""
Multi-modal graph construction for code architecture analysis.

This module builds comprehensive graphs that capture different types of relationships
between code components: syntactic, semantic, structural, and dependency-based.
"""
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

import numpy as np
from shared.logging import get_logger

from .ast_parser import CodeComponent
from .ast_parser import DependencyParser

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
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[Tuple[str, str], GraphEdge] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency_list: Dict[str, List[str]] = defaultdict(list)

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
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.graph = MultiModalGraph()
        self.dependency_parser = DependencyParser(repo_path)
        self._file_contents: Dict[str, str] = {}
        self._import_graph: Dict[str, Set[str]] = defaultdict(set)

    def build_graph(self) -> MultiModalGraph:
        """
        Build the complete multi-modal graph.

        Returns:
            MultiModalGraph: The constructed graph
        """
        logger.info('Building multi-modal graph...')

        # Step 1: Parse AST and collect components
        self._parse_code_structure()

        # Step 2: Extract different types of relationships
        self._extract_import_relationships()
        self._extract_call_relationships()
        self._extract_inheritance_relationships()
        self._extract_structural_relationships()

        # Step 3: Compute code metrics
        self._compute_code_metrics()

        # Step 4: Classify modules
        self._classify_modules()

        logger.info(f'Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges')
        return self.graph

    def _parse_code_structure(self):
        """Parse code structure using AST parser."""
        logger.info('Parsing code structure...')

        # Use existing AST parser
        components = self.dependency_parser.parse_repository()

        # Create graph nodes from components
        for component_id, component in components.items():
            # Read file content for embedding
            content = self._get_file_content(component.file_path)

            # Create graph node
            node = GraphNode(
                id=component_id,
                module_path=component_id,
                file_path=component.file_path,
                relative_path=component.relative_path,
                node_type=component.component_type,
                content=component.source_code or content[:1000],  # First 1000 chars for embedding
            )

            self.graph.add_node(node)

            # Add edges for dependencies
            for dep_id in component.depends_on:
                if dep_id in components:
                    edge = GraphEdge(
                        source=component_id,
                        target=dep_id,
                        relationship_types=[RelationshipType.DEPENDENCY],
                        interaction_weight=0.5,
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
        """Extract import-based relationships."""
        logger.info('Extracting import relationships...')

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

    def _extract_call_relationships(self):
        """Extract function/method call relationships."""
        logger.info('Extracting call relationships...')

        for file_path in self._get_python_files():
            content = self._get_file_content(file_path)
            relative_path = os.path.relpath(file_path, self.repo_path)
            module_path = self._file_to_module_path(relative_path)

            try:
                tree = ast.parse(content)

                # Find function calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            # Direct function call
                            self._analyze_function_call(module_path, node.func.id)

            except SyntaxError as e:
                logger.warning(f'Syntax error in {file_path}: {e}')

    def _extract_inheritance_relationships(self):
        """Extract class inheritance relationships."""
        logger.info('Extracting inheritance relationships...')

        for file_path in self._get_python_files():
            content = self._get_file_content(file_path)
            relative_path = os.path.relpath(file_path, self.repo_path)
            module_path = self._file_to_module_path(relative_path)

            try:
                tree = ast.parse(content)

                # Find class definitions with base classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.bases:
                        class_id = f'{module_path}.{node.name}'

                        for base in node.bases:
                            base_name = self._extract_base_class_name(base)
                            if base_name:
                                self._add_inheritance_edge(class_id, base_name)

            except SyntaxError as e:
                logger.warning(f'Syntax error in {file_path}: {e}')

    def _extract_structural_relationships(self):
        """Extract structural relationships based on file system organization."""
        logger.info('Extracting structural relationships...')

        # Group files by directory
        directory_modules = defaultdict(list)

        for node_id, node in self.graph.nodes.items():
            directory = os.path.dirname(node.relative_path)
            directory_modules[directory].append(node_id)

        # Add structural edges between modules in the same directory
        for directory, modules in directory_modules.items():
            for i, module1 in enumerate(modules):
                for module2 in modules[i + 1:]:
                    # Add bidirectional structural relationship
                    edge1 = GraphEdge(
                        source=module1,
                        target=module2,
                        relationship_types=[RelationshipType.STRUCTURAL],
                        interaction_weight=0.3,
                        metadata={'shared_directory': directory},
                    )
                    edge2 = GraphEdge(
                        source=module2,
                        target=module1,
                        relationship_types=[RelationshipType.STRUCTURAL],
                        interaction_weight=0.3,
                        metadata={'shared_directory': directory},
                    )
                    self.graph.add_edge(edge1)
                    self.graph.add_edge(edge2)

    def _compute_code_metrics(self):
        """Compute code complexity metrics for each node."""
        logger.info('Computing code metrics...')

        for node_id, node in self.graph.nodes.items():
            content = self._get_file_content(node.file_path)

            # Lines of code
            node.lines_of_code = len([line for line in content.split('\n') if line.strip()])

            # Complexity metrics
            node.complexity_metrics = self._calculate_complexity_metrics(content)

    def _classify_modules(self):
        """Classify modules by type, domain, and layer."""
        logger.info('Classifying modules...')

        for node_id, node in self.graph.nodes.items():
            # Module type classification
            node.module_type = self._classify_module_type(node)

            # Domain classification
            node.domain = self._classify_domain(node)

            # Layer classification
            node.layer = self._classify_layer(node)

            # Semantic tags
            node.semantic_tags = self._extract_semantic_tags(node)

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
                    relationship_types=[RelationshipType.CALLS],
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
        """Classify module type based on patterns."""
        path_lower = node.relative_path.lower()

        if 'test' in path_lower:
            return 'test'
        elif 'config' in path_lower or 'setting' in path_lower:
            return 'config'
        elif 'util' in path_lower or 'helper' in path_lower:
            return 'utility'
        elif 'api' in path_lower or 'router' in path_lower or 'endpoint' in path_lower:
            return 'interface'
        elif 'model' in path_lower or 'schema' in path_lower:
            return 'model'
        elif 'service' in path_lower or 'manager' in path_lower:
            return 'core'
        elif 'infra' in path_lower or 'db' in path_lower:
            return 'infrastructure'
        else:
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
