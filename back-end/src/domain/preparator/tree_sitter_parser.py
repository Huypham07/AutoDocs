from __future__ import annotations

import json
import os
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Set

try:
    from tree_sitter import Parser, Node, Tree

    # Try to use py-tree-sitter-languages for easy language access
    try:
        from tree_sitter_languages import get_language
        LANGUAGES_AVAILABLE = True
    except ImportError:
        LANGUAGES_AVAILABLE = False
        print('Warning: py-tree-sitter-languages not installed. Install with: pip install py-tree-sitter-languages')

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    LANGUAGES_AVAILABLE = False
    print('Warning: tree-sitter not installed. Install with: pip install tree-sitter py-tree-sitter-languages')

from shared.logging import get_logger
from .ast_parser import CodeComponent

logger = get_logger(__name__)


# Language configurations
LANGUAGE_CONFIG = {
    '.py': {
        'name': 'python',
        'function_query': '(function_definition name: (identifier) @function.name)',
        'class_query': '(class_definition name: (identifier) @class.name)',
        'method_query': '(function_definition name: (identifier) @method.name)',
        'import_query': [
            '(import_statement name: (dotted_name) @import)',
            '(import_from_statement module_name: (dotted_name) @module)',
        ],
        'call_query': '(call function: (identifier) @call.name)',
        'docstring_query': '(expression_statement (string) @docstring)',
    },
    '.js': {
        'name': 'javascript',
        'function_query': '(function_declaration name: (identifier) @function.name)',
        'class_query': '(class_declaration name: (identifier) @class.name)',
        'method_query': '(method_definition name: (property_identifier) @method.name)',
        'import_query': [
            '(import_statement source: (string) @import)',
            '(import_statement (import_clause (named_imports (import_specifier name: (identifier) @import))))',
        ],
        'call_query': '(call_expression function: (identifier) @call.name)',
    },
    '.ts': {
        'name': 'typescript',
        'function_query': '(function_declaration name: (identifier) @function.name)',
        'class_query': '(class_declaration name: (type_identifier) @class.name)',
        'method_query': '(method_definition name: (property_identifier) @method.name)',
        'import_query': [
            '(import_statement source: (string) @import)',
            '(import_statement (import_clause (named_imports (import_specifier name: (identifier) @import))))',
        ],
        'call_query': '(call_expression function: (identifier) @call.name)',
    },
    '.java': {
        'name': 'java',
        'function_query': '(method_declaration name: (identifier) @method.name)',
        'class_query': '(class_declaration name: (identifier) @class.name)',
        'import_query': ['(import_declaration (scoped_identifier) @import)'],
        'call_query': '(method_invocation name: (identifier) @call.name)',
    },
    '.go': {
        'name': 'go',
        'function_query': '(function_declaration name: (identifier) @function.name)',
        'type_query': '(type_declaration (type_spec name: (type_identifier) @type.name))',
        'import_query': ['(import_declaration (import_spec path: (interpreted_string_literal) @import))'],
        'call_query': '(call_expression function: (identifier) @call.name)',
    },
    '.cpp': {
        'name': 'cpp',
        'function_query': '(function_definition declarator: (function_declarator declarator: (identifier) @function.name))',
        'class_query': '(class_specifier name: (type_identifier) @class.name)',
        'include_query': '(preproc_include path: (string_literal) @include)',
    },
    '.c': {
        'name': 'c',
        'function_query': '(function_definition declarator: (function_declarator declarator: (identifier) @function.name))',
        'struct_query': '(struct_specifier name: (type_identifier) @struct.name)',
        'include_query': '(preproc_include path: (string_literal) @include)',
    },
}


class BaseLanguageParser(ABC):
    """Base class for language-specific parsers."""

    def __init__(self, language_name: str, parser, language=None):
        """Initialize with language name and parser object."""
        self.language_name = language_name
        self.parser = parser
        self.language = language

    @classmethod
    def create_from_tree_sitter(cls, language_name: str, parser, language):
        """Create parser instance from tree-sitter parser object."""
        instance = cls.__new__(cls)
        instance.language_name = language_name
        instance.parser = parser
        instance.language = language
        return instance

    @abstractmethod
    def extract_components(self, source_code: str, file_path: str, relative_path: str) -> List[CodeComponent]:
        """Extract code components from source code."""
        pass

    @abstractmethod
    def extract_dependencies(self, source_code: str, tree: Tree) -> Set[str]:
        """Extract dependencies from source code."""
        pass


class PythonParser(BaseLanguageParser):
    """Parser for Python files using tree-sitter."""

    def extract_components(self, source_code: str, file_path: str, relative_path: str) -> List[CodeComponent]:
        """Extract Python components using tree-sitter."""
        components = []
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node

        module_path = self._file_to_module_path(relative_path)

        # Extract classes and functions
        components.extend(self._extract_classes(root_node, source_code, file_path, relative_path, module_path))
        components.extend(self._extract_functions(root_node, source_code, file_path, relative_path, module_path))

        return components

    def _extract_classes(self, root_node: Node, source_code: str, file_path: str, relative_path: str, module_path: str) -> List[CodeComponent]:
        """Extract class definitions."""
        components: List[CodeComponent] = []

        # Query for class definitions
        class_query = self._create_query('(class_definition name: (identifier) @class.name)')
        if not class_query:
            return components

        captures = class_query.captures(root_node)

        for node, capture_name in captures:
            if capture_name == 'class.name':
                class_node = node.parent  # Get the full class definition
                class_name = node.text.decode('utf8')
                class_id = f'{module_path}.{class_name}'

                # Check for docstring
                has_docstring, docstring = self._extract_docstring(class_node, source_code)

                component = CodeComponent(
                    id=class_id,
                    node=None,  # tree-sitter node, not AST
                    component_type='class',
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_node_text(class_node, source_code),
                    start_line=class_node.start_point[0] + 1,
                    end_line=class_node.end_point[0] + 1,
                    has_docstring=has_docstring,
                    docstring=docstring,
                )
                components.append(component)

                # Extract methods within the class
                components.extend(self._extract_methods(class_node, source_code, file_path, relative_path, class_id))

        return components

    def _extract_functions(self, root_node: Node, source_code: str, file_path: str, relative_path: str, module_path: str) -> List[CodeComponent]:
        """Extract top-level function definitions."""
        components: List[CodeComponent] = []

        # Query for function definitions at module level
        function_query = self._create_query('(function_definition name: (identifier) @function.name)')
        if not function_query:
            return components

        captures = function_query.captures(root_node)

        for node, capture_name in captures:
            if capture_name == 'function.name':
                func_node = node.parent  # Get the full function definition

                # Check if it's a top-level function (not inside a class)
                if self._is_top_level_function(func_node):
                    func_name = node.text.decode('utf8')
                    func_id = f'{module_path}.{func_name}'

                    # Check for docstring
                    has_docstring, docstring = self._extract_docstring(func_node, source_code)

                    component = CodeComponent(
                        id=func_id,
                        node=None,
                        component_type='function',
                        file_path=file_path,
                        relative_path=relative_path,
                        source_code=self._get_node_text(func_node, source_code),
                        start_line=func_node.start_point[0] + 1,
                        end_line=func_node.end_point[0] + 1,
                        has_docstring=has_docstring,
                        docstring=docstring,
                    )
                    components.append(component)

        return components

    def _extract_methods(self, class_node: Node, source_code: str, file_path: str, relative_path: str, class_id: str) -> List[CodeComponent]:
        """Extract methods within a class."""
        components: List[CodeComponent] = []

        # Query for function definitions within the class
        method_query = self._create_query('(function_definition name: (identifier) @method.name)')
        if not method_query:
            return components

        captures = method_query.captures(class_node)

        for node, capture_name in captures:
            if capture_name == 'method.name':
                method_node = node.parent
                method_name = node.text.decode('utf8')
                method_id = f'{class_id}.{method_name}'

                # Check for docstring
                has_docstring, docstring = self._extract_docstring(method_node, source_code)

                component = CodeComponent(
                    id=method_id,
                    node=None,
                    component_type='method',
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_node_text(method_node, source_code),
                    start_line=method_node.start_point[0] + 1,
                    end_line=method_node.end_point[0] + 1,
                    has_docstring=has_docstring,
                    docstring=docstring,
                )
                components.append(component)

        return components

    def extract_dependencies(self, source_code: str, tree: Tree) -> Set[str]:
        """Extract import dependencies."""
        dependencies = set()
        root_node = tree.root_node

        # Query for import statements
        import_queries = [
            '(import_statement name: (dotted_name) @import)',
            '(import_from_statement module_name: (dotted_name) @module)',
        ]

        for query_str in import_queries:
            query = self._create_query(query_str)
            if query:
                captures = query.captures(root_node)
                for node, capture_name in captures:
                    import_name = node.text.decode('utf8')
                    dependencies.add(import_name)

        return dependencies

    def _create_query(self, query_str: str):
        """Create a tree-sitter query."""
        try:
            return self.language.query(query_str)
        except Exception as e:
            logger.warning(f"Failed to create query '{query_str}': {e}")
            return None

    def _extract_docstring(self, node: Node, source_code: str) -> tuple[bool, str]:
        """Extract docstring from a function or class node."""
        # Look for the first string literal in the body
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'string':
                                docstring_text = self._get_node_text(expr, source_code)
                                # Remove quotes and clean up
                                docstring_text = docstring_text.strip('"\'')
                                return True, docstring_text
        return False, ''

    def _is_top_level_function(self, func_node: Node) -> bool:
        """Check if a function is at the top level (not inside a class)."""
        parent = func_node.parent
        while parent:
            if parent.type == 'class_definition':
                return False
            parent = parent.parent
        return True

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Get the text content of a node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code.encode('utf8')[start_byte:end_byte].decode('utf8')

    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to Python module path."""
        path = file_path[:-3] if file_path.endswith('.py') else file_path
        return path.replace(os.path.sep, '.')


class JavaScriptParser(BaseLanguageParser):
    """Parser for JavaScript/TypeScript files."""

    def extract_components(self, source_code: str, file_path: str, relative_path: str) -> List[CodeComponent]:
        """Extract JavaScript/TypeScript components."""
        components = []
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node

        module_path = self._file_to_module_path(relative_path)

        # Extract classes and functions
        components.extend(self._extract_classes(root_node, source_code, file_path, relative_path, module_path))
        components.extend(self._extract_functions(root_node, source_code, file_path, relative_path, module_path))

        return components

    def _extract_classes(self, root_node: Node, source_code: str, file_path: str, relative_path: str, module_path: str) -> List[CodeComponent]:
        """Extract class definitions."""
        components: List[CodeComponent] = []

        query_str = '(class_declaration (identifier) @class.name)'
        query = self._create_query(query_str)
        if not query:
            return components

        captures = query.captures(root_node)

        for node, capture_name in captures:
            if capture_name == 'class.name':
                class_node = node.parent
                class_name = node.text.decode('utf8')
                class_id = f'{module_path}.{class_name}'

                component = CodeComponent(
                    id=class_id,
                    node=None,
                    component_type='class',
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_node_text(class_node, source_code),
                    start_line=class_node.start_point[0] + 1,
                    end_line=class_node.end_point[0] + 1,
                    has_docstring=False,  # JS doesn't have built-in docstrings
                    docstring='',
                )
                components.append(component)

                # Extract methods
                components.extend(self._extract_methods(class_node, source_code, file_path, relative_path, class_id))

        return components

    def _extract_functions(self, root_node: Node, source_code: str, file_path: str, relative_path: str, module_path: str) -> List[CodeComponent]:
        """Extract function definitions."""
        components = []

        # Query for various function types
        function_queries = [
            '(function_declaration (identifier) @function.name)',
        ]

        for query_str in function_queries:
            query = self._create_query(query_str)
            if query:
                captures = query.captures(root_node)
                for node, capture_name in captures:
                    if capture_name == 'function.name':
                        func_node = node.parent
                        func_name = node.text.decode('utf8')
                        func_id = f'{module_path}.{func_name}'

                        component = CodeComponent(
                            id=func_id,
                            node=None,
                            component_type='function',
                            file_path=file_path,
                            relative_path=relative_path,
                            source_code=self._get_node_text(func_node, source_code),
                            start_line=func_node.start_point[0] + 1,
                            end_line=func_node.end_point[0] + 1,
                            has_docstring=False,
                            docstring='',
                        )
                        components.append(component)

        return components

        return components

    def _extract_methods(self, class_node: Node, source_code: str, file_path: str, relative_path: str, class_id: str) -> List[CodeComponent]:
        """Extract methods within a class."""
        components: List[CodeComponent] = []

        query_str = '(method_definition (property_identifier) @method.name)'
        query = self._create_query(query_str)
        if not query:
            return components

        captures = query.captures(class_node)

        for node, capture_name in captures:
            if capture_name == 'method.name':
                method_node = node.parent
                method_name = node.text.decode('utf8')
                method_id = f'{class_id}.{method_name}'

                component = CodeComponent(
                    id=method_id,
                    node=None,
                    component_type='method',
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_node_text(method_node, source_code),
                    start_line=method_node.start_point[0] + 1,
                    end_line=method_node.end_point[0] + 1,
                    has_docstring=False,
                    docstring='',
                )
                components.append(component)

        return components

    def extract_dependencies(self, source_code: str, tree: Tree) -> Set[str]:
        """Extract import dependencies."""
        dependencies = set()
        root_node = tree.root_node

        # Query for import statements
        import_queries = [
            '(import_statement (string) @import)',
            '(call_expression function: (identifier) @call arguments: (arguments (string) @import))',
        ]

        for query_str in import_queries:
            query = self._create_query(query_str)
            if query:
                captures = query.captures(root_node)
                for node, capture_name in captures:
                    import_name = node.text.decode('utf8').strip('"\'')
                    dependencies.add(import_name)

        return dependencies

    def _create_query(self, query_str: str):
        """Create a tree-sitter query."""
        try:
            return self.language.query(query_str)
        except Exception as e:
            logger.warning(f"Failed to create query '{query_str}': {e}")
            return None

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Get the text content of a node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code.encode('utf8')[start_byte:end_byte].decode('utf8')

    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        path = file_path
        for ext in ['.js', '.ts', '.jsx', '.tsx']:
            if path.endswith(ext):
                path = path[:-len(ext)]
                break
        return path.replace(os.path.sep, '.')


class TypeScriptParser(BaseLanguageParser):
    """Parser for TypeScript files."""

    def extract_components(self, source_code: str, file_path: str, relative_path: str) -> List[CodeComponent]:
        """Extract TypeScript components."""
        components = []
        tree = self.parser.parse(bytes(source_code, 'utf8'))
        root_node = tree.root_node

        module_path = self._file_to_module_path(relative_path)

        # Extract classes and functions
        components.extend(self._extract_classes(root_node, source_code, file_path, relative_path, module_path))
        components.extend(self._extract_functions(root_node, source_code, file_path, relative_path, module_path))

        return components

    def _extract_classes(self, root_node: Node, source_code: str, file_path: str, relative_path: str, module_path: str) -> List[CodeComponent]:
        """Extract class definitions."""
        components: List[CodeComponent] = []

        query_str = '(class_declaration (type_identifier) @class.name)'
        query = self._create_query(query_str)
        if not query:
            return components

        captures = query.captures(root_node)

        for node, capture_name in captures:
            if capture_name == 'class.name':
                class_node = node.parent
                class_name = node.text.decode('utf8')
                class_id = f'{module_path}.{class_name}'

                component = CodeComponent(
                    id=class_id,
                    node=None,
                    component_type='class',
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_node_text(class_node, source_code),
                    start_line=class_node.start_point[0] + 1,
                    end_line=class_node.end_point[0] + 1,
                    has_docstring=False,  # TS doesn't have built-in docstrings
                    docstring='',
                )
                components.append(component)

                # Extract methods
                components.extend(self._extract_methods(class_node, source_code, file_path, relative_path, class_id))

        return components

    def _extract_functions(self, root_node: Node, source_code: str, file_path: str, relative_path: str, module_path: str) -> List[CodeComponent]:
        """Extract function definitions."""
        components = []

        # Query for various function types
        function_queries = [
            '(function_declaration (identifier) @function.name)',
        ]

        for query_str in function_queries:
            query = self._create_query(query_str)
            if query:
                captures = query.captures(root_node)
                for node, capture_name in captures:
                    if capture_name == 'function.name':
                        func_node = node.parent
                        func_name = node.text.decode('utf8')
                        func_id = f'{module_path}.{func_name}'

                        component = CodeComponent(
                            id=func_id,
                            node=None,
                            component_type='function',
                            file_path=file_path,
                            relative_path=relative_path,
                            source_code=self._get_node_text(func_node, source_code),
                            start_line=func_node.start_point[0] + 1,
                            end_line=func_node.end_point[0] + 1,
                            has_docstring=False,
                            docstring='',
                        )
                        components.append(component)

        return components

    def _extract_methods(self, class_node: Node, source_code: str, file_path: str, relative_path: str, class_id: str) -> List[CodeComponent]:
        """Extract methods within a class."""
        components: List[CodeComponent] = []

        query_str = '(method_definition (property_identifier) @method.name)'
        query = self._create_query(query_str)
        if not query:
            return components

        captures = query.captures(class_node)

        for node, capture_name in captures:
            if capture_name == 'method.name':
                method_node = node.parent
                method_name = node.text.decode('utf8')
                method_id = f'{class_id}.{method_name}'

                component = CodeComponent(
                    id=method_id,
                    node=None,
                    component_type='method',
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_node_text(method_node, source_code),
                    start_line=method_node.start_point[0] + 1,
                    end_line=method_node.end_point[0] + 1,
                    has_docstring=False,
                    docstring='',
                )
                components.append(component)

        return components

    def extract_dependencies(self, source_code: str, tree: Tree) -> Set[str]:
        """Extract import dependencies."""
        dependencies = set()
        root_node = tree.root_node

        # Query for import statements
        import_queries = [
            '(import_statement (string) @import)',
            '(call_expression function: (identifier) @call arguments: (arguments (string) @import))',
        ]

        for query_str in import_queries:
            query = self._create_query(query_str)
            if query:
                captures = query.captures(root_node)
                for node, capture_name in captures:
                    import_name = node.text.decode('utf8').strip('"\'')
                    dependencies.add(import_name)

        return dependencies

    def _create_query(self, query_str: str):
        """Create a tree-sitter query."""
        try:
            return self.language.query(query_str)
        except Exception as e:
            logger.warning(f"Failed to create query '{query_str}': {e}")
            return None

    def _get_node_text(self, node: Node, source_code: str) -> str:
        """Get the text content of a node."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code.encode('utf8')[start_byte:end_byte].decode('utf8')

    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        path = file_path
        for ext in ['.js', '.ts', '.jsx', '.tsx']:
            if path.endswith(ext):
                path = path[:-len(ext)]
                break
        return path.replace(os.path.sep, '.')


class MultiLanguageDependencyParser:
    """
    Multi-language dependency parser using tree-sitter.
    Supports Python, JavaScript, TypeScript, Java, Go, C/C++ and more.
    """

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, CodeComponent] = {}
        self.modules: Set[str] = set()
        self.parsers: Dict[str, BaseLanguageParser] = {}

        if not TREE_SITTER_AVAILABLE:
            raise ImportError('tree-sitter is required. Install with: pip install tree-sitter')

        self._setup_parsers()

    def _setup_parsers(self):
        """Setup language parsers using tree-sitter directly."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning('Tree-sitter not available')
            return

        if not LANGUAGES_AVAILABLE:
            logger.warning('py-tree-sitter-languages not available')
            return

        try:
            # Setup parsers for different languages
            language_configs = {
                '.py': ('python', PythonParser),
                '.js': ('javascript', JavaScriptParser),
                '.ts': ('typescript', TypeScriptParser),
            }

            for ext, (lang_name, parser_class) in language_configs.items():
                logger.info(f'Configuring parser for {lang_name} ({ext})')
                try:
                    logger.info(f'Setting up {lang_name} parser...')
                    # Get language and create parser
                    language = get_language(lang_name)
                    parser = Parser()
                    parser.set_language(language)

                    # Create wrapper
                    parser_wrapper = parser_class.create_from_tree_sitter(lang_name, parser, language)
                    self.parsers[ext] = parser_wrapper

                    logger.info(f'✅ {lang_name} parser setup successful!')

                except Exception as e:
                    logger.error(f'❌ {lang_name} parser setup failed: {e}')
                    import traceback
                    logger.debug(f'   Traceback: {traceback.format_exc()}')

        except Exception as e:
            logger.error(f'❌ Parser setup failed completely: {e}')
            import traceback
            logger.debug(f'   Traceback: {traceback.format_exc()}')

    def parse_repository(self) -> Dict[str, CodeComponent]:
        """Parse all supported files in the repository."""
        logger.info(f'Parsing multi-language repository at {self.repo_path}')

        # Get all supported files
        supported_files = self._get_supported_files()
        logger.info(f'Found {len(supported_files)} supported files to parse')
        # Parse each file with appropriate parser
        for file_path in supported_files:
            try:
                self._parse_file(file_path)
            except Exception as e:
                logger.error(f'Error parsing {file_path}: {e}')

        # Resolve dependencies
        self._resolve_dependencies()

        logger.info(f'Found {len(self.components)} components in {len(supported_files)} files')
        return self.components

    def _get_supported_files(self) -> List[str]:
        """Get all files that can be parsed."""
        supported_files = []
        supported_extensions = set(self.parsers.keys())

        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)

                if ext in supported_extensions:
                    supported_files.append(file_path)

        return supported_files

    def _parse_file(self, file_path: str):
        """Parse a single file with the appropriate parser."""
        relative_path = os.path.relpath(file_path, self.repo_path)
        _, ext = os.path.splitext(file_path)

        if ext not in self.parsers:
            return

        try:
            with open(file_path, encoding='utf-8') as f:
                source_code = f.read()

            parser = self.parsers[ext]
            components = parser.extract_components(source_code, file_path, relative_path)

            # Add components to our collection
            for component in components:
                self.components[component.id] = component

                # Track module
                module_path = component.id.split('.')[0]
                self.modules.add(module_path)

        except Exception as e:
            logger.warning(f'Error parsing {file_path}: {e}')

    def _resolve_dependencies(self):
        """Resolve dependencies between components."""
        for component_id, component in self.components.items():
            try:
                file_path = component.file_path
                _, ext = os.path.splitext(file_path)

                if ext not in self.parsers:
                    continue

                with open(file_path, encoding='utf-8') as f:
                    source_code = f.read()

                parser = self.parsers[ext]
                tree = parser.parser.parse(bytes(source_code, 'utf8'))
                dependencies = parser.extract_dependencies(source_code, tree)

                # Filter dependencies to only include those in our component set
                valid_dependencies = set()
                for dep in dependencies:
                    # Try to find matching components
                    for comp_id in self.components:
                        if comp_id.startswith(dep) or dep in comp_id:
                            valid_dependencies.add(comp_id)

                component.depends_on.update(valid_dependencies)

            except Exception as e:
                logger.warning(f'Error resolving dependencies for {component_id}: {e}')

    def save_dependency_graph(self, output_path: str):
        """Save the dependency graph to a JSON file."""
        serializable_components = {
            comp_id: component.to_dict()
            for comp_id, component in self.components.items()
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_components, f, indent=2)

        logger.info(f'Saved multi-language dependency graph to {output_path}')

    def get_language_statistics(self) -> Dict[str, int]:
        """Get statistics about languages in the repository."""
        stats: Dict[str, int] = {}

        for component in self.components.values():
            _, ext = os.path.splitext(component.file_path)
            lang_config = LANGUAGE_CONFIG.get(ext, {})
            lang_name = lang_config.get('name', ext) if isinstance(lang_config, dict) else ext
            stats[lang_name] = stats.get(lang_name, 0) + 1

        return stats
