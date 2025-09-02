from __future__ import annotations

import base64

import requests
from domain.rag import GraphRAG
from shared.logging import get_logger
from shared.utils import is_github_repo

from .base import BaseOutlineGenerator
from .base import OutlineGeneratorInput

logger = get_logger(__name__)


class OutlineGenerator(BaseOutlineGenerator):
    def __init__(self):
        self.rag = None

    def prepare_rag(self, rag: GraphRAG):
        """Prepare the RAG instance for content generation."""
        self.rag = rag

    def generate(self, input_data: OutlineGeneratorInput) -> str:
        """Generate an outline based on the provided repository URL.

        Args:
            input_data (OutlineGeneratorInput): The input data for outline generation.

        Returns:
            str: The generated outline.
        """
        if not self.rag:
            logger.error('RAG instance is not prepared for streaming generation.')
            raise ValueError('RAG instance is not prepared.')
        repo_url = input_data.repo_url
        owner = input_data.owner
        repo_name = input_data.repo_name
        platform = 'github' if is_github_repo(repo_url) else 'gitlab'
        access_token = input_data.access_token

        file_tree_data = ''
        readme_content = ''

        for branch in ['main', 'master']:
            if platform == 'gitlab':
                api_url = f'https://gitlab.com/api/v4/projects/{owner}%2F{repo_name}/repository/tree?ref={branch}&recursive=true'
            else:
                api_url = f'https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1'
            headers = {}
            if access_token and platform == 'gitlab':
                headers['Private-Token'] = access_token
            elif access_token and platform == 'github':
                headers['Authorization'] = f'token {access_token}'

            try:
                response = requests.get(api_url, headers=headers, timeout=10)
                if response.ok:
                    tree_data = response.json()
                    break
                else:
                    error_data = response.text()
                    apiErrorDetails = f'Status: {response.status}, Response: {error_data}'
                    logger.error(f'Error fetching repository structure: {apiErrorDetails}')
            except Exception:
                raise Exception('Error fetching repository structure')

        if 'tree' not in tree_data:
            raise Exception('No tree data found in the repository structure response')

        file_tree_data = '\n'.join(
            item['path'] for item in tree_data['tree'] if item['type'] == 'blob'
        )

        if platform == 'gitlab':
            readme_url = f'https://gitlab.com/api/v4/projects/{owner}%2F{repo_name}/repository/files/README.md/raw?ref={branch}'
        else:
            readme_url = f'https://api.github.com/repos/{owner}/{repo_name}/readme'
        readme_response = requests.get(readme_url, headers=headers, timeout=10)

        logger.info(f'Fetching README.md from {readme_url}')
        try:
            if readme_response.ok:
                readme_data = readme_response.json()
                readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
            else:
                logger.warning(f'Could not fetch README.md, status: {readme_response.status_code}')
        except Exception as e:
            logger.warning(f'Could not fetch README.md, continuing with empty README: {str(e)}')

        question = rf"""Analyze this {platform} repository {owner}/{repo_name} and create a documentation structure for it.
        1. The complete file tree of the project:
        <file_tree>
        {file_tree_data}
        </file_tree>

        2. The README file of the project:
        <readme>
        {readme_content}
        </readme>

        I want to create a English documentation for this repository. Determine the most logical structure for a documentation based on the repository's content.

        When designing the documentation structure, include pages that would benefit from visual diagrams, such as:
        - Architecture overviews
        - Data flow descriptions
        - Component relationships
        - Process workflows
        - State machines
        - Class hierarchies (if applicable)

        Create a documentation structure with the following main sections:
        - Overview (general information about the project)
        - System Architecture (how the system is designed)
        - Core Features (key functionality)
        - Data Management/Flow: If applicable, how data is stored, processed, accessed, and managed (e.g., database schema, data pipelines, state management).
        - Frontend Components (UI elements, if applicable.)
        - Backend Systems (server-side components)
        - Model Integration (AI model connections)
        - Deployment/Infrastructure (how to deploy, what's the infrastructure like)
        - Extensibility and Customization: If the project architecture supports it, explain how to extend or customize its functionality (e.g., plugins, theming, custom modules, hooks).

        Each section should contain relevant pages. For example, the "Frontend Components" section might include pages for "Home Page", "Repository Page", "Ask Component", etc.

        Return your analysis in the following strict **XML format**:

        <documentation_structure>
            <title>[Overall title for the documentation]</title>
            <description>[Brief description of the repository]</description>
            <sections>
                <section id="section-1">
                    <title>[Section title]</title>
                    <pages>
                        <page_ref>page-1</page_ref>
                        <page_ref>page-2</page_ref>
                    </pages>
                    <subsections>
                        <section_ref>section-2</section_ref>
                    </subsections>
                </section>
                <!-- More sections as needed -->
            </sections>
            <pages>
                <page id="page-1">
                    <title>[Page title]</title>
                    <description>[Brief description of what this page will cover]</description>
                    <relevant_files>
                        <file_path>[Path to a relevant file]</file_path>
                        <!-- More file paths as needed -->
                    </relevant_files>
                </page>
                <!-- More pages as needed -->
            </pages>
        </documentation_structure>

        **CRITICAL XML FORMATTING REQUIREMENTS:**
        - Return ONLY valid, well-formed XML - no markdown code blocks, no explanations
        - Start directly with <documentation_structure> tag
        - End directly with </documentation_structure> tag
        - Ensure all opening tags have matching closing tags
        - Use proper XML escaping for special characters (&lt; &gt; &amp; &quot; &#39;)
        - Use consistent indentation (4 spaces per level)
        - All sections must have unique IDs (section-1, section-2, etc.)
        - All pages must have unique IDs (page-1, page-2, etc.)
        - All page_ref elements must reference existing page IDs
        - Include exactly one <sections> container and one <pages> container
        - Each section should have 2-4 pages maximum
        - Create 6-10 sections total with 15-25 pages total
        - File paths should be realistic and match the actual repository structure
        - Escape any special characters in titles and descriptions

        Remember: The XML must be parseable by standard XML parsers. Any syntax errors will cause processing to fail.

        """

        rag_res = self.rag.query(
            question=question,
            repo_url=repo_url,
            query_type='outline_generation',
        )

        return rag_res
