from __future__ import annotations

from domain.rag import StructureRAG
from shared.logging import get_logger
from shared.utils import is_github_repo

from .base import BaseContentGenerator
from .base import ContentGeneratorInput
from .base import ContentGeneratorOutput

logger = get_logger(__name__)


class PageContentGenerator(BaseContentGenerator):
    def __init__(self):
        self.rag = None

    def prepare_rag(self, rag: StructureRAG):
        """Prepare the RAG instance for content generation."""
        self.rag = rag

    def generate(self, input_data: ContentGeneratorInput) -> ContentGeneratorOutput:
        """Generate content for a specific documentation page.

        Args:
            input_data (ContentGeneratorInput): The input data for content generation.

        Returns:
            ContentGeneratorOutput: The generated content.
        """
        if not self.rag:
            raise ValueError('RAG instance is not prepared.')
        return ContentGeneratorOutput(content='This method is not implemented for synchronous generation.')

    async def generate_stream(self, input_data: ContentGeneratorInput):
        """Generate content for a specific documentation page.

        Args:
            input_data (ContentGeneratorInput): The input data for content generation.

        Returns:
            AsyncIterator[str]: An asynchronous iterator over the generated content.
        """
        if not self.rag:
            raise ValueError('RAG instance is not prepared.')
        page = input_data.page
        repo_url = input_data.repo_url
        repo_name = input_data.repo_name
        platform = 'github' if is_github_repo(repo_url) else 'gitlab'

        file_paths_str = '\n'.join([f'- [{path}]({path})' for path in page.file_paths])
        query = rf"""You are an expert technical writer and software architect.
        Your task is to generate a comprehensive and accurate technical documentation page in Markdown format about a specific feature, system, or module within a given software project.

        You will be given:
        1. The "[DOCUMENTATION_PAGE_TOPIC]" for the page you need to create.
        2. A list of "[RELEVANT_SOURCE_FILES]" from the project that you MUST use as the sole basis for the content. You have access to the full content of these files. You MUST use AT LEAST 5 relevant source files for comprehensive coverage - if fewer are provided, search for additional related files in the codebase.

        CRITICAL STARTING INSTRUCTION:
        The very first thing on the page MUST be a \`<details>\` block listing ALL the \`[RELEVANT_SOURCE_FILES]\` you used to generate the content. There MUST be AT LEAST 5 source files listed - if fewer were provided, you MUST find additional related files to include.
        Format it exactly like this:
        <details>
        <summary>Relevant source files</summary>

        Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the \`<details>\` block. JUST START with the \`<details>\` block.
        The following files were used as context for generating this documentation page:

        {file_paths_str}
        <!-- Add additional relevant files if fewer than 5 were provided -->
        </details>

        Immediately after the \`<details>\` block, the main title of the page should be a H1 Markdown heading: \`# {page.page_title}\`.

        Based ONLY on the content of the \`[RELEVANT_SOURCE_FILES]\`:

        1.  **Introduction:** Start with a concise introduction (1-2 paragraphs) explaining the purpose, scope, and high-level overview of "{page.page_title}" within the context of the overall project. If relevant, and if information is available in the provided files, link to other potential documentation pages using the format \`[Link Text](#page-anchor-or-id)\`.

        2.  **Detailed Sections:** Break down "{page.page_title}" into logical sections using H2 (\`##\`) and H3 (\`###\`) Markdown headings. For each section:
            *   Explain the architecture, components, data flow, or logic relevant to the section's focus, as evidenced in the source files.
            *   Identify key functions, classes, data structures, API endpoints, or configuration elements pertinent to that section.

        3.  **Mermaid Diagrams:**
            *   EXTENSIVELY use Mermaid diagrams (e.g., \`flowchart TD\`, \`sequenceDiagram\`, \`classDiagram\`, \`erDiagram\`, \`graph TD\`) to visually represent architectures, flows, relationships, and schemas found in the source files.
            *   Ensure diagrams are accurate and directly derived from information in the \`[RELEVANT_SOURCE_FILES]\`.
            *   Provide a brief explanation before or after each diagram to give context.
            *   CRITICAL: All diagrams MUST follow strict vertical orientation:
            - Use "graph TD" (top-down) directive for flow diagrams
            - NEVER use "graph LR" (left-right)
            - Maximum node width should be 3-4 words
            - For sequence diagrams:
                - Start with "sequenceDiagram" directive on its own line
                - Define ALL participants at the beginning
                - Use descriptive but concise participant names
                - Use the correct arrow types:
                - ->> for request/asynchronous messages
                - -->> for response messages
                - -x for failed messages
                - Include activation boxes using +/- notation
                - Add notes for clarification using "Note over" or "Note right of"

        4.  **Tables:**
            *   Use Markdown tables to summarize information such as:
                *   Key features or components and their descriptions.
                *   API endpoint parameters, types, and descriptions.
                *   Configuration options, their types, and default values.
                *   Data model fields, types, constraints, and descriptions.

        5.  **Code Snippets:**
            *   Include short, relevant code snippets (e.g., Python, Java, JavaScript, SQL, JSON, YAML) directly from the \`[RELEVANT_SOURCE_FILES]\` to illustrate key implementation details, data structures, or configurations.
            *   Ensure snippets are well-formatted within Markdown code blocks with appropriate language identifiers.

        6.  **Source Citations (EXTREMELY IMPORTANT):**
            *   For EVERY piece of significant information, explanation, diagram, table entry, or code snippet, you MUST cite the specific source file(s) and relevant line numbers from which the information was derived.
            *   Place citations at the end of the paragraph, under the diagram/table, or after the code snippet.
            *   Use the exact format: \`Sources: [filename.ext:start_line-end_line]()\` for a range, or \`Sources: [filename.ext:line_number]()\` for a single line. Multiple files can be cited: \`Sources: [file1.ext:1-10](), [file2.ext:5](), [dir/file3.ext]()\` (if the whole file is relevant and line numbers are not applicable or too broad).
            *   If an entire section is overwhelmingly based on one or two files, you can cite them under the section heading in addition to more specific citations within the section.
            *   IMPORTANT: You MUST cite AT LEAST 5 different source files throughout the documentation page to ensure comprehensive coverage.

        7.  **Technical Accuracy:** All information must be derived SOLELY from the \`[RELEVANT_SOURCE_FILES]\`. Do not infer, invent, or use external knowledge about similar systems or common practices unless it's directly supported by the provided code. If information is not present in the provided files, do not include it or explicitly state its absence if crucial to the topic.

        8.  **Clarity and Conciseness:** Use clear, professional, and concise technical language suitable for other developers working on or learning about the project. Avoid unnecessary jargon, but use correct technical terms where appropriate.

        9.  **Conclusion/Summary:** End with a brief summary paragraph if appropriate for "{page.page_title}", reiterating the key aspects covered and their significance within the project.

        Remember:
        - Generate the content in English
        - Ground every claim in the provided source files.
        - Prioritize accuracy and direct representation of the code's functionality and structure.
        - Structure the document logically for easy understanding by other developers.
        """

        system_prompt = rf"""/no_think

        <role>
        You are an expert code analyst examining the {platform} repository: {repo_url} ({repo_name}).
        You provide direct, concise, and accurate information about code repositories.
        You NEVER start responses with markdown headers or code fences.
        IMPORTANT:You MUST respond in English.
        </role>

        <guidelines>
        - Answer the user's question directly without ANY preamble or filler phrases
        - DO NOT include any rationale, explanation, or extra comments.
        - DO NOT start with preambles like "Okay, here's a breakdown" or "Here's an explanation"
        - DO NOT start with markdown headers like "## Analysis of..." or any file path references
        - DO NOT start with ```markdown code fences
        - DO NOT end your response with ``` closing fences
        - DO NOT start by repeating or acknowledging the question
        - JUST START with the direct answer to the question

        <example_of_what_not_to_do>
        ```markdown
        ## Analysis of `adalflow/adalflow/datasets/gsm8k.py`

        This file contains...
        ```
        </example_of_what_not_to_do>

        - Format your response with proper markdown including headings, lists, and code blocks WITHIN your answer
        - For code analysis, organize your response with clear sections
        - Think step by step and structure your answer logically
        - Start with the most relevant information that directly addresses the user's query
        - Be precise and technical when discussing code
        - Your response language should be in the same language as the user's query
        </guidelines>

        <style>
        - Use concise, direct language
        - Prioritize accuracy over verbosity
        - When showing code, include line numbers and file paths when relevant
        - Use markdown formatting to improve readability
        </style>
        """

        async for response in self.rag.acall(query=query, system_prompt=system_prompt):
            yield response
