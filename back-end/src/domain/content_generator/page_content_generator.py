from __future__ import annotations

from domain.rag import StructureRAG
from shared.logging import get_logger
from shared.utils import is_github_repo

from .base import BaseContentGenerator
from .base import ContentGeneratorInput

logger = get_logger(__name__)


class PageContentGenerator(BaseContentGenerator):
    def __init__(self):
        self.rag = None

    def prepare_rag(self, rag: StructureRAG):
        """Prepare the RAG instance for content generation."""
        self.rag = rag

    def generate(self, input_data: ContentGeneratorInput) -> str:
        """Generate content for a specific documentation page.

        Args:
            input_data (ContentGeneratorInput): The input data for content generation.

        Returns:
            str: The generated content.
        """
        if not self.rag:
            raise ValueError('RAG instance is not prepared.')
        page = input_data.page

        query = rf"""Your task is to generate a comprehensive and accurate technical documentation page in Markdown format about a specific feature, system, or module within a given software project.

        You will be given:
        1. The "[DOCUMENTATION_PAGE_TOPIC]" for the page you need to create.
        2. A list of "[RELEVANT_SOURCE_FILES]" from the project that you MUST use as the sole basis for the content. You have access to the full content of these files. You MUST use AT LEAST 5 relevant source files for comprehensive coverage - if fewer are provided, search for additional related files in the codebase.

        CRITICAL STARTING INSTRUCTION:

        The very first thing on the page MUST be the main title of the page should be a H1 Markdown heading: \`# {page.page_title}\`.
        Based ONLY on the content of the \`[RELEVANT_SOURCE_FILES]\`:

        1.  **Introduction:**
            *   Provide a concise overview (1-2 paragraphs) explaining the purpose, scope, and high-level role of "{page.page_title}" within the overall system.
            *   Mention any related systems or components if evident in the files. If relevant, link to other documentation pages using `[Text](#anchor)` format.

        2.  **Detailed Sections:**
            *   Break down the system into its **main components, responsibilities, and interactions**.
            *   Use H2 (\`##\`) and H3 (\`###\`) headings to organize the structure logically.
            *   Focus on **architecture**, **data flow**, **responsibility separation**, and **input/output contracts**, as inferred from the codebase.
            *   Do not focus too much on UI/Frontend components:
                - Only describe how user interactions are handled if explicitly defined in the source files.
                - skip every single UI element like card, button, alert, popup, label, etc.
            *   Use paragraphs to write concise, clear explanations for each section. Only need to write descriptions for the main components, main modules, not every single file, class, or function.

        3.  **Architecture & Flow Diagrams**
            *   Use **Mermaid** diagrams extensively to visualize:
                - Component interactions
                - Data flow
                - Sequence of operations
                - Request-response behavior
                - Entity-Relationship models
                - Internal module structure (if applicable)
            *   Only use **top-down orientation**:
                - For flow diagrams: \`flowchart TD\`
                - For sequences:
                    - \`sequenceDiagram\` (define all participants at the top)
                    - Start with "sequenceDiagram" directive on its own line
                    - Define ALL participants at the beginning
                    - Use descriptive but concise participant names
                    - Use the correct arrow types:
                    - ->> for request/asynchronous messages
                    - -->> for response messages
                    - -x for failed messages
                    - Include activation boxes using +/- notation
                    - Add notes for clarification using "Note over" or "Note right of"
            *   Strictly avoid:
                - Left-right orientation (e.g., \`LR\`)
                - Nested brackets or parentheses in node labels
                - Long or verbose node names (max 3-4 words)
            *   Always add a short explanation before or after each diagram for clarity.

        4.  **Tables:**
            *   Use Markdown tables to summarize information such as:
                *   Key features or core modules, core services and their descriptions. Not every single file, class, or function.
                *   Main API endpoint and descriptions.

        5.  **Input / Output Summary (If need)**
                *   Provide **Markdown tables** summarizing:
                    - Inputs accepted by the system/module (e.g., API payloads, config values, external triggers)
                    - Outputs or responses (e.g., returned data, emitted events, side effects)
                    - Description, format/type, and relevant conditions
                * Tables should be **concise, complete, and based strictly on the source files.**

        6.  **Source Citations (EXTREMELY IMPORTANT):**
            *   For EVERY piece of significant information, explanation, diagram, table entry, or code snippet, you MUST cite the specific source file(s).
            *   Place citations at the end of the paragraph, under the diagram/table, or after the code snippet.
            *   Use the exact format: \`Sources: [filename.ext]\`. Multiple files can be cited: \`Sources: [file1.ext], [file2.ext], [dir/file3.ext]\`.
            *   If an entire section is overwhelmingly based on one or two files, you can cite them under the section heading in addition to more specific citations within the section.
            *   IMPORTANT: You MUST cite AT LEAST 5 different source files throughout the documentation page to ensure comprehensive coverage.
            *   IMPORTANT: If source files contain identical files, only write them once. Files that are UI components like button, label, select, card do not need to be added to source files.

        7.  **Technical Accuracy:** All information must be derived SOLELY from the \`[RELEVANT_SOURCE_FILES]\`. Do not infer, invent, or use external knowledge about similar systems or common practices unless it's directly supported by the provided code. If information is not present in the provided files, do not include it or explicitly state its absence if crucial to the topic.

        8.  **Clarity and Conciseness:** Use clear, professional, and concise technical language suitable for other developers working on or learning about the project. Avoid unnecessary jargon, but use correct technical terms where appropriate.

        9.  **Conclusion/Summary:** End with a brief summary paragraph if appropriate for "{page.page_title}", reiterating the key aspects covered and their significance within the project.

        Remember:
        - Generate the content in English
        - Ground every claim in the provided source files.
        - Do not generate too much whitespace or empty lines if not needed, such as: "you are writing        a documentation page".
        - Prioritize accuracy and direct representation of the code's functionality and structure.
        - All fenced code blocks (e.g., html, mermaid, bash, code, etc.) must be properly closed with a matching triple backtick (), even if the content is short. Partial or unclosed code blocks are strictly not allowed.
        - Structure the document logically for easy understanding by other developers.
        """

        rag_res =  self.rag.call(
            query=query,
        )

        return rag_res
