import logging
from app.core.logging import setup_logging
from app.core.rag import RAG
from app.db.repo_db_manager import count_tokens
from app.core.config import get_generator_model_config
from app.services.docsgen.util import is_github_repo
from adalflow import OllamaClient
import google.generativeai as genai
from adalflow.core.types import ModelType
from app.models.structure import parse_structure_from_xml, get_pages_from_structure


setup_logging()
logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self, rag: RAG, xml_string: str, repo_url: str, owner: str, repo_name: str, access_token: str = None):
        self.rag = rag
        self.xml_string = xml_string
        self.repo_url = repo_url
        self.owner = owner
        self.repo_name = repo_name
        self.platform = "github" if is_github_repo(repo_url) else "gitlab"
        
    async def __call__(self):
        structure = parse_structure_from_xml(self.xml_string)
        
        pages = get_pages_from_structure(structure)
        logger.info(f"Found {len(pages)} pages in the structure")
        
        for page in pages:
            file_paths_str = '\n'.join([f"- [{path}]({path})" for path in page.file_paths])
            query = f"""You are an expert technical writer and software architect.
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
        
            context_text = ""
            retrieved_documents = None
            
            provider = self.rag.provider
            model = self.rag.model
            
            input_too_large = False
            tokens = count_tokens(query,  provider == "ollama")
            logger.info(f"Request size: {tokens} tokens")
            if tokens > 8000:
                logger.warning(f"Request exceeds recommended token limit ({tokens} > 7500)")
                input_too_large = True
                
            # Only retrieve documents if input is not too large    
            if not input_too_large:
                try:
                    retrieved_documents = self.rag(query)
                    
                    if retrieved_documents and retrieved_documents[0].documents:
                        # Format context for the prompt in a more structured way
                        documents = retrieved_documents[0].documents
                        logger.info(f"Retrieved {len(documents)} documents")

                        # Group documents by file path
                        docs_by_file = {}
                        for doc in documents:
                            file_path = doc.meta_data.get('file_path', 'unknown')
                            if file_path not in docs_by_file:
                                docs_by_file[file_path] = []
                            docs_by_file[file_path].append(doc)

                        # Format context text with file path grouping
                        context_parts = []
                        for file_path, docs in docs_by_file.items():
                            # Add file header with metadata
                            header = f"## File Path: {file_path}\n\n"
                            # Add document content
                            content = "\n\n".join([doc.text for doc in docs])

                            context_parts.append(f"{header}{content}")

                        # Join all parts with clear separation
                        context_text = "\n\n" + "-" * 10 + "\n\n".join(context_parts)
                    else:
                        logger.warning("No documents retrieved from RAG")
                        
                except Exception as e:
                    logger.error(f"Error in RAG retrieval: {str(e)}")
                    # Continue without RAG if there's an error
                    
            system_prompt = f"""<role>
            You are an expert code analyst examining the {self.platform} repository: {self.repo_url} ({self.repo_name}).
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
            </style>"""

            prompt = f"/no_think {system_prompt}\n\n"
            
            # Only include context if it's not empty
            CONTEXT_START = "<START_OF_CONTEXT>"
            CONTEXT_END = "<END_OF_CONTEXT>"
            if context_text.strip():
                prompt += f"{CONTEXT_START}\n{context_text}\n{CONTEXT_END}\n\n"
            else:
                # Add a note that we're skipping RAG due to size constraints or because it's the isolated API
                logger.info("No context available from RAG")
                prompt += "<note>Answering without retrieval augmentation.</note>\n\n"
                
            prompt += f"<query>\n{query}\n</query>\n\nAssistant: "
            
            generator_model_config = get_generator_model_config(provider, model)["model_kwargs"]
            
            
            if provider == "ollama":
                prompt += " /no_think"

                model = OllamaClient()
                model_kwargs = {
                    "model": generator_model_config["model"],
                    "stream": True,
                    "options": {
                        "temperature": generator_model_config["temperature"],
                        "top_p": generator_model_config["top_p"],
                        "num_ctx": generator_model_config["num_ctx"]
                    }
                }

                api_kwargs = model.convert_inputs_to_api_kwargs(
                    input=prompt,
                    model_kwargs=model_kwargs,
                    model_type=ModelType.LLM
                )
            else:
                # Initialize Google Generative AI model
                model = genai.GenerativeModel(
                    model_name=generator_model_config["model"],
                    generation_config={
                        "temperature": generator_model_config["temperature"],
                        "top_p": generator_model_config["top_p"],
                        "top_k": generator_model_config["top_k"]
                    }
                )
                
            try:
                if provider == "ollama":
                    # Get the response and handle it properly using the previously created api_kwargs
                    response = await model.acall(api_kwargs=api_kwargs, model_type=ModelType.LLM)
                    # Handle streaming response from Ollama
                    async for chunk in response:
                        text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                        if text and not text.startswith('model=') and not text.startswith('created_at='):
                            text = text.replace('<think>', '').replace('</think>', '')
                            yield text
                else:
                    # Generate streaming response
                    response = model.generate_content(prompt, stream=True)
                    # Stream the response
                    for chunk in response:
                        if hasattr(chunk, 'text'):
                            yield chunk.text

            except Exception as e_outer:
                logger.error(f"Error in streaming response: {str(e_outer)}")
                error_message = str(e_outer)

                # Check for token limit errors
                if "maximum context length" in error_message or "token limit" in error_message or "too many tokens" in error_message:
                    # If we hit a token limit error, try again without context
                    logger.warning("Token limit exceeded, retrying without context")
                    try:
                        # Create a simplified prompt without context
                        simplified_prompt = f"/no_think {system_prompt}\n\n"
                        simplified_prompt += "<note>Answering without retrieval augmentation due to input size constraints.</note>\n\n"
                        simplified_prompt += f"<query>\n{query}\n</query>\n\nAssistant: "

                        if provider == "ollama":
                            simplified_prompt += " /no_think"

                            # Create new api_kwargs with the simplified prompt
                            fallback_api_kwargs = model.convert_inputs_to_api_kwargs(
                                input=simplified_prompt,
                                model_kwargs=model_kwargs,
                                model_type=ModelType.LLM
                            )

                            # Get the response using the simplified prompt
                            fallback_response = await model.acall(api_kwargs=fallback_api_kwargs, model_type=ModelType.LLM)

                            # Handle streaming fallback_response from Ollama
                            async for chunk in fallback_response:
                                text = getattr(chunk, 'response', None) or getattr(chunk, 'text', None) or str(chunk)
                                if text and not text.startswith('model=') and not text.startswith('created_at='):
                                    text = text.replace('<think>', '').replace('</think>', '')
                                    yield text
                        else:
                            # Initialize Google Generative AI model
                            fallback_model = genai.GenerativeModel(
                                model_name=generator_model_config["model"],
                                generation_config={
                                    "temperature": generator_model_config["model_kwargs"].get("temperature", 0.7),
                                    "top_p": generator_model_config["model_kwargs"].get("top_p", 0.8),
                                    "top_k": generator_model_config["model_kwargs"].get("top_k", 40)
                                }
                            )

                            # Get streaming response using simplified prompt
                            fallback_response = fallback_model.generate_content(simplified_prompt, stream=True)
                            # Stream the fallback response
                            for chunk in fallback_response:
                                if hasattr(chunk, 'text'):
                                    yield chunk.text
                    except Exception as e2:
                        logger.error(f"Error in fallback streaming response: {str(e2)}")
                        yield f"\nI apologize, but your request is too large for me to process. Please try a shorter query or break it into smaller parts."
                else:
                    # For other errors, return the error message
                    yield f"\nError: {error_message}"

        logger.info(structure.to_dict())