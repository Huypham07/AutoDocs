import logging
from app.core.logging import setup_logging
from app.core.rag import RAG
from app.db.repo_db_manager import count_tokens
from app.core.config import get_generator_model_config
from app.services.docsgen.util import is_github_repo
from adalflow import OllamaClient
import google.generativeai as genai
from adalflow.core.types import ModelType
import requests
import base64


setup_logging()
logger = logging.getLogger(__name__)

class StructureGenerator:
    def __init__(self, rag: RAG, repo_url: str, owner: str, repo_name: str, access_token: str = None):
        self.rag = rag
        self.repo_url = repo_url
        self.access_token = access_token
        self.owner = owner
        self.repo_name = repo_name
        self.platform = "github" if is_github_repo(repo_url) else "gitlab"
        
    async def __call__(self):
        file_tree_data = ""
        readme_content = ""
            
        for branch in ["main", "master"]:
            if self.platform == "gitlab":
                api_url = f"https://gitlab.com/api/v4/projects/{self.owner}%2F{self.repo_name}/repository/tree?ref={branch}&recursive=true"
            else:
                api_url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/git/trees/{branch}?recursive=1"
            headers = {}
            if self.access_token and self.platform == "gitlab":
                headers['Private-Token'] = self.access_token     
            elif self.access_token and self.platform == "github":
                headers['Authorization'] = f'token {self.access_token}'
            
            try:
                response = requests.get(api_url, headers=headers, timeout=10)
                if response.ok:
                    tree_data = response.json()
                    break
                else:
                    error_data = await response.text()
                    apiErrorDetails = f"Status: {response.status}, Response: {error_data}"
                    logger.error(f"Error fetching repository structure: {apiErrorDetails}")
            except Exception as e:
                raise Exception(f"Error fetching repository structure: {str(e)}")
            
        if 'tree' not in tree_data:
            raise Exception("No tree data found in the repository structure response")
        
        file_tree_data = '\n'.join(
            item['path'] for item in tree_data['tree'] if item['type'] == 'blob'
        )
        
        if self.platform == "gitlab":
            readme_url = f"https://gitlab.com/api/v4/projects/{self.owner}%2F{self.repo_name}/repository/files/README.md/raw?ref={branch}"
        else:
            readme_url = f"https://api.github.com/repos/{self.owner}/{self.repo_name}/readme"
        readme_response = requests.get(readme_url, headers=headers, timeout=10)
        
        try:
            if readme_response.ok:
                readme_data = readme_response.json()
                readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
            else:
                logging.warning(f"Could not fetch README.md, status: {readme_response.status_code}")
        except Exception as e:
            logger.warning(f"Could not fetch README.md, continuing with empty README: {str(e)}")
        
        query = f"""Analyze this {self.platform} repository {self.owner}/{self.repo_name} and create a documentation structure for it.
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
        - Class hierarchies

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

        Return your analysis in the following XML format:

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
                    <parent_section>section-1</parent_section>
                </page>
                <!-- More pages as needed -->
            </pages>
        </documentation_structure>

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Return ONLY the valid XML structure specified above
        - DO NOT wrap the XML in markdown code blocks (no \`\`\` or \`\`\`xml)
        - DO NOT include any explanation text before or after the XML
        - Ensure the XML is properly formatted and valid
        - Start directly with <documentation_structure> and end with </documentation_structure>

        IMPORTANT:
        1. Create 8-12 pages that would make a comprehensive documentation for this repository
        2. Each page should focus on a specific aspect of the codebase (e.g., architecture, key features, setup)
        3. The relevant_files should be actual files from the repository that would be used to generate that page
        4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters
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
        </style>
        """

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
