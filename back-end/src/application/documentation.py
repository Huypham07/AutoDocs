from __future__ import annotations

import re
from datetime import datetime
from datetime import timezone
from typing import Optional

from api.models.docs_gen import get_pages_from_structure
from api.models.docs_gen import Page
from api.models.docs_gen import parse_structure_from_xml
from api.models.docs_gen import Structure
from domain.content_generator import ContentGeneratorInput
from domain.content_generator import PageContentGenerator
from domain.outline_generator import OutlineGenerator
from domain.outline_generator import OutlineGeneratorInput
from domain.preparator import LocalDBPreparator
from domain.preparator import PreparatorInput
from domain.rag import StructureRAG
from infra.mongo.documentation_repository import DocumentationRepository
from infra.rabbitmq.publisher import RabbitMQPublisher
from shared.logging import get_logger
from shared.utils import extract_repo_info

from .schemas import GeneratorInput
from .schemas import GeneratorOutput
from .schemas import PageProcessingTask

logger = get_logger(__name__)


class DocumentationApplication:
    def __init__(
        self,
        rag: StructureRAG,
        local_db_preparator: LocalDBPreparator,
        outline_generator: OutlineGenerator,
        page_content_generator: PageContentGenerator,
        documentation_repository: DocumentationRepository,
        rabbitmq_publisher: RabbitMQPublisher,
    ):
        self.rag = rag
        self.local_db_preparator = local_db_preparator
        self.outline_generator = outline_generator
        self.page_content_generator = page_content_generator
        self.documentation_repository = documentation_repository
        self.rabbitmq_publisher = rabbitmq_publisher

    def prepare(self, preparator_input: PreparatorInput):
        """Prepare the application for documentation generation."""
        logger.info('Preparing for documentation generation...')
        transformed_docs = self.local_db_preparator.prepare(preparator_input)
        self.rag.prepare_retriever(transformed_docs)

        self.outline_generator.prepare_rag(self.rag)
        self.page_content_generator.prepare_rag(self.rag)
        logger.info('Preparation complete for documentation generation.')

    async def process(self, generator_input: GeneratorInput) -> GeneratorOutput:
        """Generate full documentation for the given repository URL."""
        time_start = datetime.now(timezone.utc)
        repo_url = generator_input.repo_url
        access_token = generator_input.access_token
        owner, repo_name = extract_repo_info(generator_input.repo_url)

        try:
            logger.info(f'Checking for existing documentation for {owner}/{repo_name}...')
            existing_doc = await self.get_documentation_by_repo_info(owner=owner, repo_name=repo_name)
            if existing_doc:
                logger.info(f'Documentation already exists for {owner}/{repo_name}.')
                processing_time = (datetime.now(timezone.utc) - time_start).total_seconds()
                existing_doc = Structure(**existing_doc)  # Convert to Structure object
                return GeneratorOutput(
                    status=existing_doc.status,
                    is_existing=True,
                    processing_time=processing_time,
                )

            self.prepare(
                PreparatorInput(
                    repo_url=repo_url,
                    access_token=access_token,
                ),
            )

            logger.info('Generating documentation...')

            outline = self.outline_generator.generate(
                OutlineGeneratorInput(
                    repo_url=repo_url,
                    access_token=access_token,
                    owner=owner,
                    repo_name=repo_name,
                ),
            )

            xml_match = re.search(r'<documentation_structure>.*?</documentation_structure>', outline, re.DOTALL)
            if not xml_match:
                logger.error('No valid documentation structure found in the outline.')
                raise ValueError('No valid documentation structure found in the outline.')

            outline_xml = xml_match.group(0)

            structure = parse_structure_from_xml(outline_xml)
            pages = get_pages_from_structure(structure)
            logger.info(f'Found {len(pages)} pages in the structure')

            # Save structure to database with status 'processing'
            saved_id = await self.documentation_repository.save_documentation(
                title=structure.title,
                description=structure.description,
                root_sections=structure.root_sections,
                repo_url=repo_url,
                owner=owner,
                repo_name=repo_name,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                status='processing',  # Status indicates content is being generated
            )

            if not saved_id:
                logger.error('Failed to save documentation structure to MongoDB')
                raise Exception('Failed to save documentation structure')

            logger.info(f'Documentation structure saved with ID: {saved_id}')

            # Schedule page content generation tasks
            await self._schedule_page_processing_tasks(
                document_id=saved_id,
                pages=pages,
                repo_url=repo_url,
                owner=owner,
                repo_name=repo_name,
                access_token=access_token,
            )

            processing_time = (datetime.now(timezone.utc) - time_start).total_seconds()
            return GeneratorOutput(
                status='processing',  # Status indicates processing is processing
                is_existing=False,
                processing_time=processing_time,
                document_id=saved_id,
            )

        except Exception as e:
            logger.error(f'Error generating documentation: {str(e)}')
            raise

    async def _schedule_page_processing_tasks(
        self,
        document_id: str,
        pages: list[Page],
        repo_url: str,
        owner: str,
        repo_name: str,
        access_token: Optional[str],
    ):
        """Schedule individual page processing tasks to RabbitMQ."""
        for page in pages:
            task = PageProcessingTask(
                document_id=document_id,
                repo_url=repo_url,
                owner=owner,
                repo_name=repo_name,
                access_token=access_token,
                page=page,
            )

            await self.rabbitmq_publisher.publish_page_task(task)
            logger.info(f'Scheduled page processing task for: {page.page_title}')

    async def process_page_content(self, task: PageProcessingTask):
        """Process content for a single page (called by message consumer)."""
        try:
            logger.info(f'Processing page content for: {task.page.page_title}')

            # Prepare RAG if needed (might need to optimize this)
            self.prepare(
                PreparatorInput(
                    repo_url=task.repo_url,
                    access_token=task.access_token,
                ),
            )

            # Create a page object for content generation

            page = task.page

            # Generate page content
            page_content = self.page_content_generator.generate(
                ContentGeneratorInput(
                    repo_url=task.repo_url,
                    owner=task.owner,
                    repo_name=task.repo_name,
                    page=page,
                ),
            )

            # if page_content start with ```markdown\n, ```html, ```text, etc and ends with \n``` remove them
            if page_content.startswith('```') and page_content.endswith('\n```'):
                if page_content.startswith('```markdown\n'):
                    page_content = page_content[12:-4].strip()
                elif page_content.startswith('```html\n'):
                    page_content = page_content[9:-4].strip()

            # Update the page in database
            doc = await self.documentation_repository.get_documentation_by_id(task.document_id)
            if not doc:
                logger.error(f'Document not found: {task.document_id}')
                return
            doc = Structure(**doc)  # Convert to Structure object
            pages = get_pages_from_structure(doc)

            # Find the page to update
            for p in pages:
                if p.page_id == page.page_id:
                    p.content = page_content
                    break

            completed_pages = sum(1 for page in pages if page.content and page.content.strip())
            total_pages = len(pages)
            if completed_pages == total_pages:
                doc.status = 'completed'

            await self.documentation_repository.update_documentation(doc)

        except Exception as e:
            logger.error(f'Error processing page {task.page.page_title}: {str(e)}')

    # existing documentation methods

    async def get_documentation_by_repo_url(self, repo_url: str):
        """Get documentation by repository URL."""
        owner, repo_name = extract_repo_info(repo_url)
        return await self.documentation_repository.get_documentation_by_repo(owner=owner, repo_name=repo_name)

    async def get_documentation_by_repo_info(self, owner: str, repo_name: str):
        """Get documentation by repository owner and name."""
        return await self.documentation_repository.get_documentation_by_repo(owner=owner, repo_name=repo_name)

    async def delete_documentation(self, doc_id: str):
        """Delete documentation by ID."""
        return await self.documentation_repository.delete_documentation(doc_id=doc_id)
