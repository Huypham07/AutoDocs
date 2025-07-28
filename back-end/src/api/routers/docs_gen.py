from __future__ import annotations

from api.models.docs_gen import DocsGenRequest
from api.models.docs_gen import DocsGenResponse
from application.documentation import DocumentationApplication
from application.schemas import GeneratorInput
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.exceptions import HTTPException

router = APIRouter()


def get_documentation_application(request: Request):
    """Dependency to get the DocumentationApplication instance."""
    documentation_application = DocumentationApplication(
        rag=request.app.state.structure_rag,
        local_db_preparator=request.app.state.local_db_preparator,
        outline_generator=request.app.state.outline_generator,
        page_content_generator=request.app.state.page_content_generator,
        documentation_repository=request.app.state.documentation_repository,
        rabbitmq_publisher=request.app.state.rabbitmq_publisher,
    )
    return documentation_application


@router.post('/generate/docs', response_model=DocsGenResponse)
async def create_docs(request: DocsGenRequest, application: DocumentationApplication = Depends(get_documentation_application)):
    """Generate documentation for the given repository URL."""

    try:
        response = await application.process(
            GeneratorInput(
                repo_url=str(request.repo_url),
                access_token=request.access_token,
            ),
        )

        if response is None or response.status == 'error':
            raise HTTPException(status_code=400, detail='Failed to generate documentation')

        # Return message for existing documentation
        if response.is_existing:
            if response.status == 'processing':
                return DocsGenResponse(
                    status=response.status,
                    message='Documentation structure created. Page content generation has been queued for processing.',
                    is_existing=response.is_existing,
                    processing_time=response.processing_time,
                    document_id=response.document_id,
                )
            return DocsGenResponse(
                status=response.status,
                message='Documentation already exists for this repository.',
                is_existing=response.is_existing,
                processing_time=response.processing_time,
                document_id=response.document_id,
            )

        # Return message for processing documentation
        if response.status == 'processing':
            return DocsGenResponse(
                status=response.status,
                message='Documentation structure created. Page content generation has been queued for processing.',
                is_existing=response.is_existing,
                processing_time=response.processing_time,
                document_id=response.document_id,
            )

        return DocsGenResponse(
            status=response.status,
            message='Documentation processed successfully.',
            is_existing=response.is_existing,
            processing_time=response.processing_time,
            document_id=response.document_id,
        )

    except Exception:
        raise
