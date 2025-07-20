from __future__ import annotations

from api.models.docs_gen import DocsGenRequest
from application.documentation import DocumentationApplication
from application.documentation import GeneratorInput
from domain.preparator import PreparatorInput
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.responses import StreamingResponse

router = APIRouter()


def get_documentation_application(request: Request):
    """Dependency to get the DocumentationApplication instance."""
    documentation_application = DocumentationApplication(
        rag=request.app.state.structure_rag,
        local_db_preparator=request.app.state.local_db_preparator,
        outline_generator=request.app.state.outline_generator,
        page_content_generator=request.app.state.page_content_generator,
        documentation_repository=request.app.state.documentation_repository,
    )
    return documentation_application


@router.post('/generate/docs')
async def create_docs(request: DocsGenRequest, application: DocumentationApplication = Depends(get_documentation_application)):
    """Generate documentation for the given repository URL."""

    try:
        exist_doc = await application.get_documentation_by_repo_url(str(request.repo_url))

        if exist_doc:
            return 'Documentation already exists for this repository.'

        application.prepare(
            PreparatorInput(
                repo_url=str(request.repo_url),
                access_token=request.access_token,
            ),
        )

        # await application.generate(GeneratorInput(
        #     repo_url=str(request.repo_url),
        #     access_token=request.access_token,
        # ))

        # return {
        #     'message': 'Documentation generation started successfully.',
        #     'status': 'success',
        # }

        return StreamingResponse(
            application.generate_stream(
                GeneratorInput(
                    repo_url=str(request.repo_url),
                    access_token=request.access_token,
                ),
            ), media_type='text/event-stream',
        )
    except Exception:
        raise
