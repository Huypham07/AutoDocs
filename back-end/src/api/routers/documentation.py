from __future__ import annotations

from application.documentation import DocumentationApplication
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request

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


@router.get('/docs/{doc_id}')
async def get_documentation(doc_id: str, application: DocumentationApplication = Depends(get_documentation_application)):
    try:
        result = await application.get_documentation_by_id(doc_id)

        if not result:
            raise HTTPException(status_code=404, detail='Documentation not found')

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/docs/{owner}/{repo_name}')
async def get_documentation_by_repo(owner: str, repo_name: str, application: DocumentationApplication = Depends(get_documentation_application)):
    try:
        result = await application.get_documentation_by_repo_info(owner=owner, repo_name=repo_name)

        if not result:
            raise HTTPException(status_code=404, detail='Documentation not found for this repository')

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/docs/{doc_id}')
async def delete_documentation(doc_id: str, application: DocumentationApplication = Depends(get_documentation_application)):
    try:
        success = await application.delete_documentation(doc_id)

        if not success:
            raise HTTPException(status_code=404, detail='Documentation not found')

        return {'message': 'Documentation deleted successfully'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
