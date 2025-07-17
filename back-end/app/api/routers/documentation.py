from fastapi import APIRouter, HTTPException
from app.services.docsgen.database_service import DocumentationService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/docs/{doc_id}")
async def get_documentation(doc_id: str):
    try:
        doc_service = DocumentationService()
        result = await doc_service.get_documentation_by_id(doc_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Documentation not found")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/docs/{owner}/{repo_name}")
async def get_documentation_by_repo(owner: str, repo_name: str):
    try:
        doc_service = DocumentationService()
        result = await doc_service.get_documentation_by_repo(owner=owner, repo_name=repo_name)
        
        if not result:
            raise HTTPException(status_code=404, detail="Documentation not found for this repository")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting documentation by repo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/docs/{doc_id}")
async def delete_documentation(doc_id: str):
    try:
        doc_service = DocumentationService()
        success = await doc_service.delete_documentation(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Documentation not found")
        
        return {"message": "Documentation deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/docs/{doc_id}/structure")
async def get_documentation_structure(doc_id: str):
    try:
        doc_service = DocumentationService()
        result = await doc_service.get_documentation_by_id(doc_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Documentation not found")
        
        return result.get("structure_data", {})
        
    except Exception as e:
        logger.error(f"Error getting documentation structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/docs/{owner}/{repo_name}/structure")
async def get_documentation_structure(owner: str, repo_name: str):
    try:
        doc_service = DocumentationService()
        result = await doc_service.get_documentation_by_repo(owner=owner, repo_name=repo_name)
        
        if not result:
            raise HTTPException(status_code=404, detail="Documentation not found")
        
        return result.get("structure_data", {})
        
    except Exception as e:
        logger.error(f"Error getting documentation structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))