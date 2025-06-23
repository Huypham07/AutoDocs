import logging

from fastapi import APIRouter, HTTPException
from app.core.logging import setup_logging
from app.schemas.repo import RepoFetchingRequest
from app.core.db_manager import DBManager
import hashlib

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/repo/fetch")
async def handle_repo_fetching(request: RepoFetchingRequest):
    """Handle repository fetching requests"""
    try:
        db_manager = DBManager()
        
        db_manager.prepare_db(
            repo_url=str(request.repo_url),
            access_token=request.access_token,
            is_ollama_embedding=True,
            excluded_dirs=request.excluded_dirs.split(",") if request.excluded_dirs else None,
            excluded_files=request.excluded_files.split(",") if request.excluded_files else None,
            included_dirs=request.included_dirs.split(",") if request.included_dirs else None,
            included_files=request.included_files.split(",") if request.included_files else None
        )
    
        return {
            "hash_id": hashlib.sha256(str(request.repo_url).encode()).hexdigest(), 
        }
    except HTTPException:
        raise
    except Exception as e_handler:
        error_msg = f"Error in streaming chat completion: {str(e_handler)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)