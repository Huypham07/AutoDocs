import logging

from fastapi import APIRouter, HTTPException
from app.core.logging import setup_logging
from app.schemas.chat_message import ChatCompletionRequest
from app.core.db_manager import DBManager

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/streaming/completion")
async def handle_streaming_completion(request: ChatCompletionRequest):
    """Handle streaming completion requests"""
    try:
        db_manager = DBManager()
        
        db_manager.prepare_db(
            repo_url=str(request.repo_url),
            access_token=request.access_token
        )
    
        return {
            "status": "success",
            "message": "Streaming completion started successfully."
        }
    except HTTPException:
        raise
    except Exception as e_handler:
        error_msg = f"Error in streaming chat completion: {str(e_handler)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)