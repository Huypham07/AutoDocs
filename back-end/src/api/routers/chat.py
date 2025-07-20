from __future__ import annotations

from api.models.chat import ChatRequest
from api.models.chat import ChatResponse
from application.chat import ChatApplication
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request

router = APIRouter()


def get_chat_application(request: Request):
    """Dependency to get the ChatApplication instance."""
    rag = request.app.state.chat_rag
    chat_application = ChatApplication(rag=rag)
    return chat_application


@router.post('/chat', response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, application: ChatApplication = Depends(get_chat_application)):
    """Chat endpoint for processing user messages.

    This endpoint processes the user's message and returns a response from the RAG service.

    Args:
        request (ChatRequest): The request containing the user's message and optional chat history.
        application (ChatApplication, optional): Injected ChatApplication instance..

    Returns:
        ChatResponse: The response containing the generated message and sources.
    """
    try:
        return await application.process(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail='Internal server error')
