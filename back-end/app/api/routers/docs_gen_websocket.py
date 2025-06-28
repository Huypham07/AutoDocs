import logging

from app.core.logging import setup_logging
from fastapi import APIRouter, WebSocket

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/completion")
async def handle_websocket_completion(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello")