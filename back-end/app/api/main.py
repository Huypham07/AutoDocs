from fastapi import APIRouter
from app.api.routers import healthcheck, websocket, http_stream, repo_fetch

router = APIRouter()

router.include_router(healthcheck.router, tags=["Health Check"])
router.include_router(repo_fetch.router, tags=["Repository Fetching"])
router.include_router(websocket.router, tags=["WebSocket"])
router.include_router(http_stream.router, tags=["HTTP Stream"])