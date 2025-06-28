from fastapi import APIRouter
from app.api.routers import docs_gen, docs_gen_websocket, healthcheck

router = APIRouter()

router.include_router(healthcheck.router, tags=["Health Check"])
router.include_router(docs_gen.router, tags=["Docs Generation"])
router.include_router(docs_gen_websocket.router, tags=["Docs Generation WebSocket"])