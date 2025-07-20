from __future__ import annotations

from api.routers import chat
from api.routers import docs_gen
from api.routers import documentation
from api.routers import healthcheck
from fastapi import APIRouter

routers = APIRouter()

routers.include_router(healthcheck.router, tags=['Health Check'])
routers.include_router(docs_gen.router, tags=['Docs Generation'])
routers.include_router(documentation.router, tags=['Documentation Management'])
routers.include_router(chat.router, tags=['Chat'])
