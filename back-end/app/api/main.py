from fastapi import APIRouter
from app.api.routers import healthcheck, repo_fetch

router = APIRouter()

router.include_router(healthcheck.router, tags=["Health Check"])
router.include_router(repo_fetch.router, tags=["Repository Fetching"])
