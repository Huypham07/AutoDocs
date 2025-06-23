import logging

import uvicorn
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.main import router
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url="/docs",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

app.include_router(router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    logger.info(f"Starting Streaming API on port {settings.PORT}")

    # enable reload in development environment
    is_development = settings.NODE_ENV == "development"
    
    if is_development:
        # Prevent infinite logging loop caused by file changes triggering log writes
        logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
        
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=settings.PORT,
        reload=is_development,
    )