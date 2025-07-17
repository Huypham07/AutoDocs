import logging

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.core.config import *
from app.api.api_router import router
from app.core.logging import setup_logging
from app.db.database import connect_to_mongo, close_mongo_connection
from contextlib import asynccontextmanager

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()
    
app = FastAPI(
    title=PROJECT_NAME,
    docs_url="/docs",
    openapi_url=f"{API_PREFIX}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)


app.include_router(router, prefix=API_PREFIX)

@app.get("/")
async def root():
    """Root endpoint configuration"""
    # Collect all routes dynamically
    endpoints = {}
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            # Skip docs and static routes
            if route.path in ["/openapi.json", "/docs", "/redoc", "/favicon.ico"]:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip("/").split("/")
            group = path_parts[0].capitalize() if path_parts[0] else "Root"
            method_list = list(route.methods - {"HEAD", "OPTIONS"})
            for method in method_list:
                endpoints.setdefault(group, []).append(f"{method} {route.path}")

    for group in endpoints:
        endpoints[group].sort()

    return {
        "message": "Welcome to Streaming API",
        "version": "1.0.0",
        "endpoints": endpoints
    }
    
if __name__ == "__main__":
    logger.info(f"Starting Streaming API on port {PORT}")

    # enable reload in development environment
    is_development = NODE_ENV == "development"
    
    if is_development:
        logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
        
    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=is_development,
    )