from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from api.routers.api_routers import routers
from domain.content_generator import PageContentGenerator
from domain.outline_generator import OutlineGenerator
from domain.preparator import LocalDBPreparator
from domain.rag import ChatRAG
from domain.rag import StructureRAG
from fastapi import FastAPI
from infra.mongo.core import close_mongo_connection
from infra.mongo.core import connect_to_mongo
from infra.mongo.documentation_repository import DocumentationRepository
from infra.rabbitmq.publisher import RabbitMQPublisher
from shared.logging import get_logger
from shared.logging import setup_logging
from shared.utils import get_settings
from starlette.middleware.cors import CORSMiddleware

setup_logging(json_logs=True)
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for the FastAPI application."""
    logger.info('Initializing domain services...')
    await connect_to_mongo()

    app.state.structure_rag = StructureRAG(provider='google', model='gemini-2.5-flash-lite-preview-06-17')
    app.state.chat_rag = ChatRAG(provider='google', model='gemini-2.5-flash-lite-preview-06-17')
    app.state.local_db_preparator = LocalDBPreparator()
    app.state.outline_generator = OutlineGenerator()
    app.state.page_content_generator = PageContentGenerator()
    app.state.documentation_repository = DocumentationRepository()
    app.state.rabbitmq_publisher = RabbitMQPublisher()
    await app.state.rabbitmq_publisher.connect()

    logger.info('Domain services initialized successfully')
    yield
    # Shutdown
    await close_mongo_connection()
    if hasattr(app.state, 'rabbitmq_publisher'):
        await app.state.rabbitmq_publisher.close()

app = FastAPI(
    title=settings.PROJECT_NAME,
    docs_url='/docs',
    openapi_url=f'{settings.API_PREFIX}/openapi.json',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


app.include_router(routers, prefix=settings.API_PREFIX)


@app.get('/')
async def root():
    """Root endpoint configuration"""
    # Collect all routes dynamically
    endpoints = {}
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            # Skip docs and static routes
            if route.path in ['/openapi.json', '/docs', '/redoc', '/favicon.ico']:
                continue
            # Group endpoints by first path segment
            path_parts = route.path.strip('/').split('/')
            group = path_parts[0].capitalize() if path_parts[0] else 'Root'
            method_list = list(route.methods - {'HEAD', 'OPTIONS'})
            for method in method_list:
                endpoints.setdefault(group, []).append(f'{method} {route.path}')

    for group in endpoints:
        endpoints[group].sort()

    return {
        'message': 'Welcome to Streaming API',
        'version': '1.0.0',
        'endpoints': endpoints,
    }

if __name__ == '__main__':
    logger.info(f'Starting Streaming API on port {settings.PORT}')

    # enable reload in development environment
    is_development = settings.NODE_ENV == 'development'

    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=settings.PORT,
        reload=is_development,
    )
