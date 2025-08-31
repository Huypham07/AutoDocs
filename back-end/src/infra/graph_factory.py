from __future__ import annotations

from infra.neo4j.graph_population_service import GraphPopulationService
from infra.neo4j.graph_repository import GraphRepository
from shared.logging import get_logger
from shared.settings.graph_config import graph_config

logger = get_logger(__name__)


class GraphRepositoryFactory:
    """Factory for creating graph repository instances."""

    @staticmethod
    def create_repository() -> GraphRepository:
        """Create appropriate graph repository based on configuration."""
        logger.info('Creating Neo4j graph repository')

        neo4j_config = graph_config.get_neo4j_config()
        return GraphRepository(
            uri=neo4j_config['uri'],
            username=neo4j_config['username'],
            password=neo4j_config['password'],
            database=neo4j_config['database'],
        )

    @staticmethod
    def create_population_service():
        """Create graph population service."""
        repository = GraphRepositoryFactory.create_repository()
        return GraphPopulationService(repository)

    @staticmethod
    def create_langchain_rag():
        """Create LangChain Graph RAG system."""
        repository = GraphRepositoryFactory.create_repository()
        from domain.rag import GraphRAG
        return GraphRAG(graph_repo=repository)
