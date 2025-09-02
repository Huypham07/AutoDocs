from __future__ import annotations

import asyncio
import signal
import sys

from application.documentation import DocumentationApplication
from domain.content_generator import PageContentGenerator
from domain.outline_generator import OutlineGenerator
from domain.preparator import PipelineConfig
from domain.preparator import PipelinePreparator
from infra.graph_factory import GraphRepositoryFactory
from infra.mongo.core import close_mongo_connection
from infra.mongo.core import connect_to_mongo
from infra.mongo.documentation_repository import DocumentationRepository
from infra.rabbitmq.consumer import RabbitMQConsumer
from shared.logging import get_logger
from shared.logging import setup_logging
from shared.utils import get_settings

setup_logging(json_logs=True)
logger = get_logger(__name__)
settings = get_settings()


class DocumentationWorker:
    def __init__(self):
        self.consumer = None
        self.documentation_app = None
        self.running = False

    async def setup(self):
        """Setup all dependencies for the worker."""
        try:
            logger.info('Setting up documentation worker...')
            await connect_to_mongo()

            # Initialize services using factory pattern (like in main.py)
            rag = GraphRepositoryFactory.create_langchain_rag()
            graph_population_service = GraphRepositoryFactory.create_population_service()

            documentation_repository = DocumentationRepository()

            # Create architecture pipeline preparator with configuration
            pipeline_config = PipelineConfig(
                target_clusters=10,
                perform_validation=True,
                save_intermediate_results=False,  # Don't save intermediate files in worker
                output_directory='worker_pipeline_output',
            )
            pipeline_preparator = PipelinePreparator(pipeline_config)

            outline_generator = OutlineGenerator()
            page_content_generator = PageContentGenerator()

            # Create DocumentationApplication (without rabbitmq_publisher for worker)
            self.documentation_app = DocumentationApplication(
                rag=rag,
                pipeline_preparator=pipeline_preparator,
                outline_generator=outline_generator,
                page_content_generator=page_content_generator,
                documentation_repository=documentation_repository,
                graph_population_service=graph_population_service,
                rabbitmq_publisher=None,
            )

            # Create consumer
            self.consumer = RabbitMQConsumer(documentation_app=self.documentation_app)

            await self.consumer.connect()
            logger.info('Worker setup completed successfully')

        except Exception as e:
            logger.error(f'Failed to setup worker: {str(e)}')
            raise

    async def start(self):
        """Start the worker."""
        try:
            self.running = True
            logger.info('Starting documentation worker...')

            # Start consuming messages from RabbitMQ
            await self.consumer.start_consuming()

            # Keep the worker running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f'Worker error: {str(e)}')
            raise

    async def stop(self):
        """Stop the worker gracefully."""
        logger.info('Stopping documentation worker...')
        self.running = False

        if self.consumer:
            await self.consumer.close()

        await close_mongo_connection()
        logger.info('Documentation worker stopped')

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f'Received signal {signum}, shutting down...')
        asyncio.create_task(self.stop())


async def main():
    """Main function to run the worker."""
    worker = DocumentationWorker()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, worker.signal_handler)
    signal.signal(signal.SIGTERM, worker.signal_handler)

    try:
        await worker.setup()
        await worker.start()
    except KeyboardInterrupt:
        logger.info('Received keyboard interrupt')
    except Exception as e:
        logger.error(f'Worker failed: {str(e)}')
        sys.exit(1)
    finally:
        await worker.stop()


if __name__ == '__main__':
    logger.info('Starting documentation worker process...')
    asyncio.run(main())
