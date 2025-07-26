from __future__ import annotations

import json
from typing import Any
from typing import Optional

import aio_pika
from application.documentation import DocumentationApplication
from application.schemas import PageProcessingTask
from shared.logging import get_logger
from shared.utils import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RabbitMQConsumer:
    def __init__(self, documentation_app: DocumentationApplication):
        self.documentation_app = documentation_app
        self.connection = None
        self.channel = None

    async def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            self.channel = await self.connection.channel()

            # Set QoS to process one message at a time
            await self.channel.set_qos(prefetch_count=1)

            logger.info('Consumer connected to RabbitMQ')
        except Exception as e:
            logger.error(f"Failed to connect consumer to RabbitMQ: {str(e)}")
            raise

    async def start_consuming(self):
        """Start consuming page processing tasks."""
        if not self.channel:
            await self.connect()

        try:
            queue = await self.channel.declare_queue(
                'page_processing_queue',
                durable=True,
            )

            await queue.consume(self.process_message)
            logger.info('Started consuming page processing tasks')

        except Exception as e:
            logger.error(f"Failed to start consuming: {str(e)}")
            raise

    async def process_message(self, message: aio_pika.IncomingMessage):
        """Process a single page processing task."""
        async with message.process():
            try:
                task_data = json.loads(message.body.decode())
                task = PageProcessingTask(**task_data)

                logger.info(f"Processing page task: {task.page.page_title}")
                await self.documentation_app.process_page_content(task)

                logger.info(f"Completed page task: {task.page.page_title}")

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                # Message will be nacked and potentially requeued
                raise

    async def close(self):
        """Close connection to RabbitMQ."""
        if self.connection:
            await self.connection.close()
