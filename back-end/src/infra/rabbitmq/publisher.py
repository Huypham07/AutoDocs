from __future__ import annotations

import json
from typing import Any
from typing import Optional

import aio_pika
from application.schemas import PageProcessingTask
from shared.logging import get_logger
from shared.utils import get_settings


logger = get_logger(__name__)
settings = get_settings()


class RabbitMQPublisher:
    def __init__(self):
        self.connection = None
        self.channel = None

    async def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
            self.channel = await self.connection.channel()

            # Declare queue for page processing tasks
            await self.channel.declare_queue(
                'page_processing_queue',
                durable=True,  # Queue survives broker restart
            )

            logger.info('Connected to RabbitMQ')
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    async def publish_page_task(self, task: PageProcessingTask):
        """Publish a page processing task to the queue."""
        if not self.channel:
            await self.connect()

        try:
            message_body = task.json()
            message = aio_pika.Message(
                message_body.encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # Message survives broker restart
            )

            await self.channel.default_exchange.publish(
                message,
                routing_key='page_processing_queue',
            )

            logger.info(f"Published page processing task: {task.page.page_title}")
        except Exception as e:
            logger.error(f"Failed to publish page task: {str(e)}")
            raise

    async def close(self):
        """Close connection to RabbitMQ."""
        if self.connection:
            await self.connection.close()
