from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient
from shared.logging import get_logger
from shared.utils import get_settings

logger = get_logger(__name__)
settings = get_settings()


class MongoDB:
    client = None
    database = None


mongodb = MongoDB()


async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
        mongodb.database = mongodb.client[settings.DATABASE_NAME]

        # Test connection
        await mongodb.client.admin.command('ping')
        logger.info('Successfully connected to MongoDB')

    except Exception as e:
        logger.error(f'Failed to connect to MongoDB: {str(e)}')
        raise


async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        logger.info('MongoDB connection closed')


def get_database():
    """Get database instance"""
    return mongodb.database
