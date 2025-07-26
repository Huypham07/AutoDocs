from __future__ import annotations

import os
from typing import Any
from typing import Dict

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import PydanticBaseSettingsSource
from pydantic_settings import YamlConfigSettingsSource
from yaml import safe_load

# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    # Environment variables from .env file
    GOOGLE_API_KEY: str = Field(default='', description='Google API Key')
    MONGODB_URL: str = Field(default='', description='MongoDB connection URL')
    DATABASE_NAME: str = Field(default='', description='Database name')

    # Environment variables from settings.yaml file
    PROJECT_NAME: str = Field(default='AutoDocs')
    PORT: int = Field(default=8000)
    NODE_ENV: str = Field(default='development')
    API_PREFIX: str = Field(default='')
    BACKEND_CORS_ORIGINS: list[str] = Field(default=['*'])
    RABBITMQ_URL: str = Field(default='', description='RabbitMQ connection URL')

    EMBEDDER_CONFIG: dict[str, Any] = {
        'embedder': {
            'model_client': 'OllamaClient',
            'model_kwargs': {
                'model': 'nomic-embed-text',
            },
        },
        'retriever': {
            'top_k': 20,
        },
        'text_splitter': {
            'split_by': 'word',
            'chunk_size': 350,
            'chunk_overlap': 100,
        },
    }

    GENERATOR_CONFIG: dict[str, Any] = {
        'default_provider': 'google',
        'providers': {
            'google': {

                'default_model': 'gemini-2.0-flash',
                'supportsCustomModel': True,
                'models': {
                    'gemini-2.0-flash': {
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 20,
                    },
                    'gemini-2.5-flash': {
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 20,
                    },
                    'gemini-2.5-flash-lite-preview-06-17': {
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 20,
                    },
                },
            },
        },
    }

    class Config:
        env_nested_delimiter = '__'
        yaml_file = 'settings.yaml'

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls),
        )
