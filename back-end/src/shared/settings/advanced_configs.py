from __future__ import annotations

from typing import Any
from typing import Dict

import google.generativeai as genai
from adalflow import GoogleGenAIClient
from adalflow import OllamaClient
from adalflow.components.model_client import BedrockAPIClient
from adalflow.components.model_client import OpenAIClient
from shared.logging import get_logger
from shared.utils import get_settings

logger = get_logger(__name__)
settings = get_settings()

DEFAULT_EXCLUDED_DIRS = [
    './.venv/', './venv/', './env/', './virtualenv/',
    './node_modules/', './bower_components/', './jspm_packages/',
    './.git/', './.svn/', './.hg/', './.bzr/',
    './__pycache__/', './.pytest_cache/', './.mypy_cache/', './.ruff_cache/', './.coverage/',
    './dist/', './build/', './out/', './target/', './bin/', './obj/',
    './docs/', './_docs/', './site-docs/', './_site/',
    './.idea/', './.vscode/', './.vs/', './.eclipse/', './.settings/',
    './logs/', './log/', './tmp/', './temp/',
]
DEFAULT_EXCLUDED_FILES = [
    'yarn.lock', 'pnpm-lock.yaml', 'npm-shrinkwrap.json', 'poetry.lock',
    'Pipfile.lock', 'requirements.txt.lock', 'Cargo.lock', 'composer.lock',
    '.lock', '.DS_Store', 'Thumbs.db', 'desktop.ini', '*.lnk', '.env',
    '.env.*', '*.env', '*.cfg', '*.ini', '.flaskenv', '.gitignore',
    '.gitattributes', '.gitmodules', '.github', '.gitlab-ci.yml',
    '.prettierrc', '.eslintrc', '.eslintignore', '.stylelintrc',
    '.editorconfig', '.jshintrc', '.pylintrc', '.flake8', 'mypy.ini',
    'pyproject.toml', 'tsconfig.json', 'webpack.config.js', 'babel.config.js',
    'rollup.config.js', 'jest.config.js', 'karma.conf.js', 'vite.config.js',
    'next.config.js', '*.min.js', '*.min.css', '*.bundle.js', '*.bundle.css',
    '*.map', '*.gz', '*.zip', '*.tar', '*.tgz', '*.rar', '*.7z', '*.iso',
    '*.dmg', '*.img', '*.msix', '*.appx', '*.appxbundle', '*.xap', '*.ipa',
    '*.deb', '*.rpm', '*.msi', '*.exe', '*.dll', '*.so', '*.dylib', '*.o',
    '*.obj', '*.jar', '*.war', '*.ear', '*.jsm', '*.class', '*.pyc', '*.pyd',
    '*.pyo', '__pycache__', '*.a', '*.lib', '*.lo', '*.la', '*.slo', '*.dSYM',
    '*.egg', '*.egg-info', '*.dist-info', '*.eggs', 'node_modules',
    'bower_components', 'jspm_packages', 'lib-cov', 'coverage', 'htmlcov',
    '.nyc_output', '.tox', 'dist', 'build', 'bld', 'out', 'bin', 'target',
    'packages/*/dist', 'packages/*/build', '.output',
]

# Configuration models with API Keys
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)

configs = {}


def load_embedder_configs():
    """Load embedder and related configurations"""
    embedder_config = settings.EMBEDDER_CONFIG
    embedder_config['embedder']['model_client'] = OllamaClient

    # Update embedder configuration
    for key in ['embedder', 'retriever', 'text_splitter']:
        if key in embedder_config:
            configs[key] = embedder_config[key]


def load_generator_config():
    """Load generator configuration"""
    generator_config = settings.GENERATOR_CONFIG
    # Add client classes to each provider
    if 'providers' in generator_config:
        for provider_id, provider_config in generator_config['providers'].items():
            default_map = {
                'google': GoogleGenAIClient,
                'openai': OpenAIClient,
                'bedrock': BedrockAPIClient,
            }
            provider_config['model_client'] = default_map[provider_id]

    for key in ['default_provider', 'providers']:
        if key in generator_config:
            configs[key] = generator_config[key]


def get_embedder_config() -> Dict[str, Any]:
    """
    Get the current embedder configuration.

    Returns:
        dict: The embedder configuration with model_client resolved
    """
    return configs.get('embedder', {})


def get_generator_model_config(provider='google', model=None):
    """
    Get configuration for the specified provider and model
    """
    provider_config = configs['providers'].get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found")

    model_client = provider_config.get('model_client')
    if not model_client:
        raise ValueError(f"Model client not specified for provider '{provider}'")

    # If model not provided, use default model for the provider
    if not model:
        model = provider_config.get('default_model')
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}'")

    # Get model parameters (if present)
    model_params = {}
    if model in provider_config.get('models', {}):
        model_params = provider_config['models'][model]
    else:
        default_model = provider_config.get('default_model')
        model_params = provider_config['models'][default_model]

    # Prepare base configuration
    result = {
        'model_client': model_client,
    }

    # Standard structure for other providers
    result['model_kwargs'] = {'model': model, **model_params}

    return result


load_embedder_configs()
load_generator_config()
