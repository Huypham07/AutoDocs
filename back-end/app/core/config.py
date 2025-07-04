import logging
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
import re

logger = logging.getLogger(__name__)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
load_dotenv(os.path.join(BASE_DIR, '.env'))

try:
    from adalflow import OllamaClient, GoogleGenAIClient
except ImportError as e:
    logger.warning(f"Some client imports failed: {e}")
    # Define empty classes as fallbacks
    class OllamaClient: pass

PROJECT_NAME = os.getenv('PROJECT_NAME', 'FASTAPI BASE')
API_PREFIX = ''
BACKEND_CORS_ORIGINS = ['*']
SECURITY_ALGORITHM = 'HS256'
PORT = int(os.getenv('PORT', 8000))
NODE_ENV = os.getenv('NODE_ENV', 'development')
DEFAULT_EXCLUDED_DIRS = [
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    "./logs/", "./log/", "./tmp/", "./temp/",
]
DEFAULT_EXCLUDED_FILES = [
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk", ".env",
    ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv", ".gitignore",
    ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    "pyproject.toml", "tsconfig.json", "webpack.config.js", "babel.config.js",
    "rollup.config.js", "jest.config.js", "karma.conf.js", "vite.config.js",
    "next.config.js", "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
    "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib", "*.o",
    "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class", "*.pyc", "*.pyd",
    "*.pyo", "__pycache__", "*.a", "*.lib", "*.lo", "*.la", "*.slo", "*.dSYM",
    "*.egg", "*.egg-info", "*.dist-info", "*.eggs", "node_modules",
    "bower_components", "jspm_packages", "lib-cov", "coverage", "htmlcov",
    ".nyc_output", ".tox", "dist", "build", "bld", "out", "bin", "target",
    "packages/*/dist", "packages/*/build", ".output"
]
MAX_EMBEDDING_TOKENS = 8192

# HOSTS
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# API Keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', None)

# --- end API Keys ---

# Configuration models with API Keys
import google.generativeai as genai
if GOOGLE_API_KEY:
    logger.info("Configuring Google GenAI with API Key")
    genai.configure(api_key=GOOGLE_API_KEY)

configs = {}

EMBEDDER_CONFIG = {
    "embedder": {
        "model_client": "OllamaClient",
        "model_kwargs": {
            "model": "nomic-embed-text"
        }
    },
    "retriever": {
        "top_k": 20
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 350,
        "chunk_overlap": 100
    }
}

GENERATOR_CONFIG = {
  "default_provider": "google",
  "providers": {
    "google": {
    
      "default_model": "gemini-2.0-flash",
      "supportsCustomModel": True,
      "models": {
        "gemini-2.0-flash": {
          "temperature": 0.7,
          "top_p": 0.8,
          "top_k": 20
        },
        "gemini-2.5-flash-preview-05-20": {
          "temperature": 0.7,
          "top_p": 0.8,
          "top_k": 20
        },
        "gemini-2.5-pro-preview-03-25": {
          "temperature": 0.7,
          "top_p": 0.8,
          "top_k": 20
        }
      }
    },
    "ollama": {
      "default_model": "qwen3:1.7b",
      "supportsCustomModel": True,
      "models": {
        "qwen3:1.7b": {
          "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "num_ctx": 32000
          }
        },
        "llama3:8b": {
          "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "num_ctx": 8000
          }
        },
        "qwen3:8b": {
          "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "num_ctx": 32000
          }
        }
      }
    },
  }
}


def replace_env_placeholders(config: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    Recursively replace placeholders like "${ENV_VAR}" in string values
    with environment variable values.
    """
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    def replacer(match: re.Match[str]) -> str:
        env_var_name = match.group(1)
        original_placeholder = match.group(0)
        env_var_value = os.environ.get(env_var_name)
        if env_var_value is None:
            logger.warning(
                f"Environment variable placeholder '{original_placeholder}' was not found. "
                f"Using placeholder as is."
            )
            return original_placeholder
        return env_var_value

    if isinstance(config, dict):
        return {k: replace_env_placeholders(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_placeholders(item) for item in config]
    elif isinstance(config, str):
        return pattern.sub(replacer, config)
    else:
        return config
        
def load_embedder_configs():
    """Load embedder and related configurations"""
    embedder_config = replace_env_placeholders(EMBEDDER_CONFIG.copy())
    embedder_config["embedder"]["model_client"] = OllamaClient
        
    # Update embedder configuration
    for key in ["embedder", "retriever", "text_splitter"]:
        if key in embedder_config:
            configs[key] = embedder_config[key]
            
def load_generator_config():
    generator_config = replace_env_placeholders(GENERATOR_CONFIG.copy())

    # Add client classes to each provider
    if "providers" in generator_config:
        for provider_id, provider_config in generator_config["providers"].items():
            default_map = {
                    "google": GoogleGenAIClient,
                    "ollama": OllamaClient,
                }
            provider_config["model_client"] = default_map[provider_id]

    for key in ["default_provider", "providers"]:
        if key in generator_config:
            configs[key] = generator_config[key]

            
def get_embedder_config() -> Dict[str, Any]:
    """
    Get the current embedder configuration.

    Returns:
        dict: The embedder configuration with model_client resolved
    """
    return configs.get("embedder", {})



def get_generator_model_config(provider="google", model=None):
    """
    Get configuration for the specified provider and model
    """
    provider_config = configs["providers"].get(provider)
    if not provider_config:
        raise ValueError(f"Configuration for provider '{provider}' not found")

    model_client = provider_config.get("model_client")
    if not model_client:
        raise ValueError(f"Model client not specified for provider '{provider}'")

    # If model not provided, use default model for the provider
    if not model:
        model = provider_config.get("default_model")
        if not model:
            raise ValueError(f"No default model specified for provider '{provider}'")

    # Get model parameters (if present)
    model_params = {}
    if model in provider_config.get("models", {}):
        model_params = provider_config["models"][model]
    else:
        default_model = provider_config.get("default_model")
        model_params = provider_config["models"][default_model]

    # Prepare base configuration
    result = {
        "model_client": model_client,
    }

    # Provider-specific adjustments
    if provider == "ollama":
        # Ollama uses a slightly different parameter structure
        if "options" in model_params:
            result["model_kwargs"] = {"model": model, **model_params["options"]}
        else:
            result["model_kwargs"] = {"model": model}
    else:
        # Standard structure for other providers
        result["model_kwargs"] = {"model": model, **model_params}

    return result

load_embedder_configs()
load_generator_config()