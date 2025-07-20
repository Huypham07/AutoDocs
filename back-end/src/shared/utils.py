from __future__ import annotations

import time
from functools import lru_cache
from functools import wraps

from shared.logging import get_logger
from shared.settings import Settings


@lru_cache
def get_settings():
    return Settings()  # type: ignore


def profile(func):
    """Decorator to profile execution time. Using default logger with info level\n
    Output: [module.function] executed in: 0.0s
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger = get_logger('profiler')
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(
            f'[{func.__module__}.{func.__name__}] executed in: {end_time - start_time}s',
        )

        if hasattr(result, 'processing_time'):
            setattr(result, 'processing_time', end_time - start_time)

        return result

    return wrapper


def extract_full_repo_name(repo_url: str) -> str:
    """
    Extracts the repository name from the given URL.
    """
    url_parts = repo_url.rstrip('/').split('/')

    # GitHub URL format: https://github.com/owner/repo
    if len(url_parts) >= 5:
        owner = url_parts[-2]
        repo = url_parts[-1].replace('.git', '')
        repo_name = f'{owner}_{repo}'
    else:
        repo_name = url_parts[-1].replace('.git', '')
    return repo_name


def extract_repo_info(repo_url: str) -> tuple[str, str]:
    """
    Extracts the owner and repository name from the given URL.
    """
    url_parts = repo_url.rstrip('/').split('/')

    if len(url_parts) >= 5:
        owner = url_parts[-2]
        repo_name = url_parts[-1].replace('.git', '')
    else:
        owner = url_parts[-1].replace('.git', '')
        repo_name = owner

    return owner, repo_name


def is_github_repo(repo_url: str) -> bool:
    """
    Checks if the given repository URL is a GitHub repository.
    """
    return 'github.com' in repo_url
