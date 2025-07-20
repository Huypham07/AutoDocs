from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from fastapi.security import HTTPBasic
from fastapi.security import HTTPBasicCredentials
from shared.utils import get_settings

settings = get_settings()
security = HTTPBasic()


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    """
    Authenticate the user based on provided HTTP Basic Auth credentials.

    This function verifies the username and password provided in the HTTP Basic Auth
    credentials against the configured settings. It uses a constant-time comparison
    to prevent timing attacks.

    Args:
        credentials (Annotated[HTTPBasicCredentials, Depends(security)]): The HTTP Basic Auth
            credentials provided by the client, obtained through FastAPI's dependency injection.

    Returns:
        str: The authenticated username if the credentials are valid.

    Raises:
        HTTPException: If the provided credentials are invalid, raises a 401 Unauthorized
            exception with appropriate headers.
    """
    current_username_bytes = credentials.username.encode('utf8')
    correct_username_bytes = settings.user_name_login.encode('utf8')
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes,
    )
    current_password_bytes = credentials.password.encode('utf8')
    correct_password_bytes = settings.password_login.encode('utf8')
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes,
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect username or password',
            headers={'WWW-Authenticate': 'Basic'},
        )
    return credentials.username
