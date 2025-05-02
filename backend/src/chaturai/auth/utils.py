"""This module contains utilities for authentication."""

# Standard Library
from typing import Optional

# Third Party Library
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

# Package Library
from chaturai.config import Settings

API_KEY_NAME = "X-API-Key"  # pragma: allowlist secret
FASTAPI_API_KEY = Settings.FASTAPI_API_KEY

_api_key_header = APIKeyHeader(auto_error=False, name=API_KEY_NAME)


async def get_api_key(api_key_header: Optional[str] = Security(_api_key_header)) -> str:
    """Validate FastAPI API key from header.

    Parameters
    ----------
    api_key_header
        The FastAPI API key provided in the request header.

    Returns
    -------
    str
        The FastAPI API key if valid.

    Raises
    ------
    HTTPException
        If the FastAPI API key is invalid or missing.
    """

    if api_key_header and api_key_header == FASTAPI_API_KEY:
        return api_key_header
    raise HTTPException(
        detail="Invalid or missing API Key.", status_code=status.HTTP_401_UNAUTHORIZED
    )
