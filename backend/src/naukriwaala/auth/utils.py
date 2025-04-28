"""This module contains utilities for authentication."""

# Standard Library
from typing import Optional

# Third Party Library
from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

# Package Library
from naukriwaala.config import Settings

# Globals.
load_dotenv()
API_KEY = Settings.API_KEY
API_KEY_NAME = "X-API-Key"  # pragma: allowlist secret

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: Optional[str] = Security(api_key_header)) -> str:
    """Validate API key from header.

    Parameters
    ----------
    api_key_header
        The API key provided in the request header.

    Returns
    -------
    str
        The API key if valid.

    Raises
    ------
    HTTPException
        If the API key is invalid or missing.
    """

    if api_key_header and api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        detail="Invalid or missing API Key.", status_code=status.HTTP_401_UNAUTHORIZED
    )
