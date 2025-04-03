"""This module contains the main entry point for the FastAPI application."""

import logging

import uvicorn
from src.api.app import app
from src.api.config import settings
from fastapi.logger import logger
from uvicorn.workers import UvicornWorker


class Worker(UvicornWorker):
    """Custom worker class to allow `root_path` to be passed to Uvicorn."""

    CONFIG_KWARGS = {"root_path": settings.BACKEND_ROOT_PATH}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        root_path=settings.BACKEND_ROOT_PATH,
    )
