"""This module contains the FastAPI application for the backend."""

# Standard Library
import os

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

# Third Party Library
import logfire
import sentry_sdk

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
from redis import asyncio as aioredis

# Package Library
from naukriwaala import admin, naukri, recommendation
from naukriwaala.config import Settings
from naukriwaala.graphs.utils import create_graph_mappings
from naukriwaala.prometheus_middleware import PrometheusMiddleware
from naukriwaala.utils.embeddings import load_embedding_model
from naukriwaala.utils.general import make_dir
from naukriwaala.utils.logging_ import initialize_logger

# Globals.
DOMAIN_NAME = os.getenv("DOMAIN_NAME", "")
LOGGING_LEVEL = Settings.LOGGING_LOG_LEVEL
MODELS_EMBEDDING_OPENAI = Settings.MODELS_EMBEDDING_OPENAI
MODELS_EMBEDDING_ST = Settings.MODELS_EMBEDDING_ST
REDIS_HOST = Settings.REDIS_HOST
REDIS_PORT = Settings.REDIS_PORT
SENTRY_DSN = Settings.SENTRY_DSN
SENTRY_TRACES_SAMPLE_RATE = Settings.SENTRY_TRACES_SAMPLE_RATE

# Only need to initialize loguru once for the entire backend!
logger = initialize_logger(logging_level=LOGGING_LEVEL)

tags_metadata = [admin.TAG_METADATA, naukri.TAG_METADATA, recommendation.TAG_METADATA]


def create_app() -> FastAPI:
    """Create the FastAPI application for the backend. The process is as follows:

    1. Set up logfire for the app.
    2. Include routers for all the endpoints.
    3. Add CORS middleware for cross-origin requests.
    4. Add Prometheus middleware for metrics.
    5. Mount the metrics app on /metrics as an independent application.

    Returns
    -------
    FastAPI
        The application instance.
    """

    app = FastAPI(
        debug=True,
        lifespan=lifespan,
        openapi_tags=tags_metadata,
        title="Naukriwaala APIs",
    )

    # 1.
    logfire.instrument_fastapi(app)
    logfire.instrument_httpx()
    logfire.instrument_redis(capture_statement=True)  # Set to `False` if sensitive data

    # 2.
    app.include_router(admin.routers.router)
    app.include_router(naukri.routers.router)
    app.include_router(recommendation.routers.router)

    origins = [
        f"http://{DOMAIN_NAME}",
        f"http://{DOMAIN_NAME}:3000",
        f"https://{DOMAIN_NAME}",
    ]

    # 3.
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_headers=["*"],
        allow_methods=["*"],
        allow_origins=origins,
    )

    # 4.
    app.add_middleware(PrometheusMiddleware)
    metrics_app = create_metrics_app()

    # 5.
    app.mount("/metrics", metrics_app)

    if not SENTRY_DSN or SENTRY_DSN == "" or SENTRY_DSN == "https://...":
        logger.log("ATTN", "No SENTRY_DSN provided. Sentry is disabled.")
    else:
        sentry_sdk.init(
            _experiments={"continuous_profiling_auto_start": True},
            dsn=SENTRY_DSN,
            traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        )

    return app


def create_metrics_app() -> Callable:
    """Create prometheus metrics app

    Returns
    -------
    Callable
        The metrics app.
    """

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return make_asgi_app(registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan events for the FastAPI application.

    The process is as follows:

    1. Load the embedding model.
    2. Connect to Redis.
    3. Create graph descriptions for the Diagnostic Agent.
    4. Yield control to the application.
    5. Close the Redis connection when the application finishes.

    Parameters
    ----------
    app
        The application instance.
    """

    logger.info("Application started!")

    make_dir(Path(os.getenv("PATHS_LOGS_DIR", "/tmp")) / "chat_sessions")

    # 1.
    app.state.embedding_model_openai = load_embedding_model(  # pylint: disable=all
        embedding_model_name=MODELS_EMBEDDING_OPENAI
    )
    app.state.embedding_model_st = load_embedding_model(  # pylint: disable=all
        embedding_model_name=MODELS_EMBEDDING_ST
    )

    # 2.
    logger.info("Initializing Redis client...")
    app.state.redis = await aioredis.from_url(
        f"{REDIS_HOST}:{REDIS_PORT}", decode_responses=True
    )
    logger.success("Redis connection established!")

    # 3.
    logger.info("Loading graph descriptions for the Diagnostic Agent...")
    create_graph_mappings()
    logger.success("Finished loading graph descriptions for the Diagnostic Agent!")

    logger.log("CELEBRATE", "Ready to roll! 🚀")

    # 4.
    yield

    # 5.
    logger.info("Closing Redis connection...")
    await app.state.redis.close()
    logger.success("Redis connection closed!")

    logger.success("Application finished!")
