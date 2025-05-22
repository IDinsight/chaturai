"""This module contains the main entry point for the backend application.

From the backend directory of this project, this entry point can be invoked from the
command line via:

python -m src.chaturai.entries.main

or

python src/chaturai/entries/main.py

If the `chaturai` package has been pip installed, then the entry point can be
invoked from the command line via:

chaturai --help
"""

# Standard Library
import os
import sys

from pathlib import Path
from typing import Any, Awaitable, Callable

# Third Party Library
import typer
import uvicorn

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError
from pydantic_core import ErrorDetails
from uvicorn_worker import UvicornWorker

# Append the framework path. NB: This is required if this entry point is invoked from
# the command line. However, it is not necessary if it is imported from a pip install.
if __name__ == "__main__":
    # Path to src directory containing chaturai
    package_path = Path(__file__).resolve().parents[2]

    if package_path not in sys.path:
        print(f"Appending '{package_path}' to system path...")
        sys.path.append(str(package_path))

# Package Library
from chaturai import create_app
from chaturai.config import Settings
from chaturai.utils.general import unwrap_union_type

assert (
    sys.version_info.major >= 3 and sys.version_info.minor >= 11
), "chaturai requires at least Python 3.11!"

# Instantiate typer apps for the command line interface.
cli = typer.Typer()


app = create_app()


class Worker(UvicornWorker):
    """Custom worker class to allow `root_path` to be passed to Uvicorn."""

    CONFIG_KWARGS = {"root_path": os.getenv("PATHS_BACKEND_ROOT", "")}


@app.exception_handler(RequestValidationError)
async def global_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handles all FastAPI validation errors. If the endpoint is annotated with
    LLM-enhanced validation metadata, it revalidates the raw input against the expected
    Pydantic model(s) and produces a contextual explanation via LLM.

    NB: If this handler is called, it means that the initial validation by FastAPI has
    already failed. Here, we are re-validating again using the Pydantic model(s) and
    contextualizing the error(s) using LLM for the service that invoked the endpoint.

    Parameters
    ----------
    request
        The FastAPI request object.
    exc
        The FastAPI validation error object.

    Returns
    -------
    JSONResponse
        A JSON response containing the validation error details and a contextual
        explanation.
    """

    endpoint = getattr(request.scope["route"], "endpoint", None)

    if endpoint and getattr(endpoint, "_pydantic_models", False):
        all_errors = await revalidate_request_for_errors(
            endpoint=endpoint, request=request
        )
        if all_errors:
            error_summary = ""  # await explain_with_llm(raw_input, all_errors)

            return JSONResponse(
                content={
                    "all_errors": [
                        {"model": model_name, "errors": errors}
                        for model_name, errors in all_errors
                    ],
                    "error_summary": error_summary,
                    "message": "Validation failed",
                },
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

    # Default FastAPI validation error structure for non-enhanced endpoints.
    return JSONResponse(
        content={"detail": exc.errors()},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def revalidate_request_for_errors(
    *, endpoint: Callable[..., Awaitable[Any]], request: Request
) -> list[tuple[str, list[ErrorDetails]]]:
    """Revalidate the request input against the expected Pydantic model(s) for error
    handling.

    Parameters
    ----------
    endpoint
        The FastAPI endpoint function.
    request
        The FastAPI request object.

    Returns
    -------
    list[tuple[str, list[ErrorDetails]]]
        A list of tuples containing the model name and the validation errors.
    """

    all_errors = []
    pydantic_models = getattr(endpoint, "_pydantic_models", None)
    if pydantic_models:
        try:
            raw_input = await request.json()
        except Exception:  # pylint: disable=W0718
            raw_input = {}

        for pydantic_model in unwrap_union_type(model_union=pydantic_models):
            try:
                pydantic_model(**raw_input)
            except ValidationError as e:
                all_errors.append((pydantic_model.__name__, e.errors()))
            except Exception as e:  # pylint: disable=W0718
                all_errors.append(
                    (
                        pydantic_model.__name__,
                        [
                            {
                                "input": raw_input,
                                "loc": ("__exception__",),
                                "msg": str(e),
                                "type": "runtime_exception",
                            }
                        ],
                    )
                )

        logger.error("The following errors were encountered during validation:")
        for model_name, errors in all_errors:
            logger.error(f"  → {model_name}:")
            for err in errors:
                logger.error(f"    - {err}")
    return all_errors


@cli.command()
def start(host: str = "0.0.0.0", port: int = 8000, reload: bool = True) -> None:
    """Start the FastAPI application using Uvicorn.

    The process is as follows:

    1. Run the Uvicorn server.

    Parameters
    ----------
    host
        The host address to bind the server to.
    port
        The port number to bind the server to.
    reload
        Specifies whether the server should automatically reload when changes are
        detected.
    """

    logger.info("Starting FastAPI with loguru...")

    # 1.
    project_dir = Path(os.getenv("PATHS_PROJECT_DIR", ""))
    assert project_dir.is_dir(), f"'{project_dir}' is not a directory."
    uvicorn.run(
        "chaturai.entries.main:app",
        host=host,
        port=port,
        log_config=None,  # Disable Uvicorn's default logging config
        log_level=Settings.LOGGING_LOG_LEVEL.lower(),
        reload=reload,
        reload_dirs=[str(project_dir / "backend" / "src")],
        root_path=os.getenv("PATHS_BACKEND_ROOT", ""),
    )


if __name__ == "__main__":
    cli()
