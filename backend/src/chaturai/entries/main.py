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
import json
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
from uvicorn_worker import UvicornWorker

# Append the framework path. NB: This is required if this entry point is invoked from
# the command line. However, it is not necessary if it is imported from a pip install.
if __name__ == "__main__":
    # Path to src directory containing chaturai.
    package_path = Path(__file__).resolve().parents[2]

    if package_path not in sys.path:
        print(f"Appending '{package_path}' to system path...")
        sys.path.append(str(package_path))

# Package Library
from chaturai import create_app
from chaturai.chatur.utils import (
    check_chatur_agent_translation_response,
    check_student_translation_response,
)
from chaturai.config import Settings
from chaturai.prompts.chatur import ChaturPrompts
from chaturai.schemas import ValidatorCall
from chaturai.utils.chat import get_chat_session_manager, prettify_chat_history
from chaturai.utils.general import unwrap_union_type
from chaturai.utils.litellm_ import get_acompletion

assert (
    sys.version_info.major >= 3 and sys.version_info.minor >= 11
), "chaturai requires at least Python 3.11!"

# Instantiate typer apps for the command line interface.
cli = typer.Typer()

AGENTS_CHATUR_AGENT = Settings.AGENTS_CHATUR_AGENT
GRAPHS_CHATUR_AGENT = Settings.GRAPHS_CHATUR_AGENT
LITELLM_MODEL_CHAT = Settings.LITELLM_MODEL_CHAT
REDIS_CACHE_PREFIX_CHAT_CLEANED = Settings.REDIS_CACHE_PREFIX_CHAT_CLEANED
TEXT_GENERATION_BEDROCK = Settings.TEXT_GENERATION_BEDROCK

app = create_app()


class Worker(UvicornWorker):
    """Custom worker class to allow `root_path` to be passed to Uvicorn."""

    CONFIG_KWARGS = {"root_path": os.getenv("PATHS_BACKEND_ROOT", "")}


async def explain_with_llm(
    *, error: dict[str, Any], raw_input: dict[str, Any], request: Request
) -> str:
    """Explain the validation error using LLM.

    The process is as follows:

    1.

    Parameters
    ----------
    error
        Dictionary containing the Pydantic model name and validation errors.
    raw_input
        The raw input for the endpoint.
    request
        The FastAPI request object.

    Returns
    -------
    str
        The LLM explanation of the validation error.
    """

    csm = await get_chat_session_manager(request=request)
    redis_client = request.app.state.redis
    user_id = raw_input["user_id"]

    # 1.
    _, session_id = await csm.check_if_chat_session_exists(
        namespace=AGENTS_CHATUR_AGENT,
        signed=True,
        user_id=f"{GRAPHS_CHATUR_AGENT}_{user_id}",
    )

    cleaned_chat_cache_key = (
        f"{REDIS_CACHE_PREFIX_CHAT_CLEANED}_Chatur_Agent_{session_id}"
    )
    cleaned_chat_history = []
    value = await redis_client.get(cleaned_chat_cache_key)
    if value is not None:
        try:
            cleaned_chat_history = json.loads(value)
        except json.JSONDecodeError:
            logger.error(
                f"Error encountered while loading cleaned chat history for "
                f"session: {session_id}"
            )
    prettified_chat_history = (
        prettify_chat_history(chat_history=cleaned_chat_history)
        if cleaned_chat_history
        else "No chat history available."
    )

    # 2.
    translation_dict = {"requires_translation": False, "translated_text": None}
    user_query = raw_input.get("user_query", None)
    if user_query:
        response = await get_acompletion(
            model=LITELLM_MODEL_CHAT,
            system_msg=ChaturPrompts.system_messages["translate_student_message"],
            text_generation_params=TEXT_GENERATION_BEDROCK,
            user_msg=ChaturPrompts.prompts["translate_student_message"].format(
                student_message=user_query
            ),
            validator_call=ValidatorCall(
                num_retries=3, validator_module=check_student_translation_response
            ),
        )
        translation_dict = json.loads(response)
        if translation_dict["requires_translation"]:
            raw_input["user_query"] = translation_dict["translated_text"]

    # 3.
    content = await get_acompletion(
        model=LITELLM_MODEL_CHAT,
        system_msg=ChaturPrompts.system_messages["explain_pydantic_errors"],
        text_generation_params=TEXT_GENERATION_BEDROCK,
        user_msg=ChaturPrompts.prompts["explain_pydantic_errors"].format(
            conversation_history=prettified_chat_history,
            student_input=raw_input,
            validation_errors=error,
        ),
    )

    # 4.
    if translation_dict["requires_translation"]:
        content = await get_acompletion(
            model=LITELLM_MODEL_CHAT,
            system_msg=ChaturPrompts.system_messages["translate_chatur_agent_message"],
            text_generation_params=TEXT_GENERATION_BEDROCK,
            user_msg=ChaturPrompts.prompts["translate_chatur_agent_message"].format(
                summary_for_student=content
            ),
            validator_call=ValidatorCall(
                num_retries=3,
                validator_module=check_chatur_agent_translation_response,
            ),
        )

    return content


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
        first_error, raw_input = await revalidate_request_for_errors(
            endpoint=endpoint, request=request
        )
        if first_error:
            error_summary = await explain_with_llm(
                error=first_error, raw_input=raw_input, request=request
            )
            return JSONResponse(
                content={
                    "error": str(first_error),
                    "error_summary": error_summary,
                    "message": "Validation failed.",
                },
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

    # Default FastAPI validation error structure for non-enhanced endpoints.
    return JSONResponse(
        content={
            "error": exc.errors(),
            "error_summary": None,
            "message": "Validation failed.",
        },
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def revalidate_request_for_errors(
    *, endpoint: Callable[..., Awaitable[Any]], request: Request
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the request input against the expected Pydantic model(s) for error
    handling.

    NB: This function breaks after the **first** error that occurs during Pydantic
    validation.

    Parameters
    ----------
    endpoint
        The FastAPI endpoint function.
    request
        The FastAPI request object.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Tuple containing a dictionary with the model name and validation errors, and
        the raw input for the endpoint.
    """

    first_error = {}
    pydantic_models = getattr(endpoint, "_pydantic_models", None)
    raw_input = {}
    if pydantic_models:
        try:
            raw_input = await request.json()
        except Exception:  # pylint: disable=W0718
            pass

        for pydantic_model in unwrap_union_type(model_union=pydantic_models):
            try:
                pydantic_model(**raw_input)
            except ValidationError as e:
                first_error = {
                    "model": pydantic_model.__name__,
                    "errors": e.errors(),
                }
                break
            except Exception as e:  # pylint: disable=W0718
                first_error = {
                    "model": pydantic_model.__name__,
                    "errors": [
                        {
                            "input": raw_input,
                            "loc": ("__exception__",),
                            "msg": str(e),
                            "type": "runtime_exception",
                        }
                    ],
                }
                break

        logger.error(
            f"The following error was encountered during validation:\n"
            f"Pydantic Model: {first_error['model']}\n"
            f"Errors: {first_error['errors']}"
        )

    return first_error, raw_input


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
