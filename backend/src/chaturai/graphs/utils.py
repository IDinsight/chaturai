"""This module contains utilities for graphs."""

# Standard Library
import json
import os

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, cast

# Third Party Library
import griffe
import httpx

from _griffe.docstrings.models import DocstringSectionText
from loguru import logger
from playwright.async_api._generated import Page
from pydantic import BaseModel
from pydantic_graph import Graph
from redis import asyncio as aioredis

# Package Library
from chaturai.config import Settings
from chaturai.utils.general import make_dir

REDIS_CACHE_PREFIX_BROWSER_STATE = Settings.REDIS_CACHE_PREFIX_BROWSER_STATE

T = TypeVar("T", bound=BaseModel)


def create_adjacency_lists() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Create the forward and reverse adjacency lists for all graphs.

    Returns
    -------
    tuple[dict[str, list[str]], dict[str, list[str]]]
        A tuple containing the adjacency list and the reverse adjacency list for all
        graphs.
    """

    adjacency_list: dict[str, list[str]] = {
        "registration.register_student": [
            "registration.register_student",
            "login.login_student",
        ],
        "login.login_student": [
            "registration.register_student",
            # "profile_completion.complete_profile",
        ],
    }
    reverse_adjacency_list = defaultdict(list)
    for module_path, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            reverse_adjacency_list[neighbor].append(module_path)
    return adjacency_list, dict(reverse_adjacency_list)


def create_graph_mappings() -> None:
    """Create the graph mappings for all graphs."""

    adj_list, adj_list_reverse = create_adjacency_lists()
    base_module_path = "chaturai.graphs"
    graph_mapping = {}
    for module_path, adjacency_list in adj_list.items():
        full_module_path = f"{base_module_path}.{module_path}"
        graph_fn = griffe.load(full_module_path, docstring_parser="numpy")
        section_text = next(
            (
                section
                for section in graph_fn.docstring.parsed  # type: ignore
                if isinstance(section, DocstringSectionText)
            ),
            None,
        )
        assert section_text, f"Docstring section text not found for: {full_module_path}"
        graph_mapping[module_path] = {
            "adj_list": adjacency_list,
            "adj_list_reverse": adj_list_reverse[module_path],
            "description": f"Assistant Name: {module_path}\nAssistant Description: {section_text.value}",
        }
    Settings._INTERNAL_GRAPH_MAPPING = graph_mapping


async def load_browser_state(*, redis_client: aioredis.Redis) -> dict[str, Any]:
    """Load browser state from Redis or return a default browser state.

    Parameters
    ----------
    redis_client
        The Redis client.

    Returns
    -------
    dict[str, Any]
        The browser state.
    """

    browser_state = await redis_client.get(REDIS_CACHE_PREFIX_BROWSER_STATE)
    if browser_state:
        return json.loads(browser_state)
    logger.warning(
        f"""{REDIS_CACHE_PREFIX_BROWSER_STATE} not found in Redis cache. Returning
        empty browser state!"""
    )
    return {"cookies": [], "origins": []}


def load_graph_run_results(
    *,
    graph_run_results: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
    load_fp: Optional[str | Path] = None,
    model_class: Type[T],
) -> T | list[T]:
    """Load graph run results from file.

    Parameters
    ----------
    graph_run_results
        If specified, then this should be raw results from the graph run. These are
        then validated using the model class.
    load_fp
        The filepath to load the graph run results from.
    model_class
        The Pydantic model class to use for validation.

    Returns
    -------
    T | list[T]
        The loaded graph run results.
    """

    assert (
        graph_run_results or load_fp and not (graph_run_results and load_fp)
    ), "Either `graph_run_results` or `load_fp` can be specified but not both."

    if load_fp:
        logger.info(f"Loading graph run results from: {load_fp}")
        load_fp = Path(load_fp)
        graph_run_results = json.loads(load_fp.read_text(encoding="utf-8"))

    logger.info(f"Validating graph run results using: {model_class.__name__}")

    if isinstance(graph_run_results, dict):
        return model_class.model_validate(graph_run_results)
    assert isinstance(graph_run_results, list)
    return [model_class.model_validate(result) for result in graph_run_results]


def save_graph_diagram(*, graph: Graph) -> None:
    """Save the graph as a Mermaid diagram.

    Parameters
    ----------
    graph
        The graph to save as a Mermaid diagram.
    """

    project_dir = Path(os.getenv("PATHS_PROJECT_DIR", ""))
    assert project_dir.is_dir(), f"{project_dir} is not a valid directory."
    save_dir = project_dir / "results" / "graph_diagrams"
    logger.info(f"Saving graph diagram for '{graph.name}' to: {save_dir}")
    make_dir(save_dir)
    try:
        graph.mermaid_save(
            str(save_dir / f"{graph.name}.png"),
            background_color="ffffff",
            theme="forest",
        )
        logger.success(
            f"Finished saving graph diagram for '{graph.name}' to: {save_dir}"
        )
    except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
        logger.error(f"Failed to save graph diagram for '{graph.name}' due to: {e}")


async def save_browser_state(
    *, page: Page, redis_client: aioredis.Redis, session_id: int | str
) -> None:
    """Save browser state to Redis.

    Parameters
    ----------
    page
        The Playwright page object.
    redis_client
        The Redis client.
    session_id
        The session ID to use for the Redis cache key.
    """

    browser_state = await page.context.storage_state()
    redis_cache_key = f"{REDIS_CACHE_PREFIX_BROWSER_STATE}_{session_id}"
    await redis_client.set(redis_cache_key, json.dumps(browser_state))


async def save_graph_run_results(
    *,
    graph_run_results: (
        dict[str, Any] | list[dict[str, Any]] | BaseModel | list[BaseModel]
    ),
    redis_cache_key: Optional[str] = None,
    redis_client: Optional[aioredis.Redis] = None,
    save_fp: Optional[str | Path] = None,
) -> None:
    """Save graph run results to Redis and/or file.

    Parameters
    ----------
    graph_run_results
        The graph run results to save.
    redis_cache_key
        The cache key to use for storing graph run results.
    redis_client
        The Redis client.
    save_fp
        The filepath to save the graph run results to.
    """

    assert (
        redis_client and redis_cache_key
    ) or save_fp, (
        "Either `redis_client` and `redis_cache_key` or `save_fp` must be specified."
    )

    if isinstance(graph_run_results, list):
        if isinstance(graph_run_results[0], BaseModel):
            items = cast(list[BaseModel], graph_run_results)
            serialized = json.dumps([item.model_dump() for item in items])
        else:
            serialized = json.dumps(graph_run_results)
    elif isinstance(graph_run_results, BaseModel):
        model = cast(BaseModel, graph_run_results)
        serialized = json.dumps(model.model_dump())
    else:
        serialized = json.dumps(graph_run_results)

    if redis_client and redis_cache_key:
        logger.info(f"Saving graph run results to Redis cache: {redis_cache_key}")
        await redis_client.set(redis_cache_key, serialized)
        logger.success(
            f"Finished saving graph run results to Redis cache: {redis_cache_key}"
        )

    if save_fp:
        logger.info(f"Saving graph run results to file: {save_fp}")
        save_fp = Path(save_fp)
        make_dir(save_fp.parent)
        save_fp.write_text(serialized, encoding="utf-8")
        logger.success(f"Finished saving graph run results to file: {save_fp}")
