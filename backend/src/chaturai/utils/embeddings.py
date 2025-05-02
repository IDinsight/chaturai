"""This module contains utilities for embeddings."""

# Standard Library
import os

from pathlib import Path
from typing import Any, Optional

# Third Party Library
import numpy as np
import openai

from bertopic.backend import OpenAIBackend
from loguru import logger
from sentence_transformers import SentenceTransformer

# Package Library
from chaturai.utils.general import write_to_json
from chaturai.utils.litellm_ import get_embedding


def get_doc_embeddings(
    *,
    docs: list[str],
    embedding_model: Any,
    embeddings_fp: Optional[Path] = None,
    responses_fp: Optional[Path] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Get document embeddings.

    Parameters
    ----------
    docs
        List containing the documents to embed.
    embedding_model
        The instantiated embedding model.
    embeddings_fp
        Filepath for saving document embeddings.
    responses_fp
        Filepath for saving embeddings response.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        The document embeddings as a NumPy array and the embeddings response. The
        embeddings response is a dictionary used when creating/updating the vector
        database. The format of the dictionary follows that of OpenAI embeddings
        response.

    Raises
    ------
    NotImplementedError
        If the logic to get embeddings from the embedding model is not implemented.
    """

    if not isinstance(embedding_model, (OpenAIBackend, SentenceTransformer)):
        raise NotImplementedError(
            f"Logic for obtaining embeddings for embedding model: "
            f"{str(embedding_model)}"
        )

    if isinstance(embedding_model, OpenAIBackend):
        embeddings_response = get_embedding(
            batch_size=100, model_name=embedding_model.embedding_model, text=docs
        )
        embeddings = np.array(
            [dict_["embedding"] for dict_ in embeddings_response["data"]]
        )
    else:
        embeddings = embedding_model.encode(docs)
        embeddings_response = {"data": []}
        for embedding in embeddings.tolist():
            embeddings_response["data"].append({"embedding": embedding})
    assert embeddings.shape[0] == len(docs) == len(embeddings_response["data"])

    if embeddings_fp:
        logger.info(f"Saving document embeddings to: {embeddings_fp}")
        np.save(embeddings_fp, embeddings)
        logger.info("Finished saving document embeddings!")
    if responses_fp:
        logger.info(f"Saving embeddings response to: {responses_fp}")
        write_to_json(responses_fp, embeddings_response)
        logger.info("Finished saving embeddings response!")
    return embeddings, embeddings_response


def load_embedding_model(
    *, embedding_model_name: str
) -> OpenAIBackend | SentenceTransformer:
    """Load the embedding model.

    Parameters
    ----------
    embedding_model_name
        The name of the embedding model to load. This should be in the format of either
        `openai/` or a `sentence-transformers` model name.

    Returns
    -------
    OpenAIBackend | SentenceTransformer
        The embedding model.
    """

    logger.log("ATTN", f"Loading embedding model: {embedding_model_name}...")

    match embedding_model_name:
        case name if name.startswith("openai/"):
            embedding_model = load_openai(model=embedding_model_name)
        case _:
            embedding_model = load_st(model=embedding_model_name)

    logger.success("Finished loading embedding model!")

    return embedding_model


def load_openai(*, model: str = "text-embedding-3-large") -> OpenAIBackend:
    """Load an OpenAI embedding model.

    Parameters
    ----------
    model
        The name of the OpenAI embedding model.

    Returns
    -------
    OpenAIBackend
        An `OpenAIBackend` object.
    """

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedding_model = OpenAIBackend(client=client, embedding_model=model)
    return embedding_model


def load_st(*, model: str = "all-mpnet-base-v2") -> SentenceTransformer:
    """Load a `SentenceTransformer` model.

    Parameters
    ----------
    model
        The name of the `SentenceTransformer` model.

    Returns
    -------
    SentenceTransformer
        A `SentenceTransformer` embedding model.
    """

    embedding_model = SentenceTransformer(model)  # pylint: disable=all
    return embedding_model
