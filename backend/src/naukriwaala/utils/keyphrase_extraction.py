"""This module contains utilities for keyphrase extraction using `KeyBERT`."""

# Standard Library
from typing import Optional

# Third Party Library
import numpy as np
import openai

from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from keybert.llm import OpenAI
from keyphrase_vectorizers import KeyphraseCountVectorizer
from loguru import logger

# Package Library
from naukriwaala.config import Settings
from naukriwaala.utils.general import convert_to_list
from naukriwaala.utils.litellm_ import get_embedding

MODELS_EMBEDDING_OPENAI = Settings.MODELS_EMBEDDING_OPENAI
MODELS_LLM = Settings.MODELS_LLM
OPENAI_API_KEY = Settings.OPENAI_API_KEY


class OpenAIEmbedder(BaseEmbedder):
    """Custom embedder class for keyphrase extraction using OpenAI embedding models."""

    def __init__(self, *, embedding_model_name: str = "text-embedding-3-large") -> None:
        """

        Parameters
        ----------
        embedding_model_name
            The name of the OpenAI embedding model to use.
        """

        super().__init__()

        self.embedding_model_name = embedding_model_name

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        """Embed documents. This method is called twice by `KeyBERT` and produces both
        document and word (i.e., keyphrase) embeddings.

        Parameters
        ----------
        documents
            List of strings to embed.
        verbose
            Specifies whether to log additional debugging information.

        Returns
        -------
        np.ndarray
            NumPy array containing the embeddings for the input documents.
        """

        documents = convert_to_list(documents)
        if verbose:
            logger.debug(f"{documents = }")
        assert len(documents) == 1, f"{documents = }"
        if isinstance(documents[0], np.ndarray):
            documents = documents[0].tolist()
        embeddings_response = get_embedding(
            model_name=self.embedding_model_name, text=documents
        )
        embeddings = np.array(
            [dict_["embedding"] for dict_ in embeddings_response["data"]]
        )
        return embeddings


def extract_keyphrases(
    *,
    documents: str | list[str],
    doc_embeddings: np.ndarray | list[np.ndarray],
    embedding_model_name: str = MODELS_EMBEDDING_OPENAI,
    prompt: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
    use_llm: bool = False,
    threshold: float = 0.8,
    use_mmr: bool = True,
) -> list[str]:
    """Extract keyphrases from documents using `KeyBERT`.

    Parameters
    ----------
    documents
        A string or a list of strings (documents) from which to extract keyphrases.
    doc_embeddings
        The embeddings for the documents.
    embedding_model_name
        The name of the embedding model to use. If an OpenAI embedding model is used,
        it should start with "openai/".
    prompt
        The prompt to be used in the LLM model. NB: Use `"[DOCUMENT]"` in the prompt to
        decide where the document needs to be inserted for keyphrase extraction.
    system_prompt
        The message that sets the behavior of the LLM. It's typically used to provide
        high-level instructions for extracting keyphrases.
    use_llm
        Specifies whether to use a language model (LLM) for keyphrase extraction.
    threshold
        Minimum similarity value between 0 and 1 used to decide how similar documents
        need to receive the same keywords. A higher value will reduce the number of
        documents that are clustered together and a lower value will increase the
        number of documents that are clustered together.
    use_mmr
        Boolean indicating whether to use Maximal Marginal Relevance (MMR) for
        keyphrase extraction.

    Returns
    -------
    list[str]
        List of keyphrases extracted from the documents.
    """

    documents = convert_to_list(documents)
    doc_embeddings_list = convert_to_list(doc_embeddings)
    assert len(documents) == len(doc_embeddings_list)
    llm = (
        OpenAI(
            openai.OpenAI(api_key=OPENAI_API_KEY),
            chat=True,
            generator_kwargs={"temperature": 0.7, "top_p": 0.9},
            model=MODELS_LLM.split("/")[1],
            prompt=prompt,
            system_prompt=system_prompt,
        )
        if use_llm
        else None
    )
    if MODELS_EMBEDDING_OPENAI.startswith("openai/"):
        embedding_model = OpenAIEmbedder(
            embedding_model_name=embedding_model_name.split("/")[1]
        )
        kw_model = KeyBERT(model=embedding_model, llm=llm)
    else:
        kw_model = KeyBERT(model=embedding_model_name, llm=llm)
    keyphrases_list = kw_model.extract_keywords(
        documents,
        doc_embeddings=np.vstack(doc_embeddings_list),
        threshold=threshold,
        use_mmr=use_mmr,
        vectorizer=KeyphraseCountVectorizer(),  # type: ignore
    )
    if isinstance(keyphrases_list[0], list):
        assert len(keyphrases_list) == 1
        keyphrases = keyphrases_list[0]
    else:
        assert isinstance(keyphrases_list[0], tuple) and isinstance(
            keyphrases_list[0][0], str
        )
        keyphrases = [x[0] for x in keyphrases_list]
    return keyphrases
