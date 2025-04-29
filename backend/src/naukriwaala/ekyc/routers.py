"""This module contains FastAPI routers for E-KYC endpoints."""

# Standard Library
import os

# Third Party Library
import logfire

from fastapi import APIRouter, Depends
from fastapi.requests import Request
from sqlalchemy.ext.asyncio import AsyncSession

# Package Library
from naukriwaala.config import Settings
from naukriwaala.db.utils import get_async_session
from naukriwaala.ekyc.schemas import EKYCQuery, EKYCResults
from naukriwaala.graphs.ekyc import ekyc_graph
from naukriwaala.utils.chat import AsyncChatSessionManager, get_chat_session_manager

# Globals.
TAG_METADATA = {
    "description": "_Requires API key._ E-KYC engine",
    "name": "E-KYC",
}
router = APIRouter(tags=[TAG_METADATA["name"]])

GOOGLE_CREDENTIALS_FP = os.getenv("PATHS_GOOGLE_CREDENTIALS")
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@router.post("/ekyc-registration", response_model=EKYCResults)
@logfire.instrument("Running E-KYC registration endpoint...")
async def ekyc_registration(
    ekyc_query: EKYCQuery,
    request: Request,
    asession: AsyncSession = Depends(get_async_session),
    csm: AsyncChatSessionManager = Depends(get_chat_session_manager),
    generate_graph_diagrams: bool = False,
    reset_chat_session: bool = False,
) -> EKYCResults:
    """E-KYC registration for students.

    Parameters
    ----------
    \n\tekyc_query
    \t\tThe E-KYC query object.
    \n\trequest
    \t\tThe FastAPI request object.
    \n\tasession
    \t\tThe SQLAlchemy async session to use for all database connections.
    \n\tcsm
    \t\tAn async chat session manager that manages the chat sessions for each user.
    \n\tgenerate_graph_diagrams
    \t\tSpecifies whether to generate graph diagrams for the endpoint.
    \n\treset_chat_session
    \t\tSpecifies whether to reset the chat session for the user. This can be used to
    \t\tclear the chat history and start a new session. This is useful for testing or
    \t\tdebugging purposes. By default, it is set to `False`.

    Returns
    -------
    \n\tEKYCResults
    \t\tThe E-KYC response.
    """

    return await ekyc_graph(
        asession=asession,
        csm=csm,
        ekyc_query=ekyc_query,
        embedding_model=request.app.state.embedding_model_openai,
        generate_graph_diagrams=generate_graph_diagrams,
        redis_client=request.app.state.redis,
        reset_chat_session=reset_chat_session,
    )
