"""This module contains FastAPI routers for naukri endpoints."""

# Standard Library
import os

# Third Party Library
import logfire

from fastapi import APIRouter, Depends
from fastapi.requests import Request

# Package Library
from naukriwaala.config import Settings
from naukriwaala.graphs.naukri import naukri
from naukriwaala.naukri.schemas import NaukriFlowResults, NaukriQueryUnion
from naukriwaala.utils.chat import AsyncChatSessionManager, get_chat_session_manager

TAG_METADATA = {
    "description": "_Requires API key._ Naukri automation flow",
    "name": "Naukri",
}
router = APIRouter(tags=[TAG_METADATA["name"]])

GOOGLE_CREDENTIALS_FP = os.getenv("PATHS_GOOGLE_CREDENTIALS")
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@router.post("/naukri-flow", response_model=NaukriFlowResults)
@logfire.instrument("Running Naurki flow endpoint...")
async def naukri_flow(
    naukri_query: NaukriQueryUnion,
    request: Request,
    csm: AsyncChatSessionManager = Depends(get_chat_session_manager),
    generate_graph_diagrams: bool = False,
    reset_chat_and_graph_state: bool = False,
) -> NaukriFlowResults:
    """Naukri flow.

    Parameters
    ----------
    \n\tnaukri_query
    \t\tThe query object.
    \n\trequest
    \t\tThe FastAPI request object.
    \n\tcsm
    \t\tAn async chat session manager that manages the chat sessions for each user.
    \n\tgenerate_graph_diagrams
    \t\tSpecifies whether to generate graph diagrams for the endpoint.
    \n\treset_chat_and_graph_state
    \t\tSpecifies whether to reset the chat session and the graph state for the user.
    \t\tThis can be used to clear the chat history and graph state, effectively
    \t\tstarting a completely new session. This is useful for testing or debugging
    \t\tpurposes.

    Returns
    -------
    \n\tNaukriFlowResults
    \t\tThe Naukri response.
    """

    return await naukri(
        csm=csm,
        generate_graph_diagrams=generate_graph_diagrams,
        naukri_query=naukri_query,
        redis_client=request.app.state.redis,
        reset_chat_and_graph_state=reset_chat_and_graph_state,
    )
