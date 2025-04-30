"""This module contains FastAPI routers for recommendation endpoints."""

# Standard Library
import os

# Third Party Library
import logfire

from fastapi import APIRouter, Depends
from fastapi.requests import Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

# Package Library
from naukriwaala.config import Settings
from naukriwaala.db.utils import get_async_session
from naukriwaala.recommendation.schemas import (
    RecommendationQuery,
    RecommendationQueryResults,
)
from naukriwaala.utils.chat import AsyncChatSessionManager, get_chat_session_manager

TAG_METADATA = {
    "description": "_Requires API key._ Recommendation engine",
    "name": "Recommendation",
}
router = APIRouter(tags=[TAG_METADATA["name"]])

GOOGLE_CREDENTIALS_FP = os.getenv("PATHS_GOOGLE_CREDENTIALS")
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@router.post("/recommend-apprenticeship", response_model=RecommendationQueryResults)
@logfire.instrument("Running apprenticeship recommendations endpoint...")
async def recommend_apprenticeship(
    rec_query: RecommendationQuery,
    request: Request,
    asession: AsyncSession = Depends(get_async_session),
    csm: AsyncChatSessionManager = Depends(get_chat_session_manager),
    reset_chat_session: bool = False,
) -> RecommendationQueryResults:
    """Recommend apprenticeships for students.

    Parameters
    ----------
    \n\trec_query
    \t\tThe recommendation query object.
    \n\trequest
    \t\tThe FastAPI request object.
    \n\tasession
    \t\tThe SQLAlchemy async session to use for all database connections.
    \n\tcsm
    \t\tAn async chat session manager that manages the chat sessions for each user.
    \n\treset_chat_session
    \t\tSpecifies whether to reset the chat session for the user. This can be used to
    \t\tclear the chat history and start a new session. This is useful for testing or
    \t\tdebugging purposes. By default, it is set to `False`.

    Returns
    -------
    \n\tRecommendationQueryResults
    \t\tThe recommendation response.
    """

    logger.info(f"{request = }")
    logger.info(f"{asession = }")
    logger.info(f"{csm = }")
    logger.info(f"{reset_chat_session = }")
    return RecommendationQueryResults(
        **rec_query.model_dump(),
        query_text_refined=rec_query.query_text,
        answer_response={"foo": "bar"},
        session_id=1,
    )
