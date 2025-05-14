"""This module contains FastAPI routers for recommendation endpoints."""

# Standard Library
import os

# Third Party Library
import logfire

from fastapi import APIRouter, Depends, Query
from fastapi.requests import Request
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

# Package Library
from chaturai.auth.utils import get_api_key
from chaturai.config import Settings
from chaturai.db.utils import get_async_session
from chaturai.recommendation.basic_recommendation import BasicRecommendationEngine
from chaturai.recommendation.schemas import RecommendationResult, StudentProfile
from chaturai.utils.chat import AsyncChatSessionManager, get_chat_session_manager

TAG_METADATA = {
    "description": "_Requires API key._ Recommendation engine",
    "name": "Recommendation",
}
router = APIRouter(tags=[TAG_METADATA["name"]])

GOOGLE_CREDENTIALS_FP = os.getenv("PATHS_GOOGLE_CREDENTIALS")
TEXT_GENERATION_GEMINI = Settings.TEXT_GENERATION_GEMINI


@router.post("/recommend-apprenticeship", response_model=list[RecommendationResult])
@logfire.instrument("Running apprenticeship recommendations endpoint...")
async def recommend_apprenticeship(
    request: Request,
    student: StudentProfile,
    api_key: str = Depends(get_api_key),
    asession: AsyncSession = Depends(get_async_session),
    csm: AsyncChatSessionManager = Depends(get_chat_session_manager),
    include_score_components: bool = Query(
        default=False,
        description="Whether to include detailed scoring components in the response.",
    ),
    limit: int = Query(
        default=10,
        description="Maximum number of recommendations to return.",
        ge=1,
        le=100,
    ),
    reset_chat_session: bool = False,
) -> list[RecommendationResult]:
    """Get personalized apprenticeship recommendations for a student.

    This endpoint returns a list of apprenticeship opportunities ranked by their match
    score with the student's profile and preferences.

    The recommendations are based on various factors including:
        - Education level match
        - Location preferences
        - Sector preferences
        - Stipend requirements

    Parameters
    ----------
    \n\trequest
    \t\tThe FastAPI request object.
    \n\tstudent
    \t\tStudent profile and preferences object.
    \n\tapi_key
    \t\tThe API key for authentication.
    \n\tasession
    \t\tThe SQLAlchemy async session to use for all database connections.
    \n\tcsm
    \t\tAn async chat session manager that manages the chat sessions for each user.
    \n\tinclude_score_components
    \t\tSpecifies whether to include detailed scoring components in the response.
    \n\tlimit
    \t\tThe maximum number of recommendations to return.
    \n\treset_chat_session
    \t\tSpecifies whether to reset the chat session for the user. This can be used to
    \t\tclear the chat history and start a new session. This is useful for testing or
    \t\tdebugging purposes. By default, it is set to `False`.

    Returns
    -------
    \n\tlist[RecommendationResult]
    \t\tThe list of recommended opportunities sorted by match score.
    """

    logger.info(f"{api_key = }")
    logger.info(f"{request = }")
    logger.info(f"{asession = }")
    logger.info(f"{csm = }")
    logger.info(f"{reset_chat_session = }")
    engine = BasicRecommendationEngine()
    try:
        recommendations = engine.get_recommendations(
            include_score_components=include_score_components,
            limit=limit,
            student=student,
        )
        return recommendations
    finally:
        engine.close()
