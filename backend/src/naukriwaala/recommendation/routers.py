"""This module contains FastAPI routers for recommendation endpoints."""

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

# from naukriwaala.recommendation.basic_recommendation import get_apprenticeship
from naukriwaala.recommendation.schemas import (
    RecommendationQuery,
    RecommendationQueryResults,
)
from naukriwaala.utils.chat import AsyncChatSessionManager, get_chat_session_manager

# Globals.
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

    return await get_apprenticeship(
        asession=asession,
        csm=csm,
        embedding_model=request.app.state.embedding_model_openai,
        recommendation_query=rec_query,
        redis_client=request.app.state.redis,
        reset_chat_session=reset_chat_session,
    )


"""This module contains the main FastAPI application for Naukriwaala."""

# Third Party Library
from fastapi import Depends, FastAPI, Query
from loguru import logger

# Package Library
from naukriwaala.auth.utils import get_api_key
from naukriwaala.config import Settings
from naukriwaala.recommendation.basic_recommendation import BasicRecommendationEngine
from naukriwaala.recommendation.schemas import RecommendationResult, StudentProfile

app = FastAPI(
    title="Naukriwaala API",
    description="API for getting personalized apprenticeship recommendations",
    version="1.0.0",
    root_path=Settings.PATHS_BACKEND_ROOT,
)


@app.get(
    "/recommendations/",
    response_model=list[RecommendationResult],
    dependencies=[Depends(get_api_key)],
    tags=["recommendations"],
)
async def get_recommendations(
    student: StudentProfile,
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return",
    ),
    include_score_components: bool = Query(
        default=False,
        description="Whether to include detailed scoring components in the response",
    ),
) -> list[RecommendationResult]:
    """Get personalized apprenticeship recommendations for a student.

    This endpoint returns a list of apprenticeship opportunities ranked by their
    match score with the student's profile and preferences.

    The recommendations are based on various factors including:
    - Education level match
    - Location preferences
    - Sector preferences
    - Stipend requirements

    Args:
        student: Student profile and preferences
        limit: Maximum number of recommendations to return (1-100)
        include_score_components: Whether to include scoring breakdown

    Returns:
        List of recommended opportunities sorted by match score
    """
    logger.info(
        f"Getting recommendations for student with education level: {student.education_level}"
    )

    engine = BasicRecommendationEngine()
    try:
        recommendations = engine.get_recommendations(
            student=student,
            limit=limit,
            include_score_components=include_score_components,
        )
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations
    finally:
        engine.close()
