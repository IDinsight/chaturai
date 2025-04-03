from fastapi import FastAPI, Depends, Query
from typing import List, Optional
from ..recommendation.engine import StudentProfile, RecommendationResult
from ..recommendation.implementations.basic_engine import BasicRecommendationEngine
from .auth import get_api_key
from .config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Naukriwaala API",
    description="API for getting personalized apprenticeship recommendations",
    version="1.0.0",
    root_path=settings.BACKEND_ROOT_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get(
    "/recommendations/",
    response_model=List[RecommendationResult],
    dependencies=[Depends(get_api_key)],
    tags=["recommendations"]
)
async def get_recommendations(
    student: StudentProfile,
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of recommendations to return"
    ),
    include_score_components: bool = Query(
        default=False,
        description="Whether to include detailed scoring components in the response"
    )
) -> List[RecommendationResult]:
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
    logger.info(f"Getting recommendations for student with education level: {student.education_level}")
    
    engine = BasicRecommendationEngine()
    try:
        recommendations = engine.get_recommendations(
            student=student,
            limit=limit,
            include_score_components=include_score_components
        )
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations
    finally:
        engine.close()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 