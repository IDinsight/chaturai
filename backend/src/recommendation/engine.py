from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sqlalchemy.orm import Session
from ..models.base import get_session
from ..models.opportunity import Opportunity


class StudentProfile(BaseModel):
    """Student profile containing attributes for matching.

    This is a flexible container for student attributes. Any matching algorithm
    can use whichever attributes it needs and ignore the rest.
    """

    # Basic information
    education_level: Optional[str] = Field(
        None,
        description="Highest education level completed (e.g., '10th', '12th', 'ITI')",
    )
    specialization: Optional[str] = Field(
        None,
        description="Field of study or specialization (e.g., 'Electronics', 'Mechanical')",
    )
    gender: Optional[str] = Field(None, description="Gender of the student")

    # Preferences
    preferred_locations: List[str] = Field(
        default_factory=list,
        description="List of preferred cities or states for apprenticeship",
    )
    preferred_sectors: List[str] = Field(
        default_factory=list,
        description="List of preferred industry sectors (e.g., ['Automotive', 'Electronics'])",
    )
    minimum_stipend: Optional[float] = Field(
        None,
        description="Minimum monthly stipend requirement",
        ge=0,  # Must be greater than or equal to 0
    )

    # Additional attributes can be stored in a flexible dictionary
    additional_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes not covered by standard fields",
    )

    @field_validator("education_level")
    def validate_education_level(cls, v: Optional[str]) -> Optional[str]:
        """Validator for education level"""
        if v is not None:
            valid_levels = {"8th", "10th", "12th", "ITI", "Diploma", "Graduate"}
            if v not in valid_levels:
                raise ValueError(f"Education level must be one of {valid_levels}")
        return v

    @field_validator("gender")
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        """Validator for gender"""
        if v is not None:
            valid_genders = {"Male", "Female", "Other"}
            if v not in valid_genders:
                raise ValueError(f"Gender must be one of {valid_genders}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "education_level": "12th",
                "specialization": "Science",
                "gender": "Female",
                "preferred_locations": ["Bangalore", "Mumbai"],
                "preferred_sectors": ["IT", "Electronics"],
                "minimum_stipend": 15000.0,
                "additional_attributes": {
                    "languages": ["English", "Hindi"],
                    "skills": ["Python", "AutoCAD"],
                    "work_experience_months": 6,
                },
            }
        }
    )


class RecommendationResult(BaseModel):
    """Container for recommendation results."""

    opportunity_id: str = Field(..., description="Unique identifier of the opportunity")
    score: float = Field(..., description="Match score between 0 and 1")
    opportunity_details: Dict[str, Any] = Field(
        ..., description="Full details of the opportunity"
    )
    score_components: Optional[Dict[str, float]] = Field(
        None, description="Optional breakdown of individual scoring components"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "opportunity_id": "67e527a6f8b697fadc0b6f18",
                "score": 0.85,
                "opportunity_details": {
                    "name": "Electronics Mechanic",
                    "establishment": "L G Balakrishnan and Bros Ltd",
                    "stipend_from": 15000,
                    "stipend_upto": 16000,
                },
                "score_components": {
                    "location": 0.8,
                    "education": 1.0,
                    "stipend": 1.0,
                    "sector": 0.6,
                },
            }
        }
    )


class BaseRecommendationEngine(ABC):
    """Abstract base class for recommendation engines."""

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the recommendation engine.

        Args:
            db_session: SQLAlchemy session. If None, creates a new session.
        """
        self.session = db_session or get_session()

    @abstractmethod
    def calculate_score(
        self, student: StudentProfile, opportunity: Opportunity
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Calculate match score between a student and an opportunity.

        Args:
            student: StudentProfile instance
            opportunity: Opportunity instance

        Returns:
            Tuple of (overall_score, score_components)
            - overall_score: float between 0 and 1
            - score_components: Optional dictionary of score components and their values
        """
        pass

    def get_recommendations(
        self,
        student: StudentProfile,
        limit: int = 10,
        include_score_components: bool = False,
    ) -> List[RecommendationResult]:
        """Get personalized opportunity recommendations for a student.

        Args:
            student: StudentProfile instance
            limit: Maximum number of recommendations to return
            include_score_components: Whether to include score component breakdown

        Returns:
            List of RecommendationResult objects, sorted by match score
        """
        # Get opportunities from the database
        opportunities = Opportunity.get_active_opportunities(self.session)

        # Calculate scores for all opportunities
        results = []
        for opportunity in opportunities:
            score, components = self.calculate_score(student, opportunity)

            results.append(
                RecommendationResult(
                    opportunity_id=opportunity.id,
                    score=score,
                    opportunity_details=opportunity.to_dict(),
                    score_components=components if include_score_components else None,
                )
            )

        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def close(self):
        """Close the database session."""
        self.session.close()
