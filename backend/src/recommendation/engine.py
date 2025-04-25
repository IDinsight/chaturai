from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sqlalchemy.orm import Session
from ..models.base import get_session
from ..models.opportunity import Opportunity
from enum import Enum


class Gender(str, Enum):
    """Enum for gender"""

    Male = "Male"
    Female = "Female"
    Transgender = "Transgender"


class QualificationType(str, Enum):
    """Qualification type for students"""

    academic = "academic"
    trained_under_scheme = "trained_under_scheme"


AcademicQualification = Literal[
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "ITI",
    "MSBSVET",
    "ITI Dual",
    "ITI Result Awaited",
    "Diploma Pursuing",
    "Advanced Diploma",
    "Graduate Pursuing",
    "Graduate",
    "Post Graduate",
    "Doctoral",
    "Others",
]


class AcademicQualificationOrdinal:
    """
    Defines academic qualifications and their ordinal relationships.

    This class combines the literal type definition with ordinal ranking functionality
    to provide a clean interface for qualification comparisons.
    """

    # Define all possible values in a ranks dictionary
    Ranks = {
        # School education
        "5th": 5,
        "6th": 6,
        "7th": 7,
        "8th": 8,
        "9th": 9,
        "10th": 10,
        "11th": 11,
        "12th": 12,
        # Vocational qualifications (all considered equivalent)
        "ITI": 13,
        "MSBSVET": 13,
        "ITI Dual": 13,
        "ITI Result Awaited": 13,
        # Higher education
        "Diploma Pursuing": 14,
        "Advanced Diploma": 15,
        "Graduate Pursuing": 16,
        "Graduate": 17,
        "Post Graduate": 18,
        "Doctoral": 19,
        # Special case
        "Others": 0,  # Lowest rank for unknown qualifications
    }

    @classmethod
    def meets_minimum_requirement(
        cls, student_qual: AcademicQualification, required_qual: AcademicQualification
    ) -> bool:
        """
        Check if a student's qualification meets or exceeds the required
        qualification.
        """
        return cls.Ranks[student_qual] >= cls.Ranks[required_qual]


TrainedUnderSchemesQualification = Literal[
    "NULM",
    "DDUGKY",
    "State Specific Schemes",
    "PMKVY",
    "SDI-MES",
    "Central Schemes",
    "Vocationalization of School Education (VSE)-MHRD",
    "PMKVY-MSDE",
    "DDUGKY-MoRD",
    "EST&P-NULM",
    "PMAYG-MoSJE",
    "CPWD-MoUD",
    "NSKFDC-MoSJE",
    "NBCFDC-MoSJE",
    "NSDFDC-MoSJE",
    "ISDS-MoTextiles",
    "Seekho aur Kamao-MoMA",
    "SSC-fee based courses",
]


class Qualification(BaseModel):
    """Qualification model for student profiles."""

    qualification_type: QualificationType
    qualification: AcademicQualification | TrainedUnderSchemesQualification

    @field_validator("qualification")
    def validate_qualification(
        cls, v: AcademicQualification | TrainedUnderSchemesQualification, info
    ) -> AcademicQualification | TrainedUnderSchemesQualification:
        """Validator for qualification"""
        qualification_type = info.data.get("qualification_type")

        # Check for academic qualifications
        academic_quals = set(AcademicQualificationOrdinal.Ranks.keys())
        trained_quals = set(
            [
                "NULM",
                "DDUGKY",
                "State Specific Schemes",
                "PMKVY",
                "SDI-MES",
                "Central Schemes",
                "Vocationalization of School Education (VSE)-MHRD",
                "PMKVY-MSDE",
                "DDUGKY-MoRD",
                "EST&P-NULM",
                "PMAYG-MoSJE",
                "CPWD-MoUD",
                "NSKFDC-MoSJE",
                "NBCFDC-MoSJE",
                "NSDFDC-MoSJE",
                "ISDS-MoTextiles",
                "Seekho aur Kamao-MoMA",
                "SSC-fee based courses",
            ]
        )

        if v in academic_quals:
            if qualification_type != QualificationType.academic:
                raise ValueError(
                    f"Qualification type must be {QualificationType.academic} "
                    "for academic qualifications"
                )
        elif v in trained_quals:
            if qualification_type != QualificationType.trained_under_scheme:
                raise ValueError(
                    f"Qualification type must be "
                    f"{QualificationType.trained_under_scheme} for trained "
                    "under schemes qualifications"
                )
        else:
            raise ValueError("Invalid qualification")
        return v


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
    gender: Optional[Gender] = Field(None, description="Gender of the student")

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
