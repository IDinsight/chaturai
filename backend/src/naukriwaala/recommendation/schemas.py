"""This module contains Pydantic models for recommendation."""

# Standard Library
from enum import Enum
from typing import Any, Literal, Optional, get_args

# Third Party Library
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

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


TrainedUnderSchemesQualification = Literal[
    "Central Schemes",
    "CPWD-MoUD",
    "DDUGKY",
    "DDUGKY-MoRD",
    "EST&P-NULM",
    "ISDS-MoTextiles",
    "NBCFDC-MoSJE",
    "NSDFDC-MoSJE",
    "NSKFDC-MoSJE",
    "NULM",
    "PMAYG-MoSJE",
    "PMKVY",
    "PMKVY-MSDE",
    "SDI-MES",
    "Seekho aur Kamao-MoMA",
    "SSC-fee based courses",
    "State Specific Schemes",
    "Vocationalization of School Education (VSE)-MHRD",
]


class AcademicQualificationOrdinal:
    """Define academic qualifications and their ordinal relationships.

    This class combines the literal type definition with ordinal ranking functionality
    to provide a clean interface for qualification comparisons.
    """

    # Define all possible values in a ranks dictionary.
    ranks = {
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
        cls,
        *,
        required_qual: AcademicQualification,
        student_qual: AcademicQualification,
    ) -> bool:
        """Check if a student's qualification meets or exceeds the required
        qualification.

        Parameters
        ----------
        required_qual
            The required qualification to meet.
        student_qual
            The student's qualification to check against the required one.

        Returns
        -------
        bool
            True if the student's qualification meets or exceeds the required one,
            False otherwise.
        """

        return cls.ranks[student_qual] >= cls.ranks[required_qual]


class Gender(str, Enum):
    """Enum for genders."""

    Female = "Female"
    Male = "Male"
    Transgender = "Transgender"


class QualificationType(str, Enum):
    """Qualification type for students."""

    academic = "academic"
    trained_under_scheme = "trained_under_scheme"


QUAL_TO_TYPE: dict[str, QualificationType] = {
    **{
        q: QualificationType.academic
        for q in frozenset(AcademicQualificationOrdinal.ranks)
    },
    **{
        q: QualificationType.trained_under_scheme
        for q in frozenset(get_args(TrainedUnderSchemesQualification))
    },
}


# Pydantic models.
class Qualification(BaseModel):
    """Pydantic model for verifying qualifications for student profiles."""

    qualification: AcademicQualification | TrainedUnderSchemesQualification
    qualification_type: QualificationType

    @field_validator("qualification")
    @classmethod
    def validate_qualification(
        cls,
        v: AcademicQualification | TrainedUnderSchemesQualification,
        info: ValidationInfo,
    ) -> AcademicQualification | TrainedUnderSchemesQualification:
        """Validator for qualification.

        Parameters
        ----------
        v
            The qualification value to validate.
        info
            Metadata about the field being validated, including the context in which
            the validation is occurring.

        Returns
        -------
        AcademicQualification | TrainedUnderSchemesQualification
            The validated qualification value.

        Raises
        ------
        ValueError
            If the qualification does not match the expected type based on the
            `qualification_type` field.
        """

        expected_type = QUAL_TO_TYPE.get(v, None)
        if expected_type is None:
            raise ValueError(f"Invalid qualification: {v}")

        if info.data.get("qualification_type") is not expected_type:
            raise ValueError(
                f"Qualification type must be {expected_type.value!r} for {v!r}"
            )

        return v


class RecommendationResult(BaseModel):
    """Pydantic model for verifying recommendation results."""

    opportunity_details: dict[str, Any] = Field(
        ..., description="Full details of the opportunity"
    )
    opportunity_id: str = Field(..., description="Unique identifier of the opportunity")
    score: float = Field(..., description="Match score between 0 and 1")
    score_components: Optional[dict[str, float]] = Field(
        None, description="Optional breakdown of individual scoring components"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "opportunity_details": {
                    "name": "Electronics Mechanic",
                    "establishment": "L G Balakrishnan and Bros Ltd",
                    "stipend_from": 15000,
                    "stipend_upto": 16000,
                },
                "opportunity_id": "67e527a6f8b697fadc0b6f18",
                "score": 0.85,
                "score_components": {
                    "location": 0.8,
                    "education": 1.0,
                    "stipend": 1.0,
                    "sector": 0.6,
                },
            }
        }
    )


class StudentProfile(BaseModel):
    """Pydantic model for verifying student profile attributes for matching.

    This is a flexible container for student attributes. Any matching algorithm can use
    whichever attributes it needs and ignore the rest.
    """

    # Basic information.
    education_level: Optional[str] = Field(
        None,
        description="Highest education level completed (e.g., '10th', '12th', 'ITI')",
    )
    gender: Optional[Gender] = Field(None, description="Gender of the student")
    specialization: Optional[str] = Field(
        None,
        description="Field of study or specialization (e.g., 'Electronics', 'Mechanical')",
    )

    # Preferences.
    minimum_stipend: Optional[float] = Field(
        None,
        description="Minimum monthly stipend requirement",
        ge=0,  # Must be greater than or equal to 0
    )
    preferred_locations: list[str] = Field(
        default_factory=list,
        description="List of preferred cities or states for apprenticeship",
    )
    preferred_sectors: list[str] = Field(
        default_factory=list,
        description="List of preferred industry sectors (e.g., ['Automotive', 'Electronics'])",
    )

    # Additional attributes can be stored in a flexible dictionary.
    additional_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attributes not covered by standard fields",
    )

    @field_validator("education_level")
    @classmethod
    def validate_education_level(cls, v: Optional[str]) -> Optional[str]:
        """Validator for education level.

        Parameters
        ----------
        v
            The education level to validate.

        Returns
        -------
        Optional[str]
            The validated education level if it is one of the allowed values.

        Raises
        ------
        ValueError
            If the education level is not one of the allowed values.
        """

        if v is not None:
            valid_levels = {"8th", "10th", "12th", "ITI", "Diploma", "Graduate"}
            if v not in valid_levels:
                raise ValueError(f"Education level must be one of: {valid_levels}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "education_level": "12th",
                "gender": "Female",
                "minimum_stipend": 15000.0,
                "preferred_locations": ["Bangalore", "Mumbai"],
                "preferred_sectors": ["IT", "Electronics"],
                "specialization": "Science",
                "additional_attributes": {
                    "languages": ["English", "Hindi"],
                    "skills": ["Python", "AutoCAD"],
                    "work_experience_months": 6,
                },
            }
        }
    )


# Queries.
class RecommendationQuery(BaseModel):
    """Pydantic model to validate recommendation query parameters."""

    query_text: str
    user_id: str

    model_config = ConfigDict(from_attributes=True)


class RecommendationQueryResultsRefined(RecommendationQuery):
    """Pydantic model to validate recommendation query parameters with additional
    fields.
    """

    query_text_refined: str


# Query responses.
class RecommendationQueryResults(RecommendationQueryResultsRefined):
    """Pydantic model to validate recommendation query responses."""

    answer_response: dict[str, Any] | list[dict[str, Any]]
    session_id: int | str
