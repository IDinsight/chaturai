"""This module contains the basic recommendation engine."""

# Package Library
from chaturai.db.opportunity import Opportunity
from chaturai.recommendation.base import BaseRecommendationEngine
from chaturai.recommendation.schemas import StudentProfile


class BasicRecommendationEngine(BaseRecommendationEngine):
    """A basic implementation of the recommendation engine.

    This implementation uses simple matching rules based on:

    1. Location match
    2. Stipend match
    3. Education level match
    4. Sector match

    Each factor contributes equally to the final score.
    """

    @staticmethod
    def _calculate_education_match(
        *, opportunity: Opportunity, student: StudentProfile
    ) -> float:
        """Calculate education level match score.

        Parameters
        ----------
        opportunity
            A `Opportunity` instance representing the opportunity.
        student
            A `StudentProfile` instance representing the student.

        Returns
        -------
        float
            A float between 0 and 1 representing the match score based on education
            level. 1.0 means perfect match, 0.0 means no match.
        """

        if not student.education_level:
            return 1.0

        if (
            not opportunity.course_data
            or "minimum_qualification" not in opportunity.course_data
        ):
            return 0.0

        # Extract education levels from opportunity.
        education_levels = []
        for qual in opportunity.course_data["minimum_qualification"]:
            if qual.get("qualification_type", {}).get("title", None):
                education_levels.append(qual["qualification_type"]["title"].lower())

        return 1.0 if student.education_level.lower() in education_levels else 0.0

    @staticmethod
    def _calculate_location_match(
        *, opportunity: Opportunity, student: StudentProfile
    ) -> float:
        """Calculate location match score.

        Parameters
        ----------
        opportunity
            A `Opportunity` instance representing the opportunity.
        student
            A `StudentProfile` instance representing the student.

        Returns
        -------
        float
            A float between 0 and 1 representing the match score based on location
            preferences. 1.0 means perfect match, 0.0 means no match.
        """

        if not student.preferred_locations:
            return 1.0  # No preference means all locations are acceptable

        # Extract location names from opportunity.
        opportunity_locations = []
        for location in opportunity.locations_data or []:
            if "address" in location:
                address = location["address"]
                if "city" in address:
                    opportunity_locations.append(address["city"].lower())
                if "state_name" in address:
                    opportunity_locations.append(address["state_name"].lower())

        # Check if any preferred location matches.
        student_locations = [loc.lower() for loc in student.preferred_locations]
        matching_locations = set(student_locations) & set(opportunity_locations)

        return (
            len(matching_locations) / len(student.preferred_locations)
            if student.preferred_locations
            else 0.0
        )

    @staticmethod
    def _calculate_sector_match(
        *, opportunity: Opportunity, student: StudentProfile
    ) -> float:
        """Calculate sector match score.

        Parameters
        ----------
        opportunity
            A `Opportunity` instance representing the opportunity.
        student
            A `StudentProfile` instance representing the student.

        Returns
        -------
        float
            A float between 0 and 1 representing the match score based on sector
            preferences. 1.0 means perfect match, 0.0 means no match.
        """

        if not student.preferred_sectors:
            return 1.0

        if not opportunity.course_data or "sector" not in opportunity.course_data:
            return 0.0

        opportunity_sector = opportunity.course_data["sector"]["name"].lower()
        student_sectors = [sector.lower() for sector in student.preferred_sectors]

        return 1.0 if opportunity_sector in student_sectors else 0.0

    @staticmethod
    def _calculate_stipend_match(
        *, opportunity: Opportunity, student: StudentProfile
    ) -> float:
        """Calculate stipend match score.

        Parameters
        ----------
        opportunity
            A `Opportunity` instance representing the opportunity.
        student
            A `StudentProfile` instance representing the student.

        Returns
        -------
        float
            A float between 0 and 1 representing the match score based on stipend
            preferences. 1.0 means perfect match, 0.0 means no match.
        """

        if student.minimum_stipend is None:
            return 1.0

        if opportunity.stipend_from is None:
            return 0.0

        if opportunity.stipend_from >= student.minimum_stipend:
            return 1.0

        return 0.0

    def calculate_score(
        self, *, opportunity: Opportunity, student: StudentProfile
    ) -> tuple[float, dict[str, float] | None]:
        """Calculate overall match score and component scores between a student and an
        opportunity.

        Parameters
        ----------
        opportunity
            A `Opportunity` instance representing the opportunity.
        student
            A `StudentProfile` instance representing the student.

        Returns
        -------
        tuple[float, dict[str, float] | None]
            - overall_score: float between 0 and 1.
            - score_components: Optional dictionary of score components and their
                values.
        """

        components = {
            "education": self._calculate_education_match(
                opportunity=opportunity, student=student
            ),
            "location": self._calculate_location_match(
                opportunity=opportunity, student=student
            ),
            "sector": self._calculate_sector_match(
                opportunity=opportunity, student=student
            ),
            "stipend": self._calculate_stipend_match(
                opportunity=opportunity, student=student
            ),
        }

        # Calculate overall score as average of components.
        overall_score = sum(components.values()) / len(components)

        return overall_score, components
