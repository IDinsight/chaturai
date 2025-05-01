"""This module contains the base recommendation engine."""

# Standard Library
from abc import ABC, abstractmethod
from typing import Optional

# Third Party Library
from sqlalchemy.orm import Session

# Package Library
from naukriwaala.db.opportunity import Opportunity
from naukriwaala.db.utils import get_session
from naukriwaala.recommendation.schemas import RecommendationResult, StudentProfile


class BaseRecommendationEngine(ABC):
    """Abstract base class for recommendation engines."""

    def __init__(self, *, db_session: Optional[Session] = None):
        """Initialize the recommendation engine.

        Parameters
        ----------
        db_session
            The SQLAlchemy session to use for all database connections.
        """

        self.session = db_session or get_session()

    @abstractmethod
    def calculate_score(
        self, opportunity: Opportunity, student: StudentProfile
    ) -> tuple[float, dict[str, float] | None]:
        """Calculate match score between a student and an opportunity.

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

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        raise NotImplementedError(
            f"Need to implement {self.__class__.__name__}.calculate_score() method!"
        )

    def close(self) -> None:
        """Close the database session."""

        self.session.close()

    def get_recommendations(
        self,
        *,
        include_score_components: bool = False,
        limit: int = 10,
        student: StudentProfile,
    ) -> list[RecommendationResult]:
        """Get personalized opportunity recommendations for a student.

        Parameters
        ----------
        include_score_components
            Specifies whether to include score component breakdown.
        limit
            Maximum number of recommendations to return.
        student
            A `StudentProfile` instance representing the student.

        Returns
        -------
        list[RecommendationResult]
            List of `RecommendationResult` objects, sorted by match score.
        """

        # Get opportunities from the database.
        opportunities = Opportunity.get_active_opportunities(session=self.session)

        # Calculate scores for all opportunities.
        results = []
        for opportunity in opportunities:
            score, components = self.calculate_score(
                opportunity=opportunity, student=student
            )
            results.append(
                RecommendationResult(
                    opportunity_details=opportunity.to_dict(),
                    opportunity_id=opportunity.id,
                    score=score,
                    score_components=components if include_score_components else None,
                )
            )

        # Sort by score and limit results.
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:limit]
