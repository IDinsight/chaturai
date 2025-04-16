from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship, Session
from typing import List, Literal, Dict, Set
from .base import Base
from ..settings import DefaultSettings
from datetime import datetime
from zoneinfo import ZoneInfo


# Define the possible status values
ProcessedStatus = Literal["outdated", "filled", "open"]


class Opportunity(Base):
    """SQLAlchemy model representing an apprenticeship opportunity.

    This model stores information about apprenticeship opportunities including basic details,
    vacancy information, stipend details, and relationships with establishments.
    """

    __tablename__ = "opportunities"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    code = Column(String, unique=True)
    short_description = Column(String, nullable=True)
    naps_benefit = Column(Boolean, default=False)
    number_of_vacancies = Column(Integer)
    available_vacancies = Column(Integer)
    gender_type = Column(String)
    stipend_from = Column(Integer)
    stipend_upto = Column(Integer)
    status = Column(Boolean, default=True)
    approval_status = Column(String)
    processed_status = Column(String, default="open")

    # Metadata
    created_by = Column(String)
    updated_by = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    # Relationships
    establishment_id = Column(String, ForeignKey("establishments.id"))
    establishment = relationship("Establishment", back_populates="opportunities")

    # Store complex nested data as JSON
    course_data = Column(JSON)
    trainings_data = Column(JSON)
    locations_data = Column(JSON)

    # Tracking fields
    last_checked = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def update_from_api(self, api_data):
        """Update opportunity details from API response"""
        self.name = api_data["name"]
        self.code = api_data["code"]
        self.short_description = api_data["short_description"]
        self.naps_benefit = api_data["naps_benefit"]
        self.number_of_vacancies = api_data["number_of_vacancies"]
        self.available_vacancies = api_data["available_vacancies"]
        self.gender_type = api_data["gender_type"]
        self.stipend_from = api_data["stipend_from"]
        self.stipend_upto = api_data["stipend_upto"]
        self.status = api_data["status"]
        self.approval_status = api_data["approval_status"]
        self.created_by = api_data["created_by"]
        self.updated_by = api_data["updated_by"]
        self.processed_status = "open"

        # Parse datetime strings
        self.created_at = datetime.strptime(
            api_data["created_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
        )
        self.updated_at = datetime.strptime(
            api_data["updated_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
        )

        # Store nested data
        self.course_data = api_data.get("course")
        self.trainings_data = api_data.get("trainings")
        self.locations_data = api_data.get("locations")

        self.last_checked = datetime.now(ZoneInfo(DefaultSettings.TIMEZONE))

    def to_dict(self) -> dict:
        """Convert opportunity to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "short_description": self.short_description,
            "naps_benefit": self.naps_benefit,
            "number_of_vacancies": self.number_of_vacancies,
            "available_vacancies": self.available_vacancies,
            "gender_type": self.gender_type,
            "stipend_from": self.stipend_from,
            "stipend_upto": self.stipend_upto,
            "status": self.status,
            "approval_status": self.approval_status,
            "establishment": (
                self.establishment.establishment_name if self.establishment else None
            ),
            "course_data": self.course_data,
            "trainings_data": self.trainings_data,
            "locations_data": self.locations_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def get_active_opportunities(cls, session: Session) -> List["Opportunity"]:
        """Get all active opportunities with available vacancies."""
        return (
            session.query(cls).filter(cls.is_active, cls.available_vacancies > 0).all()
        )

    @classmethod
    def create_or_update_opportunity(
        cls, session: Session, opportunity_data: Dict, establishment_id: str
    ) -> "Opportunity":
        """Create or update an opportunity from API data.

        Args:
            session: SQLAlchemy session
            opportunity_data: Dictionary containing opportunity data from API
            establishment_id: ID of the associated establishment

        Returns:
            The created or updated Opportunity instance
        """
        opportunity = session.query(cls).filter_by(id=opportunity_data["id"]).first()

        if opportunity:
            # Update existing opportunity
            opportunity.update_from_api(opportunity_data)
        else:
            # Create new opportunity
            opportunity = cls(
                id=opportunity_data["id"], establishment_id=establishment_id
            )
            opportunity.update_from_api(opportunity_data)
            session.add(opportunity)

        return opportunity

    @classmethod
    def update_opportunity_statuses(
        cls,
        session: Session,
        updated_opportunity_ids: Set[str],
        cutoff_date: datetime,
    ) -> Dict[str, int]:
        """Update the status of opportunities based on their age and update status.

        Args:
            session: SQLAlchemy session
            updated_opportunity_ids: Set of opportunity IDs that were updated in this run
            cutoff_date: Date before which opportunities are considered outdated

        Returns:
            Dictionary with counts of opportunities marked as outdated and filled
        """
        # Mark opportunities older than cutoff date as outdated
        outdated_count = (
            session.query(cls)
            .filter(cls.created_at < cutoff_date, cls.processed_status != "outdated")
            .update({"processed_status": "outdated"}, synchronize_session=False)
        )

        # Mark opportunities not updated during this run as filled
        filled_count = (
            session.query(cls)
            .filter(
                cls.id.notin_(updated_opportunity_ids),
                cls.processed_status == "open",
                cls.created_at >= cutoff_date,
            )
            .update({"processed_status": "filled"}, synchronize_session=False)
        )

        return {"outdated": outdated_count, "filled": filled_count}


class Establishment(Base):
    """SQLAlchemy model representing an establishment offering apprenticeships.

    This model stores information about establishments that offer apprenticeship opportunities,
    including their basic details and relationships with opportunities.
    """

    __tablename__ = "establishments"

    id = Column(String, primary_key=True)
    establishment_name = Column(String, nullable=False)
    code = Column(String, unique=True)
    registration_type = Column(String)
    working_days = Column(String)
    state_count = Column(Integer)

    # Relationships
    opportunities = relationship("Opportunity", back_populates="establishment")

    @classmethod
    def create_establishment_if_not_exists(
        cls, session: Session, establishment_data: Dict
    ) -> "Establishment":
        """Create an establishment if it doesn't exist.

        Args:
            session: SQLAlchemy session
            establishment_data: Dictionary containing establishment data from API

        Returns:
            The existing or newly created Establishment instance
        """
        establishment = (
            session.query(cls).filter_by(code=establishment_data["code"]).first()
        )

        if not establishment:
            establishment = cls(
                id=establishment_data["code"],  # Using code as ID
                establishment_name=establishment_data["establishment_name"],
                code=establishment_data["code"],
                registration_type=establishment_data["registration_type"],
                working_days=establishment_data["working_days"],
                state_count=establishment_data["state_count"],
            )
            session.add(establishment)

        return establishment
