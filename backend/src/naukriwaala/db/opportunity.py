"""This module contains SQLAlchemy models for apprenticeship opportunities and
establishments.
"""

# Future Library
from __future__ import annotations

# Standard Library
from datetime import datetime
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

# Third Party Library
from sqlalchemy import JSON, Boolean, DateTime
from sqlalchemy import Enum as PgEnum
from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

# Package Library
from naukriwaala.config import Settings
from naukriwaala.db.utils import Base
from naukriwaala.recommendation.schemas import Gender

# Globals.
SCRAPER_TIMEZONE = Settings.SCRAPER_TIMEZONE

# Define the possible status values.
ProcessedStatus = Literal["outdated", "filled", "open"]


class Opportunity(Base):
    """SQLAlchemy model representing an apprenticeship opportunity.

    This model stores information about apprenticeship opportunities including basic
    details, vacancy information, stipend details, and relationships with
    establishments.
    """

    __tablename__ = "opportunities"

    approval_status: Mapped[Optional[str]] = mapped_column(String)
    available_vacancies: Mapped[Optional[int]] = mapped_column(Integer)
    code: Mapped[str] = mapped_column(String, unique=True)
    gender_type: Mapped[Optional[Gender]] = mapped_column(
        PgEnum(Gender, name="gender_enum", native_enum=False)
    )
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    naps_benefit: Mapped[bool] = mapped_column(Boolean, default=False)
    number_of_vacancies: Mapped[Optional[int]] = mapped_column(Integer)
    processed_status: Mapped[str] = mapped_column(String, default="open")
    short_description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[bool] = mapped_column(Boolean, default=True)
    stipend_from: Mapped[Optional[int]] = mapped_column(Integer)
    stipend_upto: Mapped[Optional[int]] = mapped_column(Integer)

    # Metadata.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships.
    establishment: Mapped["Establishment"] = relationship(
        "Establishment", back_populates="opportunities"
    )
    establishment_id: Mapped[str] = mapped_column(
        String, ForeignKey("establishments.id")
    )

    # Store complex nested data as JSON.
    course_data: Mapped[Optional[dict]] = mapped_column(JSON)
    locations_data: Mapped[Optional[dict]] = mapped_column(JSON)
    trainings_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # Tracking fields.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_checked: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ZoneInfo(SCRAPER_TIMEZONE)),
    )

    @classmethod
    def create_or_update_opportunity(
        cls, *, establishment_id: str, opportunity_data: dict, session: Session
    ) -> Opportunity:
        """Create or update an opportunity from API data.

        Parameters
        ----------
        establishment_id
            The ID of the associated establishment.
        opportunity_data
            Dictionary containing opportunity data from API.
        session
            A SQLAlchemy session.

        Returns
        -------
        Opportunity
            The created or updated `Opportunity` instance.
        """

        opportunity = session.query(cls).filter_by(id=opportunity_data["id"]).first()

        if opportunity:
            # Update existing opportunity.
            opportunity.update_from_api(api_data=opportunity_data)
        else:
            # Create new opportunity.
            opportunity = cls(
                id=opportunity_data["id"], establishment_id=establishment_id
            )
            opportunity.update_from_api(api_data=opportunity_data)
            session.add(opportunity)

        return opportunity

    @classmethod
    async def get_active_opportunities(
        cls, *, asession: AsyncSession
    ) -> list[Opportunity]:
        """Get all active opportunities with available vacancies.

        Parameters
        ----------
        asession
            The SQLAlchemy async session to use for all database connections.

        Returns
        -------
        list[Opportunity]
            List of active opportunities with available vacancies.
        """

        return (
            await asession.query(cls)
            .filter(cls.is_active, cls.available_vacancies > 0)
            .all()
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert opportunity to dictionary format.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the opportunity.
        """

        return {
            "approval_status": self.approval_status,
            "available_vacancies": self.available_vacancies,
            "code": self.code,
            "course_data": self.course_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "establishment": (
                self.establishment.establishment_name if self.establishment else None
            ),
            "gender_type": self.gender_type,
            "id": self.id,
            "locations_data": self.locations_data,
            "name": self.name,
            "naps_benefit": self.naps_benefit,
            "number_of_vacancies": self.number_of_vacancies,
            "short_description": self.short_description,
            "status": self.status,
            "stipend_from": self.stipend_from,
            "stipend_upto": self.stipend_upto,
            "trainings_data": self.trainings_data,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def update_from_api(self, *, api_data: dict[str, Any]) -> None:
        """Update opportunity details from API response

        Parameters
        ----------
        api_data
            Dictionary containing opportunity data from API response.
        """

        self.approval_status = api_data["approval_status"]
        self.available_vacancies = api_data["available_vacancies"]
        self.code = api_data["code"]
        self.created_by = api_data["created_by"]
        self.gender_type = api_data["gender_type"]
        self.name = api_data["name"]
        self.naps_benefit = api_data["naps_benefit"]
        self.number_of_vacancies = api_data["number_of_vacancies"]
        self.processed_status = "open"
        self.short_description = api_data["short_description"]
        self.status = api_data["status"]
        self.stipend_from = api_data["stipend_from"]
        self.stipend_upto = api_data["stipend_upto"]
        self.updated_by = api_data["updated_by"]

        # Parse datetime strings.
        self.created_at = datetime.strptime(
            api_data["created_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
        )
        self.updated_at = datetime.strptime(
            api_data["updated_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
        )

        # Store nested data.
        self.course_data = api_data.get("course")
        self.locations_data = api_data.get("locations")
        self.trainings_data = api_data.get("trainings")

        self.last_checked = datetime.now(ZoneInfo(SCRAPER_TIMEZONE))

    @classmethod
    def update_opportunity_statuses(
        cls,
        *,
        cutoff_date: datetime,
        session: Session,
        updated_opportunity_ids: set[str],
    ) -> dict[str, int]:
        """Update the status of opportunities based on their age and update status.

        Parameters
        ----------
        cutoff_date
            Date before which opportunities are considered outdated.
        session
            A SQLAlchemy session.
        updated_opportunity_ids
            Set of opportunity IDs that were updated in this run.

        Returns
        -------
        dict[str, int]
            Dictionary with counts of opportunities marked as outdated and filled.
        """

        # Mark opportunities older than cutoff date as outdated.
        outdated_count = (
            session.query(cls)
            .filter(cls.updated_at < cutoff_date, cls.processed_status != "outdated")
            .update({"processed_status": "outdated"}, synchronize_session=False)
        )

        # Mark opportunities not updated during this run as filled.
        filled_count = (
            session.query(cls)
            .filter(
                cls.id.notin_(updated_opportunity_ids),
                cls.processed_status == "open",
                cls.updated_at >= cutoff_date,
            )
            .update({"processed_status": "filled"}, synchronize_session=False)
        )

        return {"outdated": outdated_count, "filled": filled_count}


class Establishment(Base):
    """SQLAlchemy model representing an establishment offering apprenticeships.

    This model stores information about establishments that offer apprenticeship
    opportunities, including their basic details and relationships with opportunities.
    """

    __tablename__ = "establishments"

    code: Mapped[Optional[str]] = mapped_column(String, unique=True)
    establishment_name: Mapped[str] = mapped_column(String, nullable=False)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    registration_type: Mapped[Optional[str]] = mapped_column(String)
    state_count: Mapped[Optional[int]] = mapped_column(Integer)
    working_days: Mapped[Optional[str]] = mapped_column(String)

    # Relationships.
    opportunities: Mapped[list["Opportunity"]] = relationship(
        "Opportunity", back_populates="establishment"
    )

    @classmethod
    def create_establishment_if_not_exists(
        cls, *, establishment_data: dict, session: Session
    ) -> Establishment:
        """Create an establishment if it doesn't exist.

        Parameters:
        -----------
        establishment_data
            Dictionary containing establishment data from API.
        session
            A SQLAlchemy session.

        Returns
        -------
        Establishment
            The existing or newly created `Establishment` instance.
        """

        establishment = (
            session.query(cls).filter_by(code=establishment_data["code"]).first()
        )

        if not establishment:
            establishment = cls(
                code=establishment_data["code"],
                establishment_name=establishment_data["establishment_name"],
                id=establishment_data["code"],  # Using code as ID
                registration_type=establishment_data["registration_type"],
                state_count=establishment_data["state_count"],
                working_days=establishment_data["working_days"],
            )
            session.add(establishment)

        return establishment
