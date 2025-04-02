from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship, Session
from typing import List
from .base import Base
from datetime import datetime

class Opportunity(Base):
    __tablename__ = 'opportunities'

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
    
    # Metadata
    created_by = Column(String)
    updated_by = Column(String, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Relationships
    establishment_id = Column(String, ForeignKey('establishments.id'))
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
        self.name = api_data['name']
        self.code = api_data['code']
        self.short_description = api_data['short_description']
        self.naps_benefit = api_data['naps_benefit']
        self.number_of_vacancies = api_data['number_of_vacancies']
        self.available_vacancies = api_data['available_vacancies']
        self.gender_type = api_data['gender_type']
        self.stipend_from = api_data['stipend_from']
        self.stipend_upto = api_data['stipend_upto']
        self.status = api_data['status']
        self.approval_status = api_data['approval_status']
        self.created_by = api_data['created_by']
        self.updated_by = api_data['updated_by']
        
        # Parse datetime strings
        self.created_at = datetime.strptime(api_data['created_at']['date'], '%Y-%m-%d %H:%M:%S.%f')
        self.updated_at = datetime.strptime(api_data['updated_at']['date'], '%Y-%m-%d %H:%M:%S.%f')
        
        # Store nested data
        self.course_data = api_data.get('course')
        self.trainings_data = api_data.get('trainings')
        self.locations_data = api_data.get('locations')
        
        self.last_checked = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert opportunity to dictionary format."""
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'short_description': self.short_description,
            'naps_benefit': self.naps_benefit,
            'number_of_vacancies': self.number_of_vacancies,
            'available_vacancies': self.available_vacancies,
            'gender_type': self.gender_type,
            'stipend_from': self.stipend_from,
            'stipend_upto': self.stipend_upto,
            'status': self.status,
            'approval_status': self.approval_status,
            'establishment': self.establishment.establishment_name if self.establishment else None,
            'course_data': self.course_data,
            'trainings_data': self.trainings_data,
            'locations_data': self.locations_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def get_active_opportunities(cls, session: Session) -> List['Opportunity']:
        """Get all active opportunities with available vacancies."""
        return session.query(cls).filter(
            cls.is_active == True,
            cls.available_vacancies > 0
        ).all()

class Establishment(Base):
    __tablename__ = 'establishments'

    id = Column(String, primary_key=True)
    establishment_name = Column(String, nullable=False)
    code = Column(String, unique=True)
    registration_type = Column(String)
    working_days = Column(String)
    state_count = Column(Integer)
    
    # Relationships
    opportunities = relationship("Opportunity", back_populates="establishment") 