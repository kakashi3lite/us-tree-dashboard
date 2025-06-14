"""Geospatial data models using SQLAlchemy and PostGIS."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime

Base = declarative_base()

class TreeLocation(Base):
    """Model for storing tree locations with PostGIS geometry."""
    __tablename__ = 'tree_locations'

    id = Column(Integer, primary_key=True)
    species = Column(String)
    height = Column(Float)
    diameter = Column(Float)
    health_condition = Column(String)
    last_inspection = Column(DateTime)
    location = Column(Geometry('POINT', srid=4326, spatial_index=True))
    
    # Relationships
    measurements = relationship('TreeMeasurement', back_populates='tree')
    environmental_impacts = relationship('EnvironmentalImpact', back_populates='tree')

    def __repr__(self):
        return f"<TreeLocation(id={self.id}, species='{self.species}')>"

class TreeMeasurement(Base):
    """Model for storing tree measurements over time."""
    __tablename__ = 'tree_measurements'

    id = Column(Integer, primary_key=True)
    tree_id = Column(Integer, ForeignKey('tree_locations.id'))
    measurement_date = Column(DateTime, default=datetime.utcnow)
    height = Column(Float)
    diameter = Column(Float)
    canopy_width = Column(Float)
    health_score = Column(Float)
    
    # Relationships
    tree = relationship('TreeLocation', back_populates='measurements')

    def __repr__(self):
        return f"<TreeMeasurement(tree_id={self.tree_id}, date='{self.measurement_date}')>"

class EnvironmentalImpact(Base):
    """Model for storing environmental impact calculations."""
    __tablename__ = 'environmental_impacts'

    id = Column(Integer, primary_key=True)
    tree_id = Column(Integer, ForeignKey('tree_locations.id'))
    calculation_date = Column(DateTime, default=datetime.utcnow)
    co2_absorption = Column(Float)  # kg/year
    oxygen_production = Column(Float)  # kg/year
    water_filtration = Column(Float)  # liters/year
    energy_savings = Column(Float)  # kWh/year
    
    # Relationships
    tree = relationship('TreeLocation', back_populates='environmental_impacts')

    def __repr__(self):
        return f"<EnvironmentalImpact(tree_id={self.tree_id}, date='{self.calculation_date}')>"

class Region(Base):
    """Model for storing geographical regions with PostGIS polygons."""
    __tablename__ = 'regions'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)  # e.g., 'city', 'county', 'state'
    population = Column(Integer)
    boundary = Column(Geometry('MULTIPOLYGON', srid=4326, spatial_index=True))
    
    def __repr__(self):
        return f"<Region(id={self.id}, name='{self.name}', type='{self.type}')>"

class ClimateZone(Base):
    """Model for storing climate zones with PostGIS polygons."""
    __tablename__ = 'climate_zones'

    id = Column(Integer, primary_key=True)
    zone_name = Column(String)
    temperature_range = Column(String)
    precipitation_range = Column(String)
    boundary = Column(Geometry('MULTIPOLYGON', srid=4326, spatial_index=True))
    
    def __repr__(self):
        return f"<ClimateZone(id={self.id}, name='{self.zone_name}')>"