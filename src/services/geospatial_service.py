"""Geospatial data service with optimized PostGIS queries and caching."""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from geoalchemy2.functions import ST_Contains, ST_Distance, ST_Transform
from geoalchemy2.elements import WKTElement
from redis import Redis
import json
from ..models.geospatial import (
    TreeLocation, TreeMeasurement, EnvironmentalImpact,
    Region, ClimateZone, Base
)

class GeospatialService:
    """Service for handling geospatial data operations with caching."""

    def __init__(self, db_url: str, redis_url: Optional[str] = None):
        """Initialize service with database and cache connections."""
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Redis cache if URL provided
        self.redis = Redis.from_url(redis_url) if redis_url else None
        self.cache_ttl = 3600  # 1 hour default TTL

    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key based on query parameters."""
        key_parts = [prefix]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()) if v is not None)
        return "geo:" + ":".join(key_parts)

    def get_trees_in_region(self, 
                           region_id: int,
                           species: Optional[str] = None,
                           limit: int = 1000,
                           offset: int = 0) -> List[Dict]:
        """Get trees within a region with pagination and caching."""
        cache_key = self._get_cache_key('trees_region',
                                      region_id=region_id,
                                      species=species,
                                      limit=limit,
                                      offset=offset)

        # Try cache first
        if self.redis:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        session = self.Session()
        try:
            # Get region boundary
            region = session.query(Region).get(region_id)
            if not region:
                return []

            # Build query with spatial index
            query = session.query(TreeLocation).filter(
                ST_Contains(region.boundary, TreeLocation.location)
            )

            # Apply species filter if specified
            if species:
                query = query.filter(TreeLocation.species == species)

            # Apply pagination
            trees = query.limit(limit).offset(offset).all()
            
            # Convert to dictionary format
            result = [{
                'id': tree.id,
                'species': tree.species,
                'height': tree.height,
                'diameter': tree.diameter,
                'health_condition': tree.health_condition,
                'coordinates': session.scalar(
                    func.ST_AsGeoJSON(tree.location)
                )
            } for tree in trees]

            # Cache the results
            if self.redis:
                self.redis.setex(cache_key, self.cache_ttl, json.dumps(result))

            return result
        finally:
            session.close()

    def get_environmental_impact_by_region(self,
                                         region_id: int,
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None) -> Dict:
        """Calculate aggregated environmental impact for a region."""
        cache_key = self._get_cache_key('env_impact',
                                      region_id=region_id,
                                      start_date=start_date.isoformat() if start_date else None,
                                      end_date=end_date.isoformat() if end_date else None)

        # Try cache first
        if self.redis:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        session = self.Session()
        try:
            # Get region boundary
            region = session.query(Region).get(region_id)
            if not region:
                return {}

            # Build base query with spatial join
            query = session.query(
                func.sum(EnvironmentalImpact.co2_absorption).label('total_co2'),
                func.sum(EnvironmentalImpact.oxygen_production).label('total_oxygen'),
                func.sum(EnvironmentalImpact.water_filtration).label('total_water'),
                func.sum(EnvironmentalImpact.energy_savings).label('total_energy')
            ).join(TreeLocation).filter(
                ST_Contains(region.boundary, TreeLocation.location)
            )

            # Apply date filters
            if start_date:
                query = query.filter(EnvironmentalImpact.calculation_date >= start_date)
            if end_date:
                query = query.filter(EnvironmentalImpact.calculation_date <= end_date)

            result = query.first()
            
            impact_data = {
                'total_co2_absorption': float(result.total_co2 or 0),
                'total_oxygen_production': float(result.total_oxygen or 0),
                'total_water_filtration': float(result.total_water or 0),
                'total_energy_savings': float(result.total_energy or 0)
            }

            # Cache the results
            if self.redis:
                self.redis.setex(cache_key, self.cache_ttl, json.dumps(impact_data))

            return impact_data
        finally:
            session.close()

    def get_tree_density_by_climate_zone(self) -> List[Dict]:
        """Calculate tree density for each climate zone."""
        cache_key = self._get_cache_key('tree_density_climate')

        # Try cache first
        if self.redis:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        session = self.Session()
        try:
            # Calculate tree density using PostGIS
            query = session.query(
                ClimateZone.zone_name,
                ClimateZone.temperature_range,
                func.count(TreeLocation.id).label('tree_count'),
                func.ST_Area(func.ST_Transform(ClimateZone.boundary, 3857)).label('area_m2')
            ).outerjoin(
                TreeLocation,
                ST_Contains(ClimateZone.boundary, TreeLocation.location)
            ).group_by(ClimateZone.id)

            results = query.all()
            
            density_data = [{
                'zone_name': result.zone_name,
                'temperature_range': result.temperature_range,
                'tree_count': result.tree_count,
                'density_per_km2': result.tree_count / (result.area_m2 / 1000000)
            } for result in results]

            # Cache the results
            if self.redis:
                self.redis.setex(cache_key, self.cache_ttl, json.dumps(density_data))

            return density_data
        finally:
            session.close()