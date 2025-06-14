"""Geospatial service layer for efficient spatial data operations and analysis."""

from typing import Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, box
from sqlalchemy import create_engine, text
import redis
import json
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from ..metrics import collect_latency_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpatialQueryConfig:
    """Configuration for spatial queries."""
    max_results: int = 1000
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    use_cache: bool = True
    parallel_processing: bool = True
    chunk_size: int = 100

class GeospatialService:
    """Service class for handling geospatial operations with PostGIS and Redis caching."""

    def __init__(self, postgis_url: str, redis_host: str = 'localhost', redis_port: int = 6379):
        """Initialize the service with database and cache connections."""
        self.postgis_engine = create_engine(postgis_url)
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True
        )
        self.query_config = SpatialQueryConfig()

    @collect_latency_metrics
    def get_trees_in_radius(self, lat: float, lon: float, radius_km: float) -> Dict[str, any]:
        """Get trees within a specified radius with caching."""
        cache_key = f"trees_radius_{lat}_{lon}_{radius_km}"
        
        if self.query_config.use_cache:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

        query = text("""
            WITH nearby_trees AS (
                SELECT 
                    id,
                    species,
                    diameter,
                    height,
                    health,
                    ST_AsGeoJSON(geom) as geometry,
                    ST_Distance(
                        geom,
                        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
                    ) as distance
                FROM tree_inventory
                WHERE ST_DWithin(
                    geom,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                    :radius
                )
                ORDER BY distance
                LIMIT :limit
            )
            SELECT 
                json_build_object(
                    'type', 'FeatureCollection',
                    'features', json_agg(json_build_object(
                        'type', 'Feature',
                        'geometry', geometry::json,
                        'properties', json_build_object(
                            'id', id,
                            'species', species,
                            'diameter', diameter,
                            'height', height,
                            'health', health,
                            'distance', distance
                        )
                    ))
                ) as geojson
            FROM nearby_trees;
        """)

        with self.postgis_engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    'lat': lat,
                    'lon': lon,
                    'radius': radius_km * 1000,  # Convert to meters
                    'limit': self.query_config.max_results
                }
            ).scalar()

        if self.query_config.use_cache:
            self.redis_client.setex(cache_key, self.query_config.cache_ttl, json.dumps(result))

        return result

    @collect_latency_metrics
    def analyze_species_distribution(self, bbox: Tuple[float, float, float, float]) -> Dict[str, any]:
        """Analyze species distribution within a bounding box."""
        cache_key = f"species_dist_{'_'.join(map(str, bbox))}"
        
        if self.query_config.use_cache:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

        query = text("""
            WITH species_stats AS (
                SELECT 
                    species,
                    COUNT(*) as count,
                    AVG(diameter) as avg_diameter,
                    AVG(height) as avg_height,
                    json_agg(json_build_object(
                        'health', health,
                        'count', COUNT(*)
                    )) as health_distribution
                FROM tree_inventory
                WHERE ST_Within(
                    geom,
                    ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 4326)
                )
                GROUP BY species
            )
            SELECT json_build_object(
                'total_trees', (SELECT COUNT(*) FROM tree_inventory),
                'species_distribution', json_agg(json_build_object(
                    'species', species,
                    'count', count,
                    'avg_diameter', avg_diameter,
                    'avg_height', avg_height,
                    'health_distribution', health_distribution
                ))
            ) as stats
            FROM species_stats;
        """)

        with self.postgis_engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3]
                }
            ).scalar()

        if self.query_config.use_cache:
            self.redis_client.setex(cache_key, self.query_config.cache_ttl, json.dumps(result))

        return result

    @collect_latency_metrics
    def calculate_environmental_impact(self, region_id: int) -> Dict[str, any]:
        """Calculate environmental impact metrics for a region."""
        cache_key = f"env_impact_{region_id}"
        
        if self.query_config.use_cache:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

        query = text("""
            WITH region_trees AS (
                SELECT 
                    t.*,
                    c.temperature,
                    c.precipitation
                FROM tree_inventory t
                JOIN regions r ON ST_Within(t.geom, r.geom)
                LEFT JOIN LATERAL (
                    SELECT temperature, precipitation
                    FROM climate_data cd
                    WHERE ST_DWithin(t.geom, cd.geom, 1000)
                    ORDER BY t.geom <-> cd.geom
                    LIMIT 1
                ) c ON true
                WHERE r.id = :region_id
            )
            SELECT json_build_object(
                'tree_metrics', json_build_object(
                    'total_trees', COUNT(*),
                    'avg_diameter', AVG(diameter),
                    'total_canopy_area', SUM(PI() * POWER(diameter/2, 2)),
                    'species_diversity', COUNT(DISTINCT species)
                ),
                'environmental_metrics', json_build_object(
                    'avg_temperature', AVG(temperature),
                    'total_precipitation', SUM(precipitation),
                    'carbon_sequestration', SUM(POWER(diameter, 2) * 0.15)
                )
            ) as impact
            FROM region_trees;
        """)

        with self.postgis_engine.connect() as conn:
            result = conn.execute(query, {'region_id': region_id}).scalar()

        if self.query_config.use_cache:
            self.redis_client.setex(cache_key, self.query_config.cache_ttl, json.dumps(result))

        return result

    @collect_latency_metrics
    def get_tree_density_heatmap(self, bbox: Tuple[float, float, float, float], grid_size_km: float = 1.0) -> Dict[str, any]:
        """Generate tree density heatmap with climate correlation."""
        cache_key = f"density_heatmap_{'_'.join(map(str, bbox))}_{grid_size_km}"
        
        if self.query_config.use_cache:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

        query = text("""
            WITH grid AS (
                SELECT (ST_SquareGrid(
                    :grid_size,
                    ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 4326)
                )).*
            ),
            grid_stats AS (
                SELECT 
                    g.geom,
                    COUNT(t.*) as tree_count,
                    AVG(c.temperature) as avg_temperature,
                    AVG(c.precipitation) as avg_precipitation
                FROM grid g
                LEFT JOIN tree_inventory t ON ST_Within(t.geom, g.geom)
                LEFT JOIN climate_data c ON ST_Within(c.geom, g.geom)
                GROUP BY g.geom
            )
            SELECT json_build_object(
                'type', 'FeatureCollection',
                'features', json_agg(json_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(geom)::json,
                    'properties', json_build_object(
                        'tree_count', tree_count,
                        'avg_temperature', avg_temperature,
                        'avg_precipitation', avg_precipitation
                    )
                ))
            ) as heatmap
            FROM grid_stats;
        """)

        with self.postgis_engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    'grid_size': grid_size_km * 1000,  # Convert to meters
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3]
                }
            ).scalar()

        if self.query_config.use_cache:
            self.redis_client.setex(cache_key, self.query_config.cache_ttl, json.dumps(result))

        return result
