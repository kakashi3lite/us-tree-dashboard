from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from redis import Redis
from sqlalchemy import text
from sqlalchemy.sql.expression import func
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsGeoJSON, ST_Transform

from src.models.geospatial import TreeLocation, TreeMeasurement, EnvironmentalImpact, Region, ClimateZone
from src.config.settings import REDIS_HOST, REDIS_PORT, CACHE_TTL

class GeospatialService:
    def __init__(self):
        self.redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.cache_ttl = CACHE_TTL

    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key based on the prefix and parameters."""
        sorted_params = sorted(kwargs.items())
        params_str = '_'.join(f"{k}:{v}" for k, v in sorted_params)
        return f"tree_dashboard:{prefix}:{params_str}"

    def _cache_data(self, key: str, data: dict) -> None:
        """Cache data with the specified key."""
        self.redis_client.setex(key, self.cache_ttl, json.dumps(data))

    def _get_cached_data(self, key: str) -> Optional[dict]:
        """Retrieve cached data for the specified key."""
        data = self.redis_client.get(key)
        return json.loads(data) if data else None

    async def get_tree_clusters(self, bounds: Dict[str, float], zoom_level: int) -> List[Dict]:
        """Get clustered tree locations within the specified bounds."""
        cache_key = self._get_cache_key('tree_clusters', bounds=json.dumps(bounds), zoom=zoom_level)
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        cluster_radius = self._get_cluster_radius(zoom_level)
        query = text("""
            WITH clusters AS (
                SELECT 
                    ST_ClusterDBSCAN(location, eps := :radius, minpoints := 2) over () as cluster_id,
                    location,
                    species,
                    health_condition
                FROM tree_locations
                WHERE ST_Within(
                    location,
                    ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
                )
            )
            SELECT 
                ST_AsGeoJSON(ST_Centroid(ST_Collect(location))) as centroid,
                COUNT(*) as tree_count,
                array_agg(DISTINCT species) as species_list,
                array_agg(DISTINCT health_condition) as health_conditions
            FROM clusters
            GROUP BY cluster_id;
        """)

        result = await self._execute_query(query, {
            'radius': cluster_radius,
            'min_lon': bounds['min_lon'],
            'min_lat': bounds['min_lat'],
            'max_lon': bounds['max_lon'],
            'max_lat': bounds['max_lat']
        })

        clusters = self._process_cluster_results(result)
        self._cache_data(cache_key, clusters)
        return clusters

    async def get_tree_density(self, region_id: Optional[int] = None) -> List[Dict]:
        """Get tree density statistics by region."""
        cache_key = self._get_cache_key('tree_density', region_id=region_id)
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        query = text("""
            SELECT * FROM tree_density_by_region
            WHERE (:region_id IS NULL OR region_id = :region_id)
            ORDER BY trees_per_km2 DESC;
        """)

        result = await self._execute_query(query, {'region_id': region_id})
        density_data = self._process_density_results(result)
        self._cache_data(cache_key, density_data)
        return density_data

    async def get_environmental_impact(self, region_id: Optional[int] = None, 
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> Dict:
        """Calculate environmental impact metrics for trees in a region."""
        cache_key = self._get_cache_key('environmental_impact', 
                                      region_id=region_id,
                                      start_date=start_date.isoformat() if start_date else None,
                                      end_date=end_date.isoformat() if end_date else None)
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        query = text("""
            WITH tree_impacts AS (
                SELECT 
                    ei.co2_absorption,
                    ei.oxygen_production,
                    ei.water_filtration,
                    t.species,
                    t.health_condition
                FROM environmental_impacts ei
                JOIN tree_locations t ON ei.tree_id = t.id
                WHERE 
                    (:region_id IS NULL OR 
                     ST_Contains((SELECT boundary FROM regions WHERE id = :region_id), t.location))
                    AND (:start_date IS NULL OR ei.calculation_date >= :start_date)
                    AND (:end_date IS NULL OR ei.calculation_date <= :end_date)
            )
            SELECT 
                SUM(co2_absorption) as total_co2_absorbed,
                SUM(oxygen_production) as total_oxygen_produced,
                SUM(water_filtration) as total_water_filtered,
                COUNT(DISTINCT species) as unique_species,
                json_object_agg(health_condition, COUNT(*)) as health_distribution
            FROM tree_impacts;
        """)

        result = await self._execute_query(query, {
            'region_id': region_id,
            'start_date': start_date,
            'end_date': end_date or datetime.now()
        })

        impact_data = self._process_impact_results(result)
        self._cache_data(cache_key, impact_data)
        return impact_data

    async def get_historical_trends(self, metric: str, 
                                  interval: str = 'month',
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> List[Dict]:
        """Get historical trends for specified metrics."""
        cache_key = self._get_cache_key('historical_trends',
                                      metric=metric,
                                      interval=interval,
                                      start_date=start_date.isoformat() if start_date else None,
                                      end_date=end_date.isoformat() if end_date else None)
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        interval_sql = self._get_interval_sql(interval)
        query = text(f"""
            WITH time_series AS (
                SELECT 
                    date_trunc(:interval, measurement_date) as period,
                    AVG(CASE 
                        WHEN :metric = 'height' THEN height
                        WHEN :metric = 'diameter' THEN diameter
                        WHEN :metric = 'canopy_width' THEN canopy_width
                    END) as avg_value,
                    COUNT(*) as measurement_count
                FROM tree_measurements
                WHERE 
                    measurement_date >= :start_date
                    AND measurement_date <= :end_date
                GROUP BY period
                ORDER BY period
            )
            SELECT 
                period,
                avg_value,
                measurement_count,
                LAG(avg_value) OVER (ORDER BY period) as prev_value
            FROM time_series;
        """)

        result = await self._execute_query(query, {
            'interval': interval,
            'metric': metric,
            'start_date': start_date or datetime.now() - timedelta(days=365),
            'end_date': end_date or datetime.now()
        })

        trend_data = self._process_trend_results(result)
        self._cache_data(cache_key, trend_data)
        return trend_data

    def _get_cluster_radius(self, zoom_level: int) -> float:
        """Calculate clustering radius based on zoom level."""
        base_radius = 0.1  # degrees
        return base_radius * (2 ** (15 - zoom_level))

    def _get_interval_sql(self, interval: str) -> str:
        """Get SQL interval string based on specified interval."""
        intervals = {
            'day': 'day',
            'week': 'week',
            'month': 'month',
            'quarter': 'quarter',
            'year': 'year'
        }
        return intervals.get(interval.lower(), 'month')

    async def _execute_query(self, query: text, params: Dict) -> List[Dict]:
        """Execute SQL query with parameters."""
        # Implementation depends on your database connection setup
        pass

    def _process_cluster_results(self, results: List[Dict]) -> List[Dict]:
        """Process raw cluster query results into formatted response."""
        return [{
            'centroid': json.loads(row['centroid']),
            'tree_count': row['tree_count'],
            'species': row['species_list'],
            'health_conditions': row['health_conditions']
        } for row in results]

    def _process_density_results(self, results: List[Dict]) -> List[Dict]:
        """Process raw density query results into formatted response."""
        return [{
            'region_id': row['region_id'],
            'region_name': row['region_name'],
            'tree_count': row['tree_count'],
            'area_km2': float(row['area_km2']),
            'trees_per_km2': float(row['trees_per_km2'])
        } for row in results]

    def _process_impact_results(self, results: List[Dict]) -> Dict:
        """Process raw environmental impact results into formatted response."""
        row = results[0] if results else {}
        return {
            'total_co2_absorbed': float(row.get('total_co2_absorbed', 0)),
            'total_oxygen_produced': float(row.get('total_oxygen_produced', 0)),
            'total_water_filtered': float(row.get('total_water_filtered', 0)),
            'unique_species': int(row.get('unique_species', 0)),
            'health_distribution': json.loads(row.get('health_distribution', '{}'))
        }

    def _process_trend_results(self, results: List[Dict]) -> List[Dict]:
        """Process raw trend query results into formatted response."""
        return [{
            'period': row['period'].isoformat(),
            'value': float(row['avg_value']),
            'count': int(row['measurement_count']),
            'change_percent': (
                ((float(row['avg_value']) - float(row['prev_value'])) / float(row['prev_value']) * 100)
                if row['prev_value'] is not None else 0
            )
        } for row in results]