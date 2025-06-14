from typing import List, Tuple, Dict, Any
from datetime import datetime
from sqlalchemy import text
from src.models.geospatial import TreeLocation, EnvironmentalImpact, ClimateZone
from src.utils.cache import cache
from src.monitoring.logger import logger
from src.db import db
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.utils.climate_data import fetch_climate_data

class EnvironmentalService:
    def __init__(self):
        self.climate_model = RandomForestRegressor(n_estimators=100, random_state=42)

    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def get_environmental_impact(self, bbox: List[float]) -> Dict[str, Any]:
        """
        Calculate environmental impact metrics for trees within the bounding box.
        """
        try:
            # Query tree data with environmental impact metrics
            query = text("""
                SELECT 
                    t.id,
                    t.species,
                    t.height,
                    t.diameter,
                    e.carbon_sequestration,
                    e.oxygen_production,
                    e.water_filtration,
                    e.air_quality_improvement
                FROM tree_locations t
                JOIN environmental_impact e ON t.id = e.tree_id
                WHERE ST_Within(
                    t.location,
                    ST_MakeEnvelope(:minlon, :minlat, :maxlon, :maxlat, 4326)
                )
            """)

            result = db.session.execute(
                query,
                {
                    'minlon': bbox[0],
                    'minlat': bbox[1],
                    'maxlon': bbox[2],
                    'maxlat': bbox[3]
                }
            )

            # Calculate aggregate metrics
            metrics = {
                'total_trees': 0,
                'total_carbon_sequestration': 0.0,  # in kg/year
                'total_oxygen_production': 0.0,     # in kg/year
                'total_water_filtration': 0.0,      # in liters/year
                'air_quality_improvement': 0.0,     # index 0-1
                'species_diversity': 0.0,
                'average_tree_health': 0.0
            }

            species_counts = {}
            total_health = 0

            for row in result:
                metrics['total_trees'] += 1
                metrics['total_carbon_sequestration'] += row.carbon_sequestration
                metrics['total_oxygen_production'] += row.oxygen_production
                metrics['total_water_filtration'] += row.water_filtration
                metrics['air_quality_improvement'] += row.air_quality_improvement
                total_health += self._calculate_tree_health(row.height, row.diameter)
                
                species_counts[row.species] = species_counts.get(row.species, 0) + 1

            if metrics['total_trees'] > 0:
                metrics['air_quality_improvement'] /= metrics['total_trees']
                metrics['average_tree_health'] = total_health / metrics['total_trees']
                metrics['species_diversity'] = self._calculate_shannon_diversity(species_counts)

            return metrics

        except Exception as e:
            logger.error(f'Error calculating environmental impact: {str(e)}')
            raise

    @cache.memoize(timeout=7200)  # Cache for 2 hours
    def get_historical_trends(
        self,
        bbox: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Analyze historical trends in tree growth and environmental impact.
        """
        try:
            query = text("""
                WITH yearly_metrics AS (
                    SELECT 
                        date_trunc('year', measurement_date) as year,
                        AVG(height) as avg_height,
                        AVG(diameter) as avg_diameter,
                        COUNT(*) as tree_count,
                        SUM(e.carbon_sequestration) as total_carbon_seq,
                        AVG(e.air_quality_improvement) as avg_air_quality
                    FROM tree_measurements m
                    JOIN tree_locations t ON m.tree_id = t.id
                    JOIN environmental_impact e ON t.id = e.tree_id
                    WHERE 
                        measurement_date BETWEEN :start_date AND :end_date
                        AND ST_Within(
                            t.location,
                            ST_MakeEnvelope(:minlon, :minlat, :maxlon, :maxlat, 4326)
                        )
                    GROUP BY date_trunc('year', measurement_date)
                    ORDER BY year
                )
                SELECT * FROM yearly_metrics
            """)

            result = db.session.execute(
                query,
                {
                    'start_date': start_date,
                    'end_date': end_date,
                    'minlon': bbox[0],
                    'minlat': bbox[1],
                    'maxlon': bbox[2],
                    'maxlat': bbox[3]
                }
            )

            trends = {
                'years': [],
                'metrics': {
                    'avg_height': [],
                    'avg_diameter': [],
                    'tree_count': [],
                    'carbon_sequestration': [],
                    'air_quality': []
                }
            }

            for row in result:
                trends['years'].append(row.year.year)
                trends['metrics']['avg_height'].append(float(row.avg_height))
                trends['metrics']['avg_diameter'].append(float(row.avg_diameter))
                trends['metrics']['tree_count'].append(int(row.tree_count))
                trends['metrics']['carbon_sequestration'].append(float(row.total_carbon_seq))
                trends['metrics']['air_quality'].append(float(row.avg_air_quality))

            # Calculate trend lines
            for metric in trends['metrics']:
                if trends['metrics'][metric]:
                    trends[f'{metric}_trend'] = self._calculate_trend(
                        trends['years'],
                        trends['metrics'][metric]
                    )

            return trends

        except Exception as e:
            logger.error(f'Error calculating historical trends: {str(e)}')
            raise

    @cache.memoize(timeout=86400)  # Cache for 24 hours
    def get_climate_scenarios(
        self,
        bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Generate climate impact scenarios and their effects on tree populations.
        """
        try:
            # Get current tree and climate data
            query = text("""
                SELECT 
                    t.species,
                    t.height,
                    t.diameter,
                    c.temperature,
                    c.precipitation,
                    c.soil_moisture
                FROM tree_locations t
                JOIN climate_zones c ON ST_Within(t.location, c.geometry)
                WHERE ST_Within(
                    t.location,
                    ST_MakeEnvelope(:minlon, :minlat, :maxlon, :maxlat, 4326)
                )
            """)

            result = db.session.execute(
                query,
                {
                    'minlon': bbox[0],
                    'minlat': bbox[1],
                    'maxlon': bbox[2],
                    'maxlat': bbox[3]
                }
            )

            # Prepare data for modeling
            current_data = pd.DataFrame([
                {
                    'species': row.species,
                    'height': row.height,
                    'diameter': row.diameter,
                    'temperature': row.temperature,
                    'precipitation': row.precipitation,
                    'soil_moisture': row.soil_moisture
                }
                for row in result
            ])

            if current_data.empty:
                return {'error': 'No data available for climate scenarios'}

            # Get climate projections
            climate_projections = fetch_climate_data(bbox)

            scenarios = {
                'rcp45': self._model_climate_impact(current_data, climate_projections['rcp45']),
                'rcp85': self._model_climate_impact(current_data, climate_projections['rcp85'])
            }

            return scenarios

        except Exception as e:
            logger.error(f'Error generating climate scenarios: {str(e)}')
            raise

    def _calculate_tree_health(self, height: float, diameter: float) -> float:
        """
        Calculate tree health index based on height and diameter.
        Returns a value between 0 and 1.
        """
        try:
            # Simplified health calculation
            # Could be enhanced with more sophisticated metrics
            height_factor = min(1.0, height / 30.0)  # Assume 30m is optimal height
            diameter_factor = min(1.0, diameter / 100.0)  # Assume 100cm is optimal diameter
            return (height_factor + diameter_factor) / 2
        except Exception:
            return 0.0

    def _calculate_shannon_diversity(self, species_counts: Dict[str, int]) -> float:
        """
        Calculate Shannon diversity index for species diversity.
        """
        try:
            total = sum(species_counts.values())
            proportions = [count / total for count in species_counts.values()]
            return -sum(p * np.log(p) for p in proportions if p > 0)
        except Exception:
            return 0.0

    def _calculate_trend(self, years: List[int], values: List[float]) -> Dict[str, float]:
        """
        Calculate linear trend and confidence intervals.
        """
        try:
            x = np.array(years)
            y = np.array(values)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)

            # Calculate confidence intervals
            y_hat = p(x)
            y_err = y - y_hat
            mean_x = np.mean(x)
            n = len(x)
            
            if n < 2:
                return {
                    'slope': float(z[0]),
                    'intercept': float(z[1]),
                    'confidence_interval': 0
                }

            std_err = np.sqrt(np.sum(y_err**2) / (n-2)) / np.sqrt(np.sum((x-mean_x)**2))
            confidence_interval = std_err * 1.96  # 95% confidence interval

            return {
                'slope': float(z[0]),
                'intercept': float(z[1]),
                'confidence_interval': float(confidence_interval)
            }
        except Exception:
            return {
                'slope': 0.0,
                'intercept': 0.0,
                'confidence_interval': 0.0
            }

    def _model_climate_impact(
        self,
        current_data: pd.DataFrame,
        climate_projection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Model the impact of climate changes on tree populations.
        """
        try:
            # Prepare features for modeling
            features = ['temperature', 'precipitation', 'soil_moisture']
            X = current_data[features]
            y_height = current_data['height']
            y_diameter = current_data['diameter']

            # Train models
            height_model = RandomForestRegressor(n_estimators=100, random_state=42)
            diameter_model = RandomForestRegressor(n_estimators=100, random_state=42)

            height_model.fit(X, y_height)
            diameter_model.fit(X, y_diameter)

            # Prepare projection data
            X_proj = pd.DataFrame([
                {
                    'temperature': climate_projection['temperature'],
                    'precipitation': climate_projection['precipitation'],
                    'soil_moisture': climate_projection['soil_moisture']
                }
            ])

            # Make predictions
            projected_height = height_model.predict(X_proj)
            projected_diameter = diameter_model.predict(X_proj)

            # Calculate impact metrics
            height_change = (projected_height[0] - current_data['height'].mean()) / current_data['height'].mean() * 100
            diameter_change = (projected_diameter[0] - current_data['diameter'].mean()) / current_data['diameter'].mean() * 100

            return {
                'temperature_change': climate_projection['temperature'] - current_data['temperature'].mean(),
                'precipitation_change': climate_projection['precipitation'] - current_data['precipitation'].mean(),
                'soil_moisture_change': climate_projection['soil_moisture'] - current_data['soil_moisture'].mean(),
                'projected_height_change_percent': float(height_change),
                'projected_diameter_change_percent': float(diameter_change),
                'risk_level': self._calculate_risk_level(height_change, diameter_change),
                'adaptation_recommendations': self._get_adaptation_recommendations(height_change, diameter_change)
            }

        except Exception as e:
            logger.error(f'Error modeling climate impact: {str(e)}')
            return {
                'error': 'Failed to model climate impact',
                'details': str(e)
            }

    def _calculate_risk_level(self, height_change: float, diameter_change: float) -> str:
        """
        Calculate risk level based on projected changes.
        """
        avg_change = (abs(height_change) + abs(diameter_change)) / 2
        if avg_change < 5:
            return 'LOW'
        elif avg_change < 15:
            return 'MODERATE'
        else:
            return 'HIGH'

    def _get_adaptation_recommendations(self, height_change: float, diameter_change: float) -> List[str]:
        """
        Generate adaptation recommendations based on projected changes.
        """
        recommendations = []
        avg_change = (height_change + diameter_change) / 2

        if avg_change < -10:
            recommendations.extend([
                'Consider drought-resistant species',
                'Implement irrigation systems',
                'Increase mulching to retain moisture'
            ])
        elif avg_change > 10:
            recommendations.extend([
                'Monitor for pest susceptibility',
                'Implement pruning program',
                'Consider wind protection measures'
            ])

        if abs(avg_change) > 15:
            recommendations.extend([
                'Develop emergency response plan',
                'Increase monitoring frequency',
                'Consider assisted migration of vulnerable species'
            ])

        return recommendations
