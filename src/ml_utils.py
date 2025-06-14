"""Machine Learning utilities for tree data analysis and prediction.
This module provides reusable ML components for tree health prediction,
growth forecasting, species classification, and geospatial analysis."""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
import joblib
import json
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
import logging
from datetime import datetime
from uuid import uuid4
from openai import OpenAI
import redis
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# Import metrics components
from .metrics import (
    metrics_store,
    ModelMetrics,
    BaseMetric,
    ModelTrainingMetrics,
    EnvironmentalImpactMetrics,
    calculate_confidence_intervals,
    collect_latency_metrics,
    MetricConfig,
    MetricSeries
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

# Initialize PostGIS connection
postgis_engine = create_engine(os.getenv('POSTGIS_URL', 'postgresql://user:pass@localhost:5432/trees'))

class TreeAnalyzer:
    """A class to analyze tree data using various machine learning techniques."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.health_model = None
        self.growth_model = None
        self.species_distribution_model = None
        self._load_models()

    def _load_models(self):
        if self.model_path:
            models_dir = Path(self.model_path)
            for model_file in ['health_predictor.joblib', 'growth_forecaster.joblib', 'species_distribution.joblib']:
                model_path = models_dir / model_file
                if model_path.exists():
                    setattr(self, model_file.split('.')[0], joblib.load(model_path))

    def predict_species_distribution(self, climate_data: pd.DataFrame, soil_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict tree species distribution based on climate and soil data."""
        cache_key = f"species_dist_{hash(str(climate_data) + str(soil_data))}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            return json.loads(cached_result)

        features = self._prepare_environmental_features(climate_data, soil_data)
        predictions = self.species_distribution_model.predict_proba(features)
        species_probabilities = dict(zip(self.species_distribution_model.classes_, predictions[0]))

        result = {
            'predictions': species_probabilities,
            'confidence_intervals': calculate_confidence_intervals(predictions),
            'timestamp': datetime.now().isoformat()
        }

        redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
        return result

    def analyze_spatial_patterns(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze spatial patterns in tree distribution using DBSCAN clustering."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("Input must be a GeoDataFrame")

        # Extract coordinates for clustering
        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.01, min_samples=5).fit(coords)
        
        # Analyze clusters
        clusters = pd.Series(clustering.labels_).value_counts().to_dict()
        cluster_stats = {
            'total_clusters': len([c for c in clusters if c != -1]),
            'noise_points': clusters.get(-1, 0),
            'largest_cluster_size': max(clusters.values()) if clusters else 0
        }

        # Calculate spatial density
        density_grid = self._calculate_spatial_density(gdf)

        return {
            'cluster_analysis': cluster_stats,
            'density_grid': density_grid,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_spatial_density(self, gdf: gpd.GeoDataFrame) -> Dict[str, float]:
        """Calculate tree density in a grid pattern."""
        bounds = gdf.total_bounds
        cell_size = 0.01  # Approximately 1km at equator

        grid_cells = {}
        for x in np.arange(bounds[0], bounds[2], cell_size):
            for y in np.arange(bounds[1], bounds[3], cell_size):
                cell = Polygon([
                    (x, y), (x+cell_size, y),
                    (x+cell_size, y+cell_size), (x, y+cell_size)
                ])
                trees_in_cell = gdf[gdf.geometry.intersects(cell)]
                grid_cells[f"{x:.3f},{y:.3f}"] = len(trees_in_cell)

        return grid_cells

    def _prepare_environmental_features(self, climate_data: pd.DataFrame, soil_data: pd.DataFrame) -> np.ndarray:
        """Prepare environmental features for species distribution modeling."""
        # Merge climate and soil data
        features = pd.merge(
            climate_data,
            soil_data,
            on=['lat', 'lon'],
            how='inner'
        )

        # Select relevant features
        selected_features = [
            'temperature_mean', 'precipitation_annual',
            'soil_ph', 'soil_organic_matter', 'elevation'
        ]

        return features[selected_features].values

class TreeHealthPredictor:
    """Predicts tree health based on environmental and physical features."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['diameter', 'height', 'lat', 'lon', 'soil_quality', 'canopy_density']
        self.model_path = model_path or Path(__file__).parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)
        self.model_version = str(uuid4())[:8]

        self.metrics_config = MetricConfig(
            collection_interval=60,
            enable_confidence_intervals=True,
            store_predictions=True
        )

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the health prediction model with enhanced features."""
        X = self._prepare_features(data)
        y = self.label_encoder.fit_transform(data['health'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Track training metrics
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'training_time': training_time,
            'model_version': self.model_version
        }

        # Save model and metrics
        self._save_model(metrics)
        return metrics

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features with enhanced environmental data."""
        if not all(col in data.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in data.columns]
            raise ValueError(f"Missing required feature columns: {missing}")

        # Query PostGIS for additional environmental data
        with postgis_engine.connect() as conn:
            points = [Point(row.lon, row.lat) for _, row in data.iterrows()]
            point_wkts = [f"ST_SetSRID(ST_Point({p.x}, {p.y}), 4326)" for p in points]
            
            query = text("""
                SELECT ST_Distance(point, weather_stations.geom) as distance,
                       temperature, humidity
                FROM unnest(:points) as point
                CROSS JOIN LATERAL (
                    SELECT geom, temperature, humidity
                    FROM weather_stations
                    ORDER BY point <-> geom
                    LIMIT 1
                ) as weather_stations
            """)
            
            result = conn.execute(query, {"points": point_wkts})
            env_data = pd.DataFrame(result.fetchall(), columns=['distance', 'temperature', 'humidity'])

        # Combine original features with environmental data
        features = np.column_stack([
            data[self.feature_columns].values,
            env_data[['temperature', 'humidity']].values
        ])

        return self.scaler.fit_transform(features)

    def _save_model(self, metrics: Dict[str, Any]) -> None:
        """Save model and metrics with version control."""
        model_file = self.model_path / f"health_predictor_{self.model_version}.joblib"
        metrics_file = self.model_path / f"health_predictor_{self.model_version}_metrics.json"

        joblib.dump(self.model, model_file)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)

        # Update symlink to latest model
        latest_link = self.model_path / "health_predictor_latest.joblib"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(model_file.name)
