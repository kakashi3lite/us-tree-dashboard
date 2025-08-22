"""
Machine Learning utilities for tree data analysis and prediction.
This module provides reusable ML components for tree health prediction,
growth forecasting, and species classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
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

# Import metrics components
from .metrics_store import metrics_store, ModelMetrics
from .metrics import (
    calculate_confidence_intervals,
)
from .metrics.base import MetricConfig, Metric
from .metrics.model_metrics import ModelTrainingMetrics, ModelMetric
from .metrics.environmental_metrics import EnvironmentalImpactMetrics
from .metrics.performance_metrics import collect_latency_metrics
from .metrics.base import MetricSeries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreeAnalyzer:
    """
    A class to analyze tree data using various machine learning techniques.
    
    This class provides methods for pattern analysis, health prediction,
    and growth forecasting using both traditional ML and AI approaches.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize TreeAnalyzer.
        
        Args:
            model_path: Optional path to load pre-trained models from
        """
        self.model_path = model_path
        self.health_model = None
        self.growth_model = None
        self._load_models()

    def _load_models(self):
        """Load pre-trained models if available."""
        if self.model_path:
            models_dir = Path(self.model_path)
            health_model_path = models_dir / "health_predictor.joblib"
            growth_model_path = models_dir / "growth_forecaster.joblib"
            
            if health_model_path.exists():
                self.health_model = joblib.load(health_model_path)
            if growth_model_path.exists():
                self.growth_model = joblib.load(growth_model_path)

# ----------------------------------------------------------------------
# Top-level AI pattern analysis helper (used directly in tests)
# ----------------------------------------------------------------------
def analyze_patterns_with_ai(
    data: pd.DataFrame,
    analysis_prompt: str,
    client_factory: Optional[Callable[[], Any]] = None,
) -> str:
    """Analyze patterns in tree data using an AI model (OpenAI by default).

    Accepts both (lat, lon) or (latitude, longitude) column naming variants.
    Performs strict validation prior to invoking the model.
    A factory for the OpenAI client is used to enable test mocking.
    """
    if data.empty:
        raise ValueError("Input data is empty")

    # Column normalization (support different naming conventions)
    rename_map = {}
    if "latitude" in data.columns and "lat" not in data.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in data.columns and "lon" not in data.columns:
        rename_map["longitude"] = "lon"
    if rename_map:
        data = data.rename(columns=rename_map)

    required_cols = {
        "species": str,
        "health": str,
        "diameter": (int, float),
        "lat": (int, float),
        "lon": (int, float),
    }
    for col, dtype in required_cols.items():
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
        if isinstance(dtype, tuple):
            # Numeric validation
            try:
                pd.to_numeric(data[col])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid data type in column {col}. Expected numeric.")
        else:
            if not all(isinstance(x, (str, np.str_)) for x in data[col]):
                raise ValueError(f"Invalid data type in column {col}. Expected string.")

    # Summarize data for prompt
    data_summary = {
        "total_trees": int(len(data)),
        "species_count": data["species"].value_counts().to_dict(),
        "health_distribution": data["health"].value_counts().to_dict(),
        "avg_diameter": float(pd.to_numeric(data["diameter"]).mean()),
        "location_bounds": {
            "lat": (float(data["lat"].min()), float(data["lat"].max())),
            "lon": (float(data["lon"].min()), float(data["lon"].max())),
        },
    }

    system_prompt = (
        "You are an expert arborist and data scientist analyzing urban tree data. "
        "Provide insights based on the data summary and specific analysis request. "
        "Focus on practical implications for urban forestry management."
    )

    user_prompt = (
        f"Data Summary:\n{data_summary}\n\n"
        f"Analysis Request:\n{analysis_prompt}\n\n"
        "Provide a detailed analysis with specific, actionable recommendations."
    )

    # Build / obtain client
    if client_factory is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key not found in environment variables")
        client = OpenAI()
    else:
        client = client_factory()

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:  # pragma: no cover - network errors in real env
        logger.error("Error calling OpenAI API: %s", e)
        raise RuntimeError(f"Failed to analyze patterns: {e}")

class TreeHealthPredictor:
    """Predicts tree health based on various environmental and physical features."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['diameter', 'height', 'lat', 'lon']
        self.model_path = model_path or Path(__file__).parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)
        self.known_health_states = []  # Track known health states during training
        self.model_version = str(uuid4())[:8]  # Generate unique version ID
        
        # Initialize metrics collection
        self.metrics_config = MetricConfig(
            collection_interval=60,  # Collect metrics every 60 seconds
            enable_confidence_intervals=True,
            store_predictions=True
        )
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare features and target for training or prediction."""
        if not all(col in data.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in data.columns]
            raise ValueError(f"Missing required feature columns: {missing}")
        
        X = data[self.feature_columns].copy()
        y = None
        
        if 'health' in data.columns:
            unique_health_states = data['health'].unique()
            if self.known_health_states:
                # Add any new health states found in the data
                new_states = set(unique_health_states) - set(self.known_health_states)
                if new_states:
                    self.known_health_states = list(set(self.known_health_states) | new_states)
                    self.label_encoder.fit(self.known_health_states)
            else:
                # First time training, initialize known states
                self.known_health_states = list(unique_health_states)
                self.label_encoder.fit(self.known_health_states)
            y = self.label_encoder.transform(data['health'])
        
        # Use transform instead of fit_transform for prediction to maintain training scale
        if self.model is None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
        
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """Train the health prediction model and return performance metrics."""
        train_start_time = time.time()
        
        if len(data) == 0:
            raise ValueError("Cannot train on empty dataset")
            
        X, y = self.prepare_features(data)
        if y is None:
            raise ValueError("No health data available for training")
        
        # Handle small datasets
        if len(data) < 5:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Calculate feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_.tolist()))
        metrics['feature_importance'] = feature_importance
        
        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(y_proba)
        metrics['confidence_intervals'] = confidence_intervals
        
        # Save model and components
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'known_health_states': self.known_health_states,
            'feature_columns': self.feature_columns,
            'model_version': self.model_version
        }
        joblib.dump(save_data, self.model_path / 'health_predictor.joblib')
        
        # Store training metrics
        training_time = time.time() - train_start_time
        model_metrics = ModelMetrics(
            model_name='TreeHealthPredictor',
            version=self.model_version,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            parameters={
                'n_estimators': 100,
                'random_state': 42,
                'feature_columns': self.feature_columns,
                'test_size': test_size
            },
            dataset_size=len(data),
            training_time=training_time
        )
        metrics_store.save_model_metrics(model_metrics)
        
        return metrics
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict health status for new trees."""
        predict_start_time = time.time()
        
        if self.model is None:
            model_file = self.model_path / 'health_predictor.joblib'
            if model_file.exists():
                saved_data = joblib.load(model_file)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.label_encoder = saved_data['label_encoder']
                self.known_health_states = saved_data['known_health_states']
                self.feature_columns = saved_data.get('feature_columns', self.feature_columns)
                self.model_version = saved_data.get('model_version', str(uuid4())[:8])
            else:
                raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_features(data)
        predictions = self.model.predict(X)
        prediction_proba = self.model.predict_proba(X)
        
        # Collect prediction metrics
        predict_time = time.time() - predict_start_time
        prediction_metrics = {
            'latency': predict_time,
            'batch_size': len(data),
            'mean_confidence': float(np.max(prediction_proba, axis=1).mean()),
            'min_confidence': float(np.max(prediction_proba, axis=1).min()),
            'max_confidence': float(np.max(prediction_proba, axis=1).max())
        }
        
        # Store prediction metrics as a time series
        metric_series = MetricSeries(
            name='health_prediction_metrics',
            timestamp=datetime.now().isoformat(),
            values=prediction_metrics
        )
        metrics_store.save_metric_series(metric_series)
        
        logger.info(f"Health prediction completed in {predict_time:.2f}s for {len(data)} trees")
        return self.label_encoder.inverse_transform(predictions)

class TreeGrowthForecaster:
    """Forecasts tree growth based on historical data and environmental factors."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None  # Track which columns were used for training
        self.model_path = model_path or Path(__file__).parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)
        self.model_version = str(uuid4())[:8]  # Generate unique version ID
        
        # Initialize metrics collection
        self.metrics_config = MetricConfig(
            collection_interval=60,  # Collect metrics every 60 seconds
            enable_confidence_intervals=True,
            store_predictions=True
        )
        
    def train(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, float]:
        """Train the growth forecasting model."""
        train_start_time = time.time()
        
        if data.empty:
            raise ValueError("Cannot train on empty dataset")
            
        # Support both dbh and diameter column names
        if target_column is None:
            target_column = 'dbh' if 'dbh' in data.columns else 'diameter'
            
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found in data. Expected 'dbh' or 'diameter'.")
        
        # Validate non-negative values
        if (data[target_column] < 0).any():
            raise ValueError(f"Found negative values in {target_column}. All measurements must be non-negative.")
        
        self.feature_cols = ['lat', 'lon', 'age'] if 'age' in data.columns else ['lat', 'lon']
        X = self.scaler.fit_transform(data[self.feature_cols])
        y = data[target_column].values
        
        # Ensure we have enough data to split
        if len(data) < 5:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2_score': max(-1.0, r2_score(y_test, y_pred)),  # Clip R² to -1 minimum
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))
        }
        
        # Calculate confidence intervals using bootstrapping
        confidence_intervals = calculate_confidence_intervals(
            [tree.predict(X_test) for tree in self.model.estimators_]
        )
        metrics['confidence_intervals'] = confidence_intervals
        
        # Save model components
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_version': self.model_version
        }, self.model_path / 'growth_forecaster.joblib')
        
        # Store training metrics
        training_time = time.time() - train_start_time
        model_metrics = ModelMetrics(
            model_name='TreeGrowthForecaster',
            version=self.model_version,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            parameters={
                'n_estimators': 100,
                'random_state': 42,
                'feature_columns': self.feature_cols,
                'target_column': target_column
            },
            dataset_size=len(data),
            training_time=training_time
        )
        metrics_store.save_model_metrics(model_metrics)
        
        return metrics
        
    def forecast(self, data: pd.DataFrame, years_ahead: int = 5) -> pd.DataFrame:
        """Forecast tree growth for specified number of years."""
        forecast_start_time = time.time()
        
        if years_ahead <= 0:
            raise ValueError("years_ahead must be a positive integer")
            
        if self.model is None:
            model_file = self.model_path / 'growth_forecaster.joblib'
            if model_file.exists():
                saved_data = joblib.load(model_file)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_cols = saved_data['feature_cols']
                self.model_version = saved_data.get('model_version', str(uuid4())[:8])
            else:
                raise ValueError("Model not trained. Call train() first.")
        
        if self.feature_cols is None:
            raise ValueError("Model not properly initialized. Feature columns unknown.")
        
        feature_cols = self.feature_cols
        if not all(col in data.columns for col in feature_cols):
            missing = [col for col in feature_cols if col not in data.columns]
            raise ValueError(f"Missing required feature columns: {missing}")
        
        X = self.scaler.transform(data[feature_cols])
        
        forecasts = []
        forecast_uncertainties = []
        
        for year in range(years_ahead):
            if 'age' in feature_cols:
                X[:, feature_cols.index('age')] += 1
            
            # Get predictions from all trees in the forest
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            
            # Calculate mean prediction and uncertainty
            forecast = np.mean(tree_predictions, axis=0)
            uncertainty = np.std(tree_predictions, axis=0)
            
            forecasts.append(forecast)
            forecast_uncertainties.append(uncertainty)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(
            np.array(forecasts).T,
            columns=[f'year_{i+1}' for i in range(years_ahead)],
            index=data.index
        )
        
        # Add uncertainty estimates
        uncertainty_df = pd.DataFrame(
            np.array(forecast_uncertainties).T,
            columns=[f'year_{i+1}_uncertainty' for i in range(years_ahead)],
            index=data.index
        )
        
        # Collect prediction metrics
        forecast_time = time.time() - forecast_start_time
        prediction_metrics = {
            'latency': forecast_time,
            'batch_size': len(data),
            'years_ahead': years_ahead,
            'mean_uncertainty': float(np.mean(forecast_uncertainties)),
            'max_uncertainty': float(np.max(forecast_uncertainties))
        }
        
        # Store prediction metrics
        metric_series = MetricSeries(
            name='growth_forecast_metrics',
            timestamp=datetime.now().isoformat(),
            values=prediction_metrics
        )
        metrics_store.save_metric_series(metric_series)
        
        logger.info(f"Growth forecast completed in {forecast_time:.2f}s for {len(data)} trees")
        return pd.concat([forecast_df, uncertainty_df], axis=1)
        
def calculate_environmental_impact(trees_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate environmental impact metrics for a group of trees."""
    # Constants for environmental calculations (base rates for a mature tree with diameter = 30")
    BASE_CO2_ABSORPTION_RATE = 48  # pounds per year
    BASE_O2_PRODUCTION_RATE = 260  # pounds per year
    BASE_RAINFALL_INTERCEPTED = 760  # gallons per year
    BASE_DIAMETER = 30  # inches
    
    # Health state multipliers
    health_multipliers = {
        'Good': 1.0,
        'Fair': 0.7,
        'Poor': 0.4
    }
    
    # Calculate size-adjusted metrics for each tree
    total_co2 = 0
    total_o2 = 0
    total_rainfall = 0
    
    for _, tree in trees_data.iterrows():
        # Size adjustment factor (relative to 30" diameter baseline)
        size_factor = (tree['diameter'] / BASE_DIAMETER) ** 2  # Square relationship with diameter
        
        # Health adjustment
        health_factor = health_multipliers.get(tree['health'], 0.7)  # Default to 'Fair' if unknown
        
        # Calculate individual tree impact
        total_co2 += BASE_CO2_ABSORPTION_RATE * size_factor * health_factor
        total_o2 += BASE_O2_PRODUCTION_RATE * size_factor * health_factor
        total_rainfall += BASE_RAINFALL_INTERCEPTED * size_factor * health_factor
    
    metrics = {
        'co2_absorbed_lbs_per_year': total_co2,
        'o2_produced_lbs_per_year': total_o2,
        'rainfall_intercepted_gallons': total_rainfall,
        'equivalent_car_emissions_offset': total_co2 / 11000  # Average car emits 11,000 lbs CO2/year
    }
    
    return metrics

def identify_optimal_planting_locations(
    region_data: pd.DataFrame,
    existing_trees: pd.DataFrame,
    constraints: Dict[str, Any]
) -> pd.DataFrame:
    """Identify optimal locations for new tree planting."""
    # Implementation would consider factors like:
    # - Distance from existing trees
    # - Soil quality
    # - Urban heat island effect
    # - Population density
    # - Available space
    # This is a placeholder for the actual implementation
    pass

class TreeValueEstimator:
    """Estimates market value of trees based on multiple factors."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['diameter', 'height', 'lat', 'lon', 'age', 'health_score']
        self.model_path = model_path or Path(__file__).parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)
        self.model_version = str(uuid4())[:8]  # Generate unique version ID
        
        # Initialize metrics collection
        self.metrics_config = MetricConfig(
            collection_interval=60,  # Collect metrics every 60 seconds
            enable_confidence_intervals=True,
            store_predictions=True
        )
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for value estimation."""
        # Convert categorical health to numerical score
        health_scores = {'Poor': 0.3, 'Fair': 0.6, 'Good': 1.0}
        data['health_score'] = data['health'].map(health_scores)
        
        # Calculate tree age if not provided
        if 'age' not in data.columns:
            # Approximate age using diameter (DBH)
            data['age'] = data['diameter'] * 0.5  # Simple approximation
        
        X = data[self.feature_columns].copy()
        return self.scaler.transform(X)
        
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the value estimation model."""
        train_start_time = time.time()
        
        if data.empty:
            raise ValueError("Cannot train on empty dataset")
        
        X = self.prepare_features(data)
        
        # Calculate baseline values based on size and health
        base_values = self._calculate_base_values(data)
        
        # Split data for training
        if len(data) < 5:
            X_train, y_train = X, base_values
            X_test, y_test = X, base_values
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, base_values, test_size=0.2)
        
        # Train model to learn location-based value adjustments
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_.tolist()))
        }
        
        # Calculate confidence intervals using bootstrapping
        confidence_intervals = calculate_confidence_intervals(
            [tree.predict(X_test) for tree in self.model.estimators_]
        )
        metrics['confidence_intervals'] = confidence_intervals
        
        # Save model components
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_version': self.model_version
        }, self.model_path / 'value_estimator.joblib')
        
        # Store training metrics
        training_time = time.time() - train_start_time
        model_metrics = ModelMetrics(
            model_name='TreeValueEstimator',
            version=self.model_version,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            parameters={
                'n_estimators': 100,
                'random_state': 42,
                'feature_columns': self.feature_columns
            },
            dataset_size=len(data),
            training_time=training_time
        )
        metrics_store.save_model_metrics(model_metrics)
        
        return metrics
        
    def estimate_value(self, data: pd.DataFrame) -> pd.DataFrame:
        """Estimate tree values considering multiple factors."""
        estimate_start_time = time.time()
        
        if self.model is None:
            model_file = self.model_path / 'value_estimator.joblib'
            if model_file.exists():
                saved_data = joblib.load(model_file)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.model_version = saved_data.get('model_version', str(uuid4())[:8])
            else:
                raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(data)
        
        # Get predictions from all trees for uncertainty estimation
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        location_multipliers = np.mean(tree_predictions, axis=0)
        uncertainties = np.std(tree_predictions, axis=0)
        
        # Calculate all value components
        base_values = self._calculate_base_values(data)
        eco_values = self._calculate_ecosystem_value(data)
        property_values = self._calculate_property_impact(data)
        
        results = pd.DataFrame({
            'base_value': base_values,
            'location_adjusted_value': base_values * location_multipliers,
            'location_uncertainty': base_values * uncertainties,
            'ecosystem_value': eco_values,
            'property_value_impact': property_values
        })
        
        results['total_value'] = (
            results['location_adjusted_value'] + 
            results['ecosystem_value'] + 
            results['property_value_impact']
        )
        
        # Collect estimation metrics
        estimate_time = time.time() - estimate_start_time
        estimation_metrics = {
            'latency': estimate_time,
            'batch_size': len(data),
            'mean_uncertainty': float(uncertainties.mean()),
            'max_uncertainty': float(uncertainties.max()),
            'mean_total_value': float(results['total_value'].mean()),
            'total_value_std': float(results['total_value'].std())
        }
        
        # Store estimation metrics
        metric_series = MetricSeries(
            name='value_estimation_metrics',
            timestamp=datetime.now().isoformat(),
            values=estimation_metrics
        )
        metrics_store.save_metric_series(metric_series)
        
        logger.info(f"Value estimation completed in {estimate_time:.2f}s for {len(data)} trees")
        return results
        
    def _calculate_base_values(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate base values considering size, species, and health."""
        # Base value calculations based on trunk formula method
        base_values = data['diameter'] * 100  # $100 per inch of diameter as base
        
        # Adjust for health
        health_multipliers = {'Poor': 0.4, 'Fair': 0.7, 'Good': 1.0}
        health_factors = data['health'].map(health_multipliers).fillna(0.7)
        
        # Species value adjustment (simplified)
        species_multipliers = {
            'Oak': 1.2,
            'Maple': 1.1,
            'Pine': 0.9
            # Add more species multipliers as needed
        }
        species_factors = data['species'].map(species_multipliers).fillna(1.0)
        
        return base_values * health_factors * species_factors
        
    def _calculate_ecosystem_value(self, data: pd.DataFrame) -> pd.Series:
        """Calculate monetary value of ecosystem services."""
        # Values based on research studies (simplified)
        co2_price_per_lb = 0.05  # Carbon credit price
        o2_value_per_lb = 0.03   # Oxygen production value
        water_value_per_gallon = 0.01  # Stormwater management value
        
        # Get environmental impact metrics
        impact = calculate_environmental_impact(data)
        
        eco_values = (
            impact['co2_absorbed_lbs_per_year'] * co2_price_per_lb +
            impact['o2_produced_lbs_per_year'] * o2_value_per_lb +
            impact['rainfall_intercepted_gallons'] * water_value_per_gallon
        )
        
        # Convert to present value assuming 20-year lifespan and 5% discount rate
        discount_rate = 0.05
        years = 20
        present_value_factor = (1 - (1 + discount_rate)**-years) / discount_rate
        
        return eco_values * present_value_factor
        
    def _calculate_property_impact(self, data: pd.DataFrame) -> pd.Series:
        """Estimate impact on property values."""
        # Based on research showing trees can increase property value by 3-15%
        avg_property_value = 300000  # Placeholder - should be location-based
        
        # Calculate impact based on tree size and health
        max_impact_pct = 0.05  # Maximum 5% impact per tree
        
        size_factor = (data['diameter'] / 100).clip(0, 1)  # Normalize by max expected size
        health_impact = {'Poor': 0.3, 'Fair': 0.7, 'Good': 1.0}
        health_factor = data['health'].map(health_impact).fillna(0.7)
        
        impact_pct = max_impact_pct * size_factor * health_factor
        return impact_pct * avg_property_value
