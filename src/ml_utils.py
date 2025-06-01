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
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreeHealthPredictor:
    """Predicts tree health based on various environmental and physical features."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['dbh', 'height', 'lat', 'lon']
        self.model_path = model_path or Path(__file__).parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare features and target for training or prediction."""
        X = data[self.feature_columns].copy()
        y = None
        
        if 'health' in data.columns:
            y = self.label_encoder.fit_transform(data['health'])
        
        return self.scaler.fit_transform(X), y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """Train the health prediction model and return performance metrics."""
        X, y = self.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Save the model
        joblib.dump(self.model, self.model_path / 'health_predictor.joblib')
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict health status for new trees."""
        if self.model is None:
            model_file = self.model_path / 'health_predictor.joblib'
            if model_file.exists():
                self.model = joblib.load(model_file)
            else:
                raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_features(data)
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)

class TreeGrowthForecaster:
    """Forecasts tree growth based on historical data and environmental factors."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or Path(__file__).parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)
    
    def train(self, data: pd.DataFrame, target_column: str = 'dbh') -> Dict[str, float]:
        """Train the growth forecasting model."""
        feature_cols = ['lat', 'lon', 'age'] if 'age' in data.columns else ['lat', 'lon']
        X = self.scaler.fit_transform(data[feature_cols])
        y = data[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        joblib.dump(self.model, self.model_path / 'growth_forecaster.joblib')
        return metrics
    
    def forecast(self, data: pd.DataFrame, years_ahead: int = 5) -> pd.DataFrame:
        """Forecast tree growth for specified number of years."""
        if self.model is None:
            model_file = self.model_path / 'growth_forecaster.joblib'
            if model_file.exists():
                self.model = joblib.load(model_file)
            else:
                raise ValueError("Model not trained. Call train() first.")
        
        feature_cols = ['lat', 'lon', 'age'] if 'age' in data.columns else ['lat', 'lon']
        X = self.scaler.transform(data[feature_cols])
        
        forecasts = []
        for year in range(years_ahead):
            if 'age' in feature_cols:
                X[:, feature_cols.index('age')] += 1
            forecast = self.model.predict(X)
            forecasts.append(forecast)
        
        return pd.DataFrame(np.array(forecasts).T,
                          columns=[f'year_{i+1}' for i in range(years_ahead)],
                          index=data.index)

def calculate_environmental_impact(trees_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate environmental impact metrics for a group of trees."""
    # Constants for environmental calculations
    CO2_ABSORPTION_RATE = 48  # pounds per year for a mature tree
    O2_PRODUCTION_RATE = 260  # pounds per year for a mature tree
    RAINFALL_INTERCEPTED = 760  # gallons per year for a mature tree
    
    # Calculate total impacts
    tree_count = len(trees_data)
    metrics = {
        'co2_absorbed_lbs_per_year': tree_count * CO2_ABSORPTION_RATE,
        'o2_produced_lbs_per_year': tree_count * O2_PRODUCTION_RATE,
        'rainfall_intercepted_gallons': tree_count * RAINFALL_INTERCEPTED,
        'equivalent_car_emissions_offset': (tree_count * CO2_ABSORPTION_RATE) / 11000  # Average car emits 11,000 lbs CO2/year
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
