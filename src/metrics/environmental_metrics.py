"""
Environmental impact metrics collection.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from .base import BaseMetric, MetricSeries

@dataclass
class EnvironmentalMetric(BaseMetric):
    """Environmental impact metric."""
    impact_type: str
    location: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None

    def validate(self) -> bool:
        """Validate environmental metric value."""
        if self.impact_type in ['co2_absorption', 'o2_production', 'rainfall_interception']:
            return isinstance(self.value, (int, float)) and self.value >= 0
        return super().validate()

@dataclass
class EnvironmentalImpactMetrics:
    """Collection of environmental impact metrics."""
    total_trees: int
    species_distribution: Dict[str, int]
    metrics: Dict[str, float]
    location_info: Dict[str, Any]
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    health_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    def __post_init__(self):
        """Calculate confidence intervals if not provided."""
        if not self.confidence_intervals:
            self.confidence_intervals = calculate_confidence_intervals(self.metrics)

    def to_metric_list(self) -> List[EnvironmentalMetric]:
        """Convert impact metrics to list of individual metrics."""
        metrics = []
        timestamp = self.timestamp

        # Core environmental metrics
        for impact_type, value in self.metrics.items():
            confidence = self.confidence_intervals.get(impact_type, {}).get('confidence', None)
            metrics.append(EnvironmentalMetric(
                name=f"environmental_{impact_type}",
                value=value,
                timestamp=timestamp,
                tags={'total_trees': self.total_trees},
                impact_type=impact_type,
                confidence=confidence
            ))

        # Species distribution metrics
        for species, count in self.species_distribution.items():
            metrics.append(EnvironmentalMetric(
                name=f"species_count_{species}",
                value=count,
                timestamp=timestamp,
                tags={'metric_type': 'species_distribution'},
                impact_type='population'
            ))

        # Health distribution metrics
        for health, count in self.health_distribution.items():
            metrics.append(EnvironmentalMetric(
                name=f"health_status_{health}",
                value=count,
                timestamp=timestamp,
                tags={'metric_type': 'health_distribution'},
                impact_type='health'
            ))

        # Seasonal impact factors
        for season, factor in self.seasonal_factors.items():
            metrics.append(EnvironmentalMetric(
                name=f"seasonal_factor_{season}",
                value=factor,
                timestamp=timestamp,
                tags={'metric_type': 'seasonal_adjustment'},
                impact_type='seasonal'
            ))

        return metrics

def calculate_confidence_intervals(metrics: Dict[str, float],
                                confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """Calculate confidence intervals for environmental metrics."""
    intervals = {}
    for metric_name, value in metrics.items():
        # Simplified confidence interval calculation
        # In a real implementation, this would use proper statistical methods
        std_error = value * 0.1  # Assume 10% standard error
        z_score = 1.96  # 95% confidence level
        
        margin_of_error = z_score * std_error
        intervals[metric_name] = {
            'lower': max(0, value - margin_of_error),
            'upper': value + margin_of_error,
            'confidence': confidence_level
        }
    
    return intervals

def calculate_energy_savings(trees_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate energy savings from tree shade and wind reduction."""
    # Energy savings calculations based on research studies
    # Values based on:
    # - Tree size (diameter, height)
    # - Species type
    # - Location relative to buildings
    # - Climate zone
    
    avg_cooling_savings = 100  # kWh per large tree per year
    avg_heating_reduction = 50  # therms per large tree per year
    
    energy_savings = {
        'cooling_kwh_per_year': 0,
        'heating_therms_per_year': 0,
        'total_energy_savings_usd': 0
    }
    
    for _, tree in trees_data.iterrows():
        # Size factor (0-1 based on tree size)
        size_factor = min(1, tree['diameter'] / 30)
        
        # Location factor (0-1 based on proximity to buildings)
        location_factor = 0.5  # Placeholder - would use actual building proximity
        
        # Species factor (based on canopy density and deciduous/evergreen)
        species_factors = {
            'Oak': 1.0,
            'Maple': 0.9,
            'Pine': 0.8
        }
        species_factor = species_factors.get(tree['species'], 0.7)
        
        # Calculate savings
        tree_cooling = avg_cooling_savings * size_factor * location_factor * species_factor
        tree_heating = avg_heating_reduction * size_factor * location_factor * species_factor
        
        energy_savings['cooling_kwh_per_year'] += tree_cooling
        energy_savings['heating_therms_per_year'] += tree_heating
    
    # Convert to monetary savings
    kwh_rate = 0.12  # $ per kWh
    therm_rate = 1.50  # $ per therm
    energy_savings['total_energy_savings_usd'] = (
        energy_savings['cooling_kwh_per_year'] * kwh_rate +
        energy_savings['heating_therms_per_year'] * therm_rate
    )
    
    return energy_savings

def calculate_air_quality_improvement(trees_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate air quality improvements from trees."""
    # Air quality improvement calculations based on:
    # - Pollutant removal rates
    # - Leaf surface area
    # - Species characteristics
    # - Local air quality conditions
    
    pollutant_removal_rates = {
        'PM10': 0.0055,  # lbs per sq meter of leaf area per year
        'O3': 0.0045,    # lbs per sq meter of leaf area per year
        'NO2': 0.0025,   # lbs per sq meter of leaf area per year
        'SO2': 0.0015    # lbs per sq meter of leaf area per year
    }
    
    air_quality_metrics = {
        'PM10_removed_lbs': 0,
        'O3_removed_lbs': 0,
        'NO2_removed_lbs': 0,
        'SO2_removed_lbs': 0,
        'monetized_benefit_usd': 0
    }
    
    for _, tree in trees_data.iterrows():
        # Calculate leaf area (simplified)
        leaf_area = tree['diameter'] * 0.5  # rough estimate in sq meters
        
        # Adjust for species characteristics
        species_multipliers = {
            'Oak': 1.2,
            'Maple': 1.0,
            'Pine': 0.8
        }
        species_factor = species_multipliers.get(tree['species'], 1.0)
        
        # Calculate pollutant removal
        for pollutant, rate in pollutant_removal_rates.items():
            removed = leaf_area * rate * species_factor
            air_quality_metrics[f'{pollutant}_removed_lbs'] += removed
    
    # Monetize benefits (EPA values)
    benefit_values = {
        'PM10': 20,   # $ per lb
        'O3': 4.5,    # $ per lb
        'NO2': 3.5,   # $ per lb
        'SO2': 2.0    # $ per lb
    }
    
    total_value = sum(
        air_quality_metrics[f'{p}_removed_lbs'] * v
        for p, v in benefit_values.items()
    )
    air_quality_metrics['monetized_benefit_usd'] = total_value
    
    return air_quality_metrics

def calculate_urban_heat_mitigation(trees_data: pd.DataFrame,
                                  temperature_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Calculate urban heat island mitigation effects."""
    # Heat mitigation calculations based on:
    # - Canopy cover
    # - Tree size and species
    # - Local temperature data
    # - Urban density
    
    heat_metrics = {
        'temperature_reduction_celsius': 0,
        'cooling_days_impact': 0,
        'energy_savings_kwh': 0,
        'comfort_improvement_index': 0
    }
    
    # Default temperature impact if no data provided
    base_temp_reduction = 0.5  # °C per large tree in immediate vicinity
    
    for _, tree in trees_data.iterrows():
        # Size-based cooling factor
        size_factor = min(1, tree['diameter'] / 30)
        
        # Species cooling effectiveness
        species_cooling = {
            'Oak': 1.2,
            'Maple': 1.0,
            'Pine': 0.8
        }
        species_factor = species_cooling.get(tree['species'], 1.0)
        
        # Calculate cooling effect
        tree_cooling = base_temp_reduction * size_factor * species_factor
        
        heat_metrics['temperature_reduction_celsius'] += tree_cooling
    
    # Adjust for overlapping effects
    total_trees = len(trees_data)
    if total_trees > 0:
        overlap_factor = 1 / (1 + np.log(total_trees) * 0.1)
        heat_metrics['temperature_reduction_celsius'] *= overlap_factor
    
    # Calculate derived metrics
    cooling_days_base = 90  # Typical cooling days per year
    heat_metrics['cooling_days_impact'] = (
        heat_metrics['temperature_reduction_celsius'] * cooling_days_base * 0.05
    )
    
    # Energy savings from reduced cooling needs
    kwh_per_degree_day = 0.5  # kWh saved per degree-day of cooling avoided
    heat_metrics['energy_savings_kwh'] = (
        heat_metrics['cooling_days_impact'] * kwh_per_degree_day * 24
    )
    
    # Comfort improvement index (0-100 scale)
    max_temp_reduction = 3.0  # °C
    heat_metrics['comfort_improvement_index'] = min(
        100,
        (heat_metrics['temperature_reduction_celsius'] / max_temp_reduction) * 100
    )
    
    return heat_metrics
