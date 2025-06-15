#!/usr/bin/env python3
"""
PlantsWorld Dashboard Configuration
Makes the dashboard scalable and easily configurable for integration into other projects.
"""

import os
from pathlib import Path

# Dashboard Configuration
class DashboardConfig:
    """Configuration class for PlantsWorld Dashboard"""
    
    # Basic Settings
    APP_NAME = "PlantsWorld"
    APP_TITLE = "🌿 PlantsWorld - Interactive Plant Biodiversity Dashboard"
    APP_DESCRIPTION = "Explore the fascinating world of plants through interactive data visualizations"
    
    # Server Configuration
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8050))
    
    # Data Configuration
    DATA_DIR = Path(__file__).parent / 'data'
    DATASETS_CONFIG = {
        'gbif_species': {
            'filename_pattern': 'gbif_species_*.csv',
            'display_name': 'GBIF Species Data',
            'description': 'Global plant species occurrence data from GBIF'
        },
        'plant_families': {
            'filename_pattern': 'plant_families_*.csv',
            'display_name': 'Plant Families',
            'description': 'Taxonomic family information and diversity metrics'
        },
        'conservation_status': {
            'filename_pattern': 'conservation_status_*.csv',
            'display_name': 'Conservation Status',
            'description': 'IUCN conservation status data for plant species'
        },
        'biodiversity_hotspots': {
            'filename_pattern': 'biodiversity_hotspots_*.csv',
            'display_name': 'Biodiversity Hotspots',
            'description': 'Global biodiversity hotspot locations and metrics'
        }
    }
    
    # UI Theme Configuration
    THEME_CONFIG = {
        'primary_color': '#28a745',
        'secondary_color': '#6c757d',
        'success_color': '#28a745',
        'info_color': '#17a2b8',
        'warning_color': '#ffc107',
        'danger_color': '#dc3545',
        'light_color': '#f8f9fa',
        'dark_color': '#343a40'
    }
    
    # Map Configuration
    MAP_CONFIG = {
        'default_center': {'lat': 20, 'lon': 0},
        'default_zoom': 2,
        'mapbox_style': 'open-street-map',
        'height': 600
    }
    
    # Chart Configuration
    CHART_CONFIG = {
        'default_height': 400,
        'color_palette': ['#28a745', '#17a2b8', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14'],
        'font_family': 'Arial, sans-serif'
    }
    
    # Data Update Configuration
    UPDATE_CONFIG = {
        'auto_update': os.getenv('AUTO_UPDATE', 'False').lower() == 'true',
        'update_interval_hours': int(os.getenv('UPDATE_INTERVAL_HOURS', 24)),
        'max_records_per_dataset': int(os.getenv('MAX_RECORDS', 10000))
    }
    
    # Integration Settings
    INTEGRATION_CONFIG = {
        'api_enabled': os.getenv('API_ENABLED', 'False').lower() == 'true',
        'api_prefix': '/api/v1',
        'cors_enabled': os.getenv('CORS_ENABLED', 'True').lower() == 'true',
        'export_formats': ['csv', 'json', 'xlsx']
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_enabled': os.getenv('LOG_TO_FILE', 'True').lower() == 'true',
        'file_path': 'logs/plantsworld.log'
    }
    
    @classmethod
    def get_latest_dataset_file(cls, dataset_key):
        """Get the most recent file for a given dataset"""
        if dataset_key not in cls.DATASETS_CONFIG:
            return None
            
        pattern = cls.DATASETS_CONFIG[dataset_key]['filename_pattern']
        files = list(cls.DATA_DIR.glob(pattern))
        
        if not files:
            return None
            
        # Return the most recent file based on modification time
        return max(files, key=lambda f: f.stat().st_mtime)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        # Ensure data directory exists
        cls.DATA_DIR.mkdir(exist_ok=True)
        
        # Ensure logs directory exists if logging to file
        if cls.LOGGING_CONFIG['file_enabled']:
            log_path = Path(cls.LOGGING_CONFIG['file_path'])
            log_path.parent.mkdir(exist_ok=True)
        
        return True

# Environment-specific configurations
class DevelopmentConfig(DashboardConfig):
    """Development environment configuration"""
    DEBUG = True
    
class ProductionConfig(DashboardConfig):
    """Production environment configuration"""
    DEBUG = False
    
class TestingConfig(DashboardConfig):
    """Testing environment configuration"""
    DEBUG = True
    DATA_DIR = Path(__file__).parent / 'test_data'

# Configuration factory
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig
    elif env == 'testing':
        return TestingConfig
    else:
        return DevelopmentConfig

# Default configuration
config = get_config()