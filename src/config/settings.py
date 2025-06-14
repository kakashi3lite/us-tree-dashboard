import os
from datetime import timedelta

# Flask Configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
DEBUG = FLASK_ENV == 'development'

# CORS Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# Database Configuration
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'us_tree_dashboard')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

DATABASE_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Cache Configuration
ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # 5 minutes default

# Rate Limiting Configuration
ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '100 per minute')

# Monitoring Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENABLE_PROMETHEUS = os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'

# Security Configuration
SSL_CERT_PATH = os.getenv('SSL_CERT_PATH')
SSL_KEY_PATH = os.getenv('SSL_KEY_PATH')
ENABLE_SSL = all([SSL_CERT_PATH, SSL_KEY_PATH])

# API Configuration
API_VERSION = 'v1'
API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))  # seconds

# Geospatial Configuration
DEFAULT_ZOOM_LEVEL = int(os.getenv('DEFAULT_ZOOM_LEVEL', 12))
MAX_CLUSTER_RADIUS = float(os.getenv('MAX_CLUSTER_RADIUS', 0.1))  # degrees

# Performance Configuration
MAX_CONNECTIONS = int(os.getenv('MAX_CONNECTIONS', 100))
POOL_SIZE = int(os.getenv('POOL_SIZE', 10))
POOL_TIMEOUT = int(os.getenv('POOL_TIMEOUT', 30))  # seconds

# Feature Flags
ENABLE_HISTORICAL_TRENDS = os.getenv('ENABLE_HISTORICAL_TRENDS', 'true').lower() == 'true'
ENABLE_ENVIRONMENTAL_IMPACT = os.getenv('ENABLE_ENVIRONMENTAL_IMPACT', 'true').lower() == 'true'
ENABLE_SPECIES_FILTERING = os.getenv('ENABLE_SPECIES_FILTERING', 'true').lower() == 'true'