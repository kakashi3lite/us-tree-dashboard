"""Application configuration settings with environment variable support."""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings with environment variable support."""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    METRICS_DIR = BASE_DIR / 'metrics'

    # Flask configuration
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

    # Database configuration
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'us_tree_dashboard')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Get the PostgreSQL database URI."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"\
               f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', '6379')
    REDIS_DB = os.getenv('REDIS_DB', '0')
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

    @property
    def REDIS_URL(self) -> str:
        """Get the Redis connection URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:"\
                   f"{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Cache configuration
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'redis')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '3600'))
    CACHE_KEY_PREFIX = os.getenv('CACHE_KEY_PREFIX', 'us_tree_')

    # API configuration
    API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', '100/minute')
    API_RATE_LIMIT_STORAGE_URL = REDIS_URL

    # Geospatial configuration
    DEFAULT_SRID = 4326  # WGS 84
    TREE_QUERY_LIMIT = int(os.getenv('TREE_QUERY_LIMIT', '1000'))
    SPATIAL_INDEX_ENABLED = True

    # Monitoring configuration
    SENTRY_DSN = os.getenv('SENTRY_DSN', '')
    ENABLE_PERFORMANCE_MONITORING = os.getenv('ENABLE_PERFORMANCE_MONITORING',
                                            'True').lower() == 'true'

    # AWS configuration (if needed)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET', '')

    def get_aws_credentials(self) -> Dict[str, str]:
        """Get AWS credentials if configured."""
        if self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY:
            return {
                'aws_access_key_id': self.AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': self.AWS_SECRET_ACCESS_KEY,
                'region_name': self.AWS_REGION
            }
        return {}

    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.getenv('LOG_FILE', str(BASE_DIR / 'logs/app.log'))
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', str(10 * 1024 * 1024)))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

    # Feature flags
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
    ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'True').lower() == 'true'
    ENABLE_AWS_INTEGRATION = os.getenv('ENABLE_AWS_INTEGRATION',
                                      'False').lower() == 'true'

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        return {
            'enable_caching': self.ENABLE_CACHING,
            'enable_rate_limiting': self.ENABLE_RATE_LIMITING,
            'enable_aws_integration': self.ENABLE_AWS_INTEGRATION,
            'enable_performance_monitoring': self.ENABLE_PERFORMANCE_MONITORING
        }

# Create global settings instance
settings = Settings()