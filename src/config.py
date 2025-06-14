import os
import logging
from logging.handlers import RotatingFileHandler

# Application configuration
class Config:
    # Flask configuration
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/app.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5

    @staticmethod
    def init_logging():
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up logging configuration
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=Config.LOG_FORMAT
        )
        
        # Add file handler with rotation
        handler = RotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=Config.LOG_MAX_SIZE,
            backupCount=Config.LOG_BACKUP_COUNT
        )
        handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        
        # Add handler to root logger
        logging.getLogger('').addHandler(handler)

# Error handling configuration
class ErrorConfig:
    # Custom error messages
    ERROR_MESSAGES = {
        'data_not_found': 'Required data not found',
        'processing_error': 'Error processing request',
        'invalid_parameters': 'Invalid parameters provided',
        'model_error': 'Error in model prediction',
        'system_error': 'Internal system error'
    }
    
    # HTTP status codes
    HTTP_STATUS_CODES = {
        'success': 200,
        'bad_request': 400,
        'not_found': 404,
        'server_error': 500
    }
    
    @staticmethod
    def get_error_response(error_type, details=None):
        """Generate standardized error response"""
        response = {
            'error': {
                'type': error_type,
                'message': ErrorConfig.ERROR_MESSAGES.get(
                    error_type,
                    'Unknown error'
                )
            }
        }
        
        if details:
            response['error']['details'] = details
            
        return response