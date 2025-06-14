import logging
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from flask import request, has_request_context
from src.config.settings import LOG_DIR, LOG_LEVEL, LOG_FORMAT
import os

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

        # Add request context if available
        if has_request_context():
            log_record['request_id'] = request.headers.get('X-Request-ID')
            log_record['user_agent'] = request.headers.get('User-Agent')
            log_record['ip_address'] = request.remote_addr
            log_record['endpoint'] = request.endpoint
            log_record['method'] = request.method
            log_record['path'] = request.path

        # Add environment info
        log_record['environment'] = os.getenv('FLASK_ENV', 'development')

class RequestContextFilter(logging.Filter):
    def filter(self, record):
        if has_request_context():
            record.request_id = request.headers.get('X-Request-ID')
            record.user_agent = request.headers.get('User-Agent')
            record.ip_address = request.remote_addr
            record.endpoint = request.endpoint
            record.method = request.method
            record.path = request.path
        else:
            record.request_id = None
            record.user_agent = None
            record.ip_address = None
            record.endpoint = None
            record.method = None
            record.path = None
        return True

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logger
logger = logging.getLogger('us_tree_dashboard')
logger.setLevel(LOG_LEVEL)

# Add request context filter
logger.addFilter(RequestContextFilter())

# Configure handlers
handlers = []

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
formatter = CustomJsonFormatter(LOG_FORMAT)
console_handler.setFormatter(formatter)
handlers.append(console_handler)

# File handler for general logs
general_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'app.log'),
    maxBytes=10485760,  # 10MB
    backupCount=10
)
general_handler.setLevel(LOG_LEVEL)
general_handler.setFormatter(formatter)
handlers.append(general_handler)

# File handler for error logs
error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'error.log'),
    maxBytes=10485760,  # 10MB
    backupCount=10
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
handlers.append(error_handler)

# Add handlers to logger
for handler in handlers:
    logger.addHandler(handler)

def log_api_request(response):
    """
    Log API request details and response
    """
    if has_request_context():
        logger.info(
            'API Request',
            extra={
                'request_data': {
                    'method': request.method,
                    'path': request.path,
                    'args': dict(request.args),
                    'headers': dict(request.headers),
                },
                'response_data': {
                    'status_code': response.status_code,
                    'content_length': response.content_length,
                }
            }
        )

def log_error(error):
    """
    Log error details with stack trace
    """
    logger.error(
        'Application Error',
        extra={
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': error.__traceback__,
        },
        exc_info=True
    )

def log_performance_metric(operation, duration_ms):
    """
    Log performance metrics
    """
    logger.info(
        'Performance Metric',
        extra={
            'metric_type': 'performance',
            'operation': operation,
            'duration_ms': duration_ms
        }
    )

def log_cache_metric(operation, hit):
    """
    Log cache metrics
    """
    logger.info(
        'Cache Metric',
        extra={
            'metric_type': 'cache',
            'operation': operation,
            'cache_hit': hit
        }
    )

def log_geospatial_metric(operation, bbox, result_count):
    """
    Log geospatial query metrics
    """
    logger.info(
        'Geospatial Metric',
        extra={
            'metric_type': 'geospatial',
            'operation': operation,
            'bbox': bbox,
            'result_count': result_count
        }
    )

def log_ml_metric(model_name, prediction_count, execution_time):
    """
    Log machine learning metrics
    """
    logger.info(
        'ML Metric',
        extra={
            'metric_type': 'ml',
            'model_name': model_name,
            'prediction_count': prediction_count,
            'execution_time': execution_time
        }
    )
