import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from functools import wraps
import time
import os

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['environment'] = os.getenv('FLASK_ENV', 'development')

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    json_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    return logger

def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info({
                'message': f'Function {func.__name__} executed',
                'execution_time': execution_time,
                'function': func.__name__
            })
            return result
        return wrapper
    return decorator

def log_api_call(logger: logging.Logger):
    """Decorator to log API calls with request and response details."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info({
                    'message': f'API call to {func.__name__} successful',
                    'execution_time': execution_time,
                    'endpoint': func.__name__,
                    'status': 'success'
                })
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error({
                    'message': f'API call to {func.__name__} failed',
                    'execution_time': execution_time,
                    'endpoint': func.__name__,
                    'error': str(e),
                    'status': 'error'
                })
                raise
        return wrapper
    return decorator

def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log error with additional context."""
    error_data = {
        'message': str(error),
        'error_type': error.__class__.__name__,
        'timestamp': datetime.utcnow().isoformat()
    }

    if context:
        error_data['context'] = context

    logger.error(json.dumps(error_data))

def setup_monitoring(app_name: str) -> logging.Logger:
    """Set up monitoring infrastructure for the application."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'{app_name}.log')
    logger = setup_logger(
        name=app_name,
        log_file=log_file,
        level=logging.INFO if os.getenv('FLASK_ENV') == 'production' else logging.DEBUG
    )

    return logger