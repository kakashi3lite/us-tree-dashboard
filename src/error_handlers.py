import logging
from functools import wraps
from src.config import ErrorConfig

logger = logging.getLogger(__name__)

def handle_exceptions(f):
    """Decorator for handling exceptions in routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f'Invalid input: {str(e)}')
            return ErrorConfig.get_error_response(
                'invalid_parameters',
                str(e)
            ), ErrorConfig.HTTP_STATUS_CODES['bad_request']
        except FileNotFoundError as e:
            logger.error(f'Data not found: {str(e)}')
            return ErrorConfig.get_error_response(
                'data_not_found',
                str(e)
            ), ErrorConfig.HTTP_STATUS_CODES['not_found']
        except Exception as e:
            logger.error(f'Unexpected error: {str(e)}')
            return ErrorConfig.get_error_response(
                'system_error',
                str(e)
            ), ErrorConfig.HTTP_STATUS_CODES['server_error']
    return wrapper

def init_error_handlers(app):
    """Initialize error handlers for the application"""
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f'Route not found: {error}')
        return ErrorConfig.get_error_response(
            'not_found',
            'Requested resource not found'
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Internal server error: {error}')
        return ErrorConfig.get_error_response(
            'system_error',
            'Internal server error'
        ), 500