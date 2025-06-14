from flask import Blueprint, jsonify, request, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
from typing import Optional, Dict, Any

from src.services.geospatial_service import GeospatialService
from src.error_handlers import APIError, handle_api_error

api = Blueprint('api', __name__)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

geospatial_service = GeospatialService()

@api.errorhandler(APIError)
def handle_error(error):
    return handle_api_error(error)

@api.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@api.route('/tree-clusters')
@limiter.limit("30 per minute")
async def get_tree_clusters():
    """Get clustered tree locations within specified bounds."""
    try:
        bounds = _parse_bounds_params(request.args)
        zoom_level = int(request.args.get('zoom', 12))
        
        clusters = await geospatial_service.get_tree_clusters(bounds, zoom_level)
        return jsonify({
            'status': 'success',
            'data': clusters
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching tree clusters: {str(e)}")
        raise APIError(str(e))

@api.route('/tree-density')
@limiter.limit("60 per minute")
async def get_tree_density():
    """Get tree density statistics by region."""
    try:
        region_id = request.args.get('region_id', type=int)
        density_data = await geospatial_service.get_tree_density(region_id)
        return jsonify({
            'status': 'success',
            'data': density_data
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching tree density: {str(e)}")
        raise APIError(str(e))

@api.route('/environmental-impact')
@limiter.limit("60 per minute")
async def get_environmental_impact():
    """Calculate environmental impact metrics."""
    try:
        params = _parse_impact_params(request.args)
        impact_data = await geospatial_service.get_environmental_impact(**params)
        return jsonify({
            'status': 'success',
            'data': impact_data
        })
    except Exception as e:
        current_app.logger.error(f"Error calculating environmental impact: {str(e)}")
        raise APIError(str(e))

@api.route('/historical-trends')
@limiter.limit("60 per minute")
async def get_historical_trends():
    """Get historical trends for specified metrics."""
    try:
        params = _parse_trend_params(request.args)
        trend_data = await geospatial_service.get_historical_trends(**params)
        return jsonify({
            'status': 'success',
            'data': trend_data
        })
    except Exception as e:
        current_app.logger.error(f"Error fetching historical trends: {str(e)}")
        raise APIError(str(e))

def _parse_bounds_params(args) -> Dict[str, float]:
    """Parse and validate bounds parameters from request."""
    try:
        return {
            'min_lat': float(args.get('min_lat')),
            'max_lat': float(args.get('max_lat')),
            'min_lon': float(args.get('min_lon')),
            'max_lon': float(args.get('max_lon'))
        }
    except (TypeError, ValueError) as e:
        raise APIError(f"Invalid bounds parameters: {str(e)}")

def _parse_impact_params(args) -> Dict[str, Any]:
    """Parse and validate environmental impact parameters."""
    params = {}
    
    if 'region_id' in args:
        try:
            params['region_id'] = int(args['region_id'])
        except ValueError:
            raise APIError("Invalid region_id parameter")
    
    if 'start_date' in args:
        try:
            params['start_date'] = datetime.fromisoformat(args['start_date'])
        except ValueError:
            raise APIError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if 'end_date' in args:
        try:
            params['end_date'] = datetime.fromisoformat(args['end_date'])
        except ValueError:
            raise APIError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    return params

def _parse_trend_params(args) -> Dict[str, Any]:
    """Parse and validate historical trends parameters."""
    valid_metrics = {'height', 'diameter', 'canopy_width'}
    valid_intervals = {'day', 'week', 'month', 'quarter', 'year'}
    
    metric = args.get('metric')
    if not metric or metric not in valid_metrics:
        raise APIError(f"Invalid metric. Must be one of: {', '.join(valid_metrics)}")
    
    interval = args.get('interval', 'month')
    if interval not in valid_intervals:
        raise APIError(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
    
    params = {
        'metric': metric,
        'interval': interval
    }
    
    if 'start_date' in args:
        try:
            params['start_date'] = datetime.fromisoformat(args['start_date'])
        except ValueError:
            raise APIError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if 'end_date' in args:
        try:
            params['end_date'] = datetime.fromisoformat(args['end_date'])
        except ValueError:
            raise APIError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    return params