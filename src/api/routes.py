from flask import Blueprint, request, jsonify
from src.services.geospatial_service import GeospatialService
from src.services.environmental_service import EnvironmentalService
from src.utils.validators import validate_bbox, validate_coordinates
from src.utils.rate_limiter import rate_limit
from src.monitoring.logger import logger
from datetime import datetime

api = Blueprint('api', __name__)
geospatial_service = GeospatialService()
environmental_service = EnvironmentalService()

@api.route('/trees/radius', methods=['GET'])
@rate_limit(limit=100, per=60)  # 100 requests per minute
def get_trees_in_radius():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 1.0))  # Default 1km radius

        if not validate_coordinates(lat, lon):
            return jsonify({'error': 'Invalid coordinates'}), 400

        trees = geospatial_service.get_trees_in_radius(lat, lon, radius)
        return jsonify(trees)

    except ValueError as e:
        logger.error(f'Invalid parameters for trees in radius: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting trees in radius: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/trees/species-distribution', methods=['GET'])
@rate_limit(limit=60, per=60)  # 60 requests per minute
def get_species_distribution():
    try:
        bbox = request.args.get('bbox')
        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        distribution = geospatial_service.get_species_distribution(bbox)
        return jsonify(distribution)

    except ValueError as e:
        logger.error(f'Invalid parameters for species distribution: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting species distribution: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/trees/density-heatmap', methods=['GET'])
@rate_limit(limit=60, per=60)  # 60 requests per minute
def get_tree_density_heatmap():
    try:
        bbox = request.args.get('bbox')
        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        heatmap = geospatial_service.get_tree_density_heatmap(bbox)
        return jsonify(heatmap)

    except ValueError as e:
        logger.error(f'Invalid parameters for density heatmap: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting density heatmap: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/predictions/species-distribution', methods=['GET'])
@rate_limit(limit=30, per=60)  # 30 requests per minute due to computational intensity
def get_predicted_species_distribution():
    try:
        bbox = request.args.get('bbox')
        climate_scenario = request.args.get('climate_scenario', 'rcp45')  # Default scenario

        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        predictions = geospatial_service.get_predicted_species_distribution(bbox, climate_scenario)
        return jsonify(predictions)

    except ValueError as e:
        logger.error(f'Invalid parameters for predicted species distribution: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting predicted species distribution: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/analysis/climate-impact', methods=['GET'])
@rate_limit(limit=30, per=60)  # 30 requests per minute
def get_climate_impact_analysis():
    try:
        bbox = request.args.get('bbox')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid date format'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        analysis = environmental_service.get_climate_impact_analysis(bbox, start_date, end_date)
        return jsonify(analysis)

    except ValueError as e:
        logger.error(f'Invalid parameters for climate impact analysis: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting climate impact analysis: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/environmental/impact', methods=['GET'])
@rate_limit(limit=60, per=60)  # 60 requests per minute
def get_environmental_impact():
    try:
        bbox = request.args.get('bbox')
        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        impact = environmental_service.get_environmental_impact(bbox)
        return jsonify(impact)

    except ValueError as e:
        logger.error(f'Invalid parameters for environmental impact: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting environmental impact: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/environmental/historical-trends', methods=['GET'])
@rate_limit(limit=30, per=60)  # 30 requests per minute
def get_historical_trends():
    try:
        bbox = request.args.get('bbox')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid date format'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        trends = environmental_service.get_historical_trends(bbox, start_date, end_date)
        return jsonify(trends)

    except ValueError as e:
        logger.error(f'Invalid parameters for historical trends: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting historical trends: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@api.route('/environmental/climate-scenarios', methods=['GET'])
@rate_limit(limit=30, per=60)  # 30 requests per minute
def get_climate_scenarios():
    try:
        bbox = request.args.get('bbox')
        if not bbox or not validate_bbox(bbox):
            return jsonify({'error': 'Invalid bounding box'}), 400

        bbox = [float(x) for x in bbox.split(',')]
        scenarios = environmental_service.get_climate_scenarios(bbox)
        return jsonify(scenarios)

    except ValueError as e:
        logger.error(f'Invalid parameters for climate scenarios: {str(e)}')
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f'Error getting climate scenarios: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500
