import pytest
import requests
import redis
import psycopg2
import json
from datetime import datetime, timedelta
from unittest.mock import patch
from src.services.geospatial_service import GeospatialService
from src.services.environmental_service import EnvironmentalService
from src.ml_utils import TreeAnalyzer, TreeHealthPredictor

@pytest.fixture
def app_client():
    from app import app
    with app.test_client() as client:
        yield client

@pytest.fixture
def redis_client():
    return redis.Redis(host='redis', port=6379, db=0)

@pytest.fixture
def postgres_conn():
    return psycopg2.connect(
        dbname='treedash',
        user='postgres',
        password='postgres',
        host='postgres'
    )

class TestSystemIntegration:
    def test_service_health(self, app_client):
        """Test health endpoints of all services"""
        response = app_client.get('/health')
        assert response.status_code == 200
        assert response.data == b'healthy\n'

    def test_redis_connection(self, redis_client):
        """Test Redis connectivity and operations"""
        redis_client.set('test_key', 'test_value')
        assert redis_client.get('test_key') == b'test_value'
        redis_client.delete('test_key')

    def test_postgres_spatial_query(self, postgres_conn):
        """Test PostGIS spatial queries"""
        with postgres_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*)
                FROM trees
                WHERE ST_DWithin(
                    geom,
                    ST_SetSRID(ST_MakePoint(-122.4194, 37.7749), 4326),
                    1000
                )
            """)
            count = cur.fetchone()[0]
            assert isinstance(count, int)

class TestDataFlow:
    def test_data_ingestion_pipeline(self, app_client, redis_client):
        """Test complete data flow from ingestion to cache"""
        test_data = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'species': 'Test Tree',
            'height': 10.5
        }
        response = app_client.post('/api/trees', json=test_data)
        assert response.status_code == 201

        # Verify cache update
        cache_key = f"tree:lat:{test_data['latitude']}:lon:{test_data['longitude']}"
        cached_data = redis_client.get(cache_key)
        assert cached_data is not None

    def test_real_time_updates(self, app_client):
        """Test WebSocket real-time updates"""
        from flask_socketio import SocketIO
        socketio = SocketIO(app_client)

        received_data = []
        def handle_update(data):
            received_data.append(data)

        socketio.on('tree_update')(handle_update)
        
        test_update = {'type': 'new_tree', 'data': {'id': 1, 'species': 'Test'}}
        socketio.emit('tree_update', test_update)
        
        assert len(received_data) == 1
        assert received_data[0] == test_update

class TestPerformance:
    def test_cache_performance(self, app_client, redis_client):
        """Test caching system under load"""
        test_points = [
            (37.7749, -122.4194),
            (37.7750, -122.4195),
            (37.7751, -122.4196)
        ]

        # Warm up cache
        for lat, lon in test_points:
            app_client.get(f'/api/trees?lat={lat}&lon={lon}')

        # Test cached responses
        start_time = datetime.now()
        for lat, lon in test_points:
            response = app_client.get(f'/api/trees?lat={lat}&lon={lon}')
            assert response.status_code == 200

        duration = (datetime.now() - start_time).total_seconds()
        assert duration < 0.1  # Less than 100ms for cached responses

    def test_spatial_index_performance(self, postgres_conn):
        """Test spatial index performance"""
        with postgres_conn.cursor() as cur:
            # Test query execution time with spatial index
            cur.execute("""
                EXPLAIN ANALYZE
                SELECT COUNT(*)
                FROM trees
                WHERE ST_DWithin(
                    geom,
                    ST_SetSRID(ST_MakePoint(-122.4194, 37.7749), 4326),
                    1000
                )
            """)
            results = cur.fetchall()
            execution_time = float(results[-1][0].split('time=')[-1].split(' ms')[0])
            assert execution_time < 100  # Less than 100ms

class TestAnalytics:
    def test_climate_modeling(self):
        """Test climate scenario modeling accuracy"""
        env_service = EnvironmentalService()
        scenarios = env_service.generate_climate_scenarios(
            latitude=37.7749,
            longitude=-122.4194,
            timeframe=2050
        )
        assert len(scenarios) > 0
        assert all(s['year'] <= 2050 for s in scenarios)

    def test_species_forecasting(self):
        """Test species migration forecasting"""
        predictor = TreeHealthPredictor()
        forecast = predictor.predict_species_distribution(
            latitude=37.7749,
            longitude=-122.4194,
            years_ahead=10
        )
        assert isinstance(forecast, dict)
        assert 'species_distribution' in forecast

class TestSecurity:
    def test_rate_limiting(self, app_client):
        """Test API rate limiting"""
        for _ in range(101):  # Exceed rate limit
            response = app_client.get('/api/trees')
        assert response.status_code == 429

    def test_authentication(self, app_client):
        """Test authentication system"""
        # Test without auth
        response = app_client.get('/api/admin/users')
        assert response.status_code == 401

        # Test with auth
        headers = {'Authorization': 'Bearer test_token'}
        response = app_client.get('/api/admin/users', headers=headers)
        assert response.status_code in [200, 403]  # Either success or forbidden

class TestUserInterface:
    def test_drag_drop_feature(self, app_client):
        """Test drag and drop data layer creation"""
        test_layer = {
            'name': 'Test Layer',
            'type': 'heatmap',
            'data': [{'lat': 37.7749, 'lon': -122.4194, 'value': 1}]
        }
        response = app_client.post('/api/layers', json=test_layer)
        assert response.status_code == 201

    def test_time_slider(self, app_client):
        """Test time-slider functionality"""
        response = app_client.get('/api/historical-data?start=2020&end=2023')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert all(2020 <= year <= 2023 for year in data['years'])

    def test_mobile_responsiveness(self, app_client):
        """Test mobile device compatibility"""
        headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'}
        response = app_client.get('/', headers=headers)
        assert response.status_code == 200
        assert 'viewport' in response.data.decode()