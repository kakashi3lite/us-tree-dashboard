#!/usr/bin/env python3

import subprocess
import sys
import time\import requests
import docker
import psycopg2
import redis
from elasticsearch import Elasticsearch
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_tests.log'),
        logging.StreamHandler()
    ]
)

class IntegrationTestRunner:
    def __init__(self):
        self.docker_client = docker.from_client()
        self.services_health = {}
        self.test_results = {}

    def wait_for_service(self, service_name, check_function, timeout=60):
        """Wait for a service to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if check_function():
                    logging.info(f'{service_name} is ready')
                    return True
            except Exception as e:
                logging.debug(f'Waiting for {service_name}: {str(e)}')
            time.sleep(2)
        logging.error(f'Timeout waiting for {service_name}')
        return False

    def check_postgres(self):
        """Check PostgreSQL connection."""
        try:
            conn = psycopg2.connect(
                dbname='treedashboard',
                user='postgres',
                password='postgres',
                host='localhost',
                port=5432
            )
            conn.close()
            return True
        except:
            return False

    def check_redis(self):
        """Check Redis connection."""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            return r.ping()
        except:
            return False

    def check_elasticsearch(self):
        """Check Elasticsearch connection."""
        try:
            es = Elasticsearch(['http://localhost:9200'])
            return es.ping()
        except:
            return False

    def check_nginx(self):
        """Check Nginx health."""
        try:
            response = requests.get('http://localhost:80/health')
            return response.status_code == 200
        except:
            return False

    def check_prometheus(self):
        """Check Prometheus metrics endpoint."""
        try:
            response = requests.get('http://localhost:9090/-/healthy')
            return response.status_code == 200
        except:
            return False

    def check_grafana(self):
        """Check Grafana health."""
        try:
            response = requests.get('http://localhost:3000/api/health')
            return response.status_code == 200
        except:
            return False

    def run_pytest(self):
        """Run pytest integration tests."""
        try:
            result = subprocess.run(
                ['pytest', 'tests/test_integration.py', '-v', '--junitxml=test-results.xml'],
                capture_output=True,
                text=True
            )
            return result.returncode == 0, result.stdout
        except Exception as e:
            logging.error(f'Error running pytest: {str(e)}')
            return False, str(e)

    def verify_data_flow(self):
        """Verify data flow between services."""
        try:
            # Test data insertion and retrieval
            test_data = {
                'tree_id': 'TEST_001',
                'species': 'Test Tree',
                'location': 'POINT(-73.935242 40.730610)',
                'height': 20.5,
                'diameter': 0.5,
                'health': 'Good'
            }

            # Insert into PostgreSQL
            with psycopg2.connect(
                dbname='treedashboard',
                user='postgres',
                password='postgres',
                host='localhost'
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trees (tree_id, species, location, height, diameter, health)
                        VALUES (%(tree_id)s, %(species)s, ST_GeomFromText(%(location)s, 4326),
                                %(height)s, %(diameter)s, %(health)s)
                    """, test_data)

            # Verify Redis cache
            r = redis.Redis(host='localhost', port=6379, db=0)
            cached_data = r.get(f"tree:{test_data['tree_id']}")
            if not cached_data:
                return False, 'Redis cache verification failed'

            # Verify Elasticsearch index
            es = Elasticsearch(['http://localhost:9200'])
            es_result = es.search(index='trees', body={
                'query': {
                    'match': {
                        'tree_id': test_data['tree_id']
                    }
                }
            })
            if es_result['hits']['total']['value'] == 0:
                return False, 'Elasticsearch index verification failed'

            return True, 'Data flow verification successful'
        except Exception as e:
            return False, f'Data flow verification failed: {str(e)}'

    def check_websocket(self):
        """Verify WebSocket functionality."""
        try:
            import websocket
            ws = websocket.create_connection('ws://localhost:8050/ws')
            ws.send('ping')
            result = ws.recv()
            ws.close()
            return result == 'pong'
        except Exception as e:
            logging.error(f'WebSocket test failed: {str(e)}')
            return False

    def run_all_tests(self):
        """Run all integration tests."""
        logging.info('Starting integration tests...')

        # Check all services
        services = {
            'PostgreSQL': self.check_postgres,
            'Redis': self.check_redis,
            'Elasticsearch': self.check_elasticsearch,
            'Nginx': self.check_nginx,
            'Prometheus': self.check_prometheus,
            'Grafana': self.check_grafana
        }

        for service_name, check_func in services.items():
            self.services_health[service_name] = self.wait_for_service(service_name, check_func)

        if not all(self.services_health.values()):
            logging.error('Not all services are healthy. Aborting tests.')
            return False

        # Run tests
        tests = [
            ('Pytest Integration', self.run_pytest),
            ('Data Flow', self.verify_data_flow),
            ('WebSocket', self.check_websocket)
        ]

        for test_name, test_func in tests:
            logging.info(f'Running {test_name} test...')
            success, details = test_func()
            self.test_results[test_name] = {
                'success': success,
                'details': details
            }

        # Generate report
        self.generate_report()

        return all(result['success'] for result in self.test_results.values())

    def generate_report(self):
        """Generate integration test report."""
        report = f"Integration Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 50 + "\n\n"

        report += "Service Health:\n"
        report += "-" * 20 + "\n"
        for service, healthy in self.services_health.items():
            status = "✓" if healthy else "✗"
            report += f"{service}: {status}\n"

        report += "\nTest Results:\n"
        report += "-" * 20 + "\n"
        for test_name, result in self.test_results.items():
            status = "✓" if result['success'] else "✗"
            report += f"{test_name}: {status}\n"
            if not result['success']:
                report += f"Details: {result['details']}\n"

        with open('integration_test_report.txt', 'w') as f:
            f.write(report)

        logging.info(f'Test report generated: integration_test_report.txt')

def main():
    runner = IntegrationTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()