import os
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix

from src.api.routes import api
from src.monitoring.logger import setup_monitoring
from src.monitoring.metrics import MetricsCollector, MetricsConfig
from src.error_handlers import register_error_handlers
from src.config.settings import (
    FLASK_ENV,
    SECRET_KEY,
    CORS_ORIGINS,
    ENABLE_PROMETHEUS,
    LOG_LEVEL
)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object('src.config.settings')
    app.secret_key = SECRET_KEY

    # Configure logging and monitoring
    logger = setup_monitoring('us_tree_dashboard')
    metrics_config = MetricsConfig(
        app_name='us_tree_dashboard',
        environment=FLASK_ENV,
        enable_prometheus=ENABLE_PROMETHEUS
    )
    metrics_collector = MetricsCollector(metrics_config)

    # Configure CORS
    CORS(app, resources={
        r"/api/*": {"origins": CORS_ORIGINS}
    })

    # Configure rate limiting
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )

    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')

    # Register error handlers
    register_error_handlers(app)

    # Configure Prometheus metrics endpoint
    if ENABLE_PROMETHEUS:
        app.wsgi_app = DispatcherMiddleware(
            app.wsgi_app,
            {
                '/metrics': make_wsgi_app()
            }
        )

    # Configure proxy settings
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=1,
        x_proto=1,
        x_host=1,
        x_port=1,
        x_prefix=1
    )

    # Request handlers for metrics collection
    @app.before_request
    def before_request():
        from flask import request, g
        import time
        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        from flask import request, g
        import time

        # Skip metrics collection for prometheus endpoint
        if request.path == '/metrics':
            return response

        duration = time.time() - g.start_time
        metrics_collector.record_api_request(
            endpoint=request.endpoint,
            method=request.method,
            status=str(response.status_code),
            duration=duration
        )

        return response

    return app

def run_app(app):
    """Run the Flask application with the appropriate configuration."""
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8050))
    debug = FLASK_ENV == 'development'

    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        threaded=True
    )

if __name__ == '__main__':
    app = create_app()
    run_app(app)