import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from flask import Flask
from src.metrics.environmental_metrics import EnvironmentalMetrics
from src.metrics.model_metrics import ModelMetrics
from src.metrics.performance_metrics import PerformanceMetrics
from src.config import Config, ErrorConfig
from src.error_handlers import init_error_handlers, handle_exceptions
import logging

# Initialize logging
Config.init_logging()
logger = logging.getLogger(__name__)

# Initialize the Flask server
server = Flask(__name__)

# Initialize Dash app with Flask server
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Initialize error handlers
init_error_handlers(server)

# Create instances of metric classes with error handling
try:
    environmental_metrics = EnvironmentalMetrics()
    model_metrics = ModelMetrics()
    performance_metrics = PerformanceMetrics()
    logger.info('Successfully initialized metric components')
except Exception as e:
    logger.error(f'Failed to initialize metric components: {str(e)}')
    raise

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("US Tree Dashboard", className="text-center mb-4"),
            html.P("Interactive visualization of US tree data with ML insights",
                   className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Environmental Metrics"),
                dbc.CardBody(id='environmental-metrics')
            ], className="mb-4")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Metrics"),
                dbc.CardBody(id='model-metrics')
            ], className="mb-4")
        ], md=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody(id='performance-metrics')
            ])
        ])
    ])
], fluid=True)

# Callback for environmental metrics
@app.callback(
    dash.dependencies.Output('environmental-metrics', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
@handle_exceptions
def update_environmental_metrics(_):
    return environmental_metrics.get_visualization()

# Callback for model metrics
@app.callback(
    dash.dependencies.Output('model-metrics', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
@handle_exceptions
def update_model_metrics(_):
    return model_metrics.get_visualization()

# Callback for performance metrics
@app.callback(
    dash.dependencies.Output('performance-metrics', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
@handle_exceptions
def update_performance_metrics(_):
    return performance_metrics.get_visualization()

# Error handling for the main application
@server.errorhandler(Exception)
def handle_error(error):
    logger.error(f'Unhandled error: {str(error)}')
    return ErrorConfig.get_error_response('system_error'), 500

if __name__ == '__main__':
    try:
        logger.info('Starting the application')
        app.run(host='0.0.0.0', port=8050, debug=Config.DEBUG)
    except Exception as e:
        logger.critical(f'Failed to start application: {str(e)}')
        raise