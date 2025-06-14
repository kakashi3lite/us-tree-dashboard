import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from src.metrics.environmental_metrics import EnvironmentalMetrics
from src.metrics.model_metrics import ModelMetrics
from src.metrics.performance_metrics import PerformanceMetrics

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create instances of metric classes
environmental_metrics = EnvironmentalMetrics()
model_metrics = ModelMetrics()
performance_metrics = PerformanceMetrics()

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
                dbc.CardBody([
                    # Environmental metrics visualization components
                ])
            ], className="mb-4")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Metrics"),
                dbc.CardBody([
                    # Model metrics visualization components
                ])
            ], className="mb-4")
        ], md=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody([
                    # Performance metrics visualization components
                ])
            ])
        ])
    ])
], fluid=True)

# For development and production
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)
