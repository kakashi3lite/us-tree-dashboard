import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from src.charitable_orgs import CHARITABLE_ORGS

# Import EnhanceX integration
from src.app_enhancex_integration import apply_enhancex_to_app

# Load environment variables
load_dotenv()

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    title="US Tree Dashboard"
)

# For Gunicorn deployment
server = app.server

# Navigation bar with links
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.I(className="fas fa-tree me-2"),
                dbc.NavbarBrand("US Tree Dashboard", className="ms-2")
            ]),
        ], align="center"),
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Overview", href="#overview")),
                    dbc.NavItem(dbc.NavLink("Distribution", href="#distribution")),
                    dbc.NavItem(dbc.NavLink("Species", href="#species")),
                    dbc.NavItem(dbc.NavLink("Health", href="#health")),
                    dbc.NavItem(dbc.NavLink("Organizations", href="#organizations")),
                    dbc.NavItem(dbc.NavLink("Analysis", href="#analysis")),
                ], navbar=True)
            ])
        ])
    ]),
    color="primary",
    dark=True,
    className="mb-4"
)

# Charitable organizations are imported from src.charitable_orgs

# Ensure the data directory exists
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

class TreeDashboard:
    def __init__(self):
        self.df_trees = None
        self.df_canopy = None
        self.gdf_counties = None
        self.state_options = []
        self.city_options = []
        self.charitable_orgs = CHARITABLE_ORGS
        self.load_data()

    def load_data(self):
        """Load and prepare all necessary datasets"""
        # TODO: Replace with actual data loading when we have the files
        # For now, we'll create sample data
        self.create_sample_data()
        self.prepare_dropdown_options()

    def create_sample_data(self):
        """Create sample data for development"""
        # Sample tree data
        self.df_trees = pd.DataFrame({
            'city': ['Boston', 'New York', 'Chicago'] * 1000,
            'species': ['Maple', 'Oak', 'Pine', 'Birch', 'Elm'] * 600,
            'health': ['Good', 'Fair', 'Poor'] * 1000,
            'latitude': np.random.uniform(25, 49, 3000),
            'longitude': np.random.uniform(-125, -70, 3000),
            'dbh': np.random.uniform(10, 100, 3000)
        })

        # Sample canopy data
        self.df_canopy = pd.DataFrame({
            'state': ['MA', 'NY', 'IL'] * 20,
            'county': [f'County_{i}' for i in range(60)],
            'canopy_pct': np.random.uniform(10, 60, 60)
        })

    def prepare_dropdown_options(self):
        """Prepare options for dropdown menus"""
        self.state_options = [
            {'label': state, 'value': state}
            for state in sorted(self.df_canopy['state'].unique())
        ]
        self.city_options = [
            {'label': city, 'value': city}
            for city in sorted(self.df_trees['city'].unique())
        ]

    def create_layout(self) -> html.Div:
        """Create the main dashboard layout"""
        return html.Div([            # Header
            dbc.Navbar(
                dbc.Container([
                    html.A(
                        dbc.Row([
                            dbc.Col(html.H2("🌳 AroundTheTrees")),
                        ]),
                        href="#",
                        style={"textDecoration": "none"},
                    )
                ]),
                color="success",
                dark=True,
            ),
            # Main content
            dbc.Container([
                # Controls section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Select Region", className="mt-4"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("State:"),
                                    dcc.Dropdown(
                                        id='state-dropdown',
                                        options=self.state_options,
                                        placeholder="Select a state...",
                                        className="mb-2"
                                    ),
                                ], md=6),
                                dbc.Col([
                                    html.Label("City:"),
                                    dcc.Dropdown(
                                        id='city-dropdown',
                                        options=self.city_options,
                                        placeholder="Select a city...",
                                        className="mb-2"
                                    ),
                                ], md=6),
                            ]),
                            html.Label("Canopy Coverage Range:"),
                            dcc.RangeSlider(
                                id='canopy-slider',
                                min=0,
                                max=100,
                                step=1,
                                value=[0, 100],
                                marks={
                                    0: '0%',
                                    25: '25%',
                                    50: '50%',
                                    75: '75%',
                                    100: '100%'
                                },
                                className="mb-4"
                            ),
                        ], className="p-3 border rounded")
                    ], md=12),
                ]),

                # Summary Cards
                dbc.Row([
                    dbc.Col(id='summary-cards', md=12)
                ], className="mt-4"),

                # Charts
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-charts",
                            type="circle",
                            children=html.Div(id='charts-container')
                        )
                    ], md=12)
                ], className="mt-4"),

                # Map
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-map",
                            type="circle",
                            children=html.Div(id='map-container')
                        )
                    ], md=12)
                ], className="mt-4 mb-4"),
            ], fluid=True),

            # Footer
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P([
                            "© 2023 AroundTheTrees | ",
                            html.A("Data Sources", href="#", className="text-decoration-none"),
                            " | ",
                            html.A("About", href="#", className="text-decoration-none")
                        ], className="text-center text-muted")
                    ])
                ])
            ], fluid=True)
        ])

    def create_summary_cards(self, state=None, city=None) -> List[dbc.Col]:
        """Create summary cards with key metrics"""
        # Filter data based on selections
        filtered_trees = self.df_trees
        if city:
            filtered_trees = filtered_trees[filtered_trees['city'] == city]
        
        # Calculate metrics
        total_trees = len(filtered_trees)
        avg_dbh = filtered_trees['dbh'].mean()
        health_counts = filtered_trees['health'].value_counts(normalize=True) * 100
        good_health_pct = health_counts.get('Good', 0)
        
        # Create cards
        cards = [
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_trees:,}", className="card-title"),
                        html.P("Total Trees", className="card-text")
                    ])
                ], color="success", inverse=True)
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{avg_dbh:.1f} cm", className="card-title"),
                        html.P("Average Diameter", className="card-text")
                    ])
                ], color="info", inverse=True)
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{good_health_pct:.1f}%", className="card-title"),
                        html.P("Trees in Good Health", className="card-text")
                    ])
                ], color="warning", inverse=True)
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(filtered_trees['species'].unique())}", className="card-title"),
                        html.P("Unique Species", className="card-text")
                    ])
                ], color="primary", inverse=True)
            ], md=3)
        ]
        
        return cards

    def create_charts(self, state=None, city=None) -> List[dbc.Col]:
        """Create charts based on filtered data"""
        # Filter data based on selections
        filtered_trees = self.df_trees
        if city:
            filtered_trees = filtered_trees[filtered_trees['city'] == city]
        
        # Species distribution chart
        species_counts = filtered_trees['species'].value_counts().nlargest(5)
        species_fig = px.bar(
            x=species_counts.index,
            y=species_counts.values,
            labels={'x': 'Species', 'y': 'Count'},
            title="Top 5 Tree Species"
        )
        species_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        
        # Health distribution chart
        health_counts = filtered_trees['health'].value_counts()
        health_fig = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            title="Tree Health Distribution"
        )
        health_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        
        # Diameter distribution chart
        dbh_fig = px.histogram(
            filtered_trees,
            x="dbh",
            nbins=20,
            labels={'dbh': 'Diameter at Breast Height (cm)'},
            title="Tree Size Distribution"
        )
        dbh_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        
        # Create chart columns
        chart_columns = [
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=species_fig)
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=health_fig)
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=dbh_fig)
                    ])
                ])
            ], md=4)
        ]
        
        return chart_columns

    def create_map(self, state=None, city=None, canopy_range=None) -> html.Div:
        """Create an interactive map with tree locations"""
        # Filter data based on selections
        filtered_trees = self.df_trees
        if city:
            filtered_trees = filtered_trees[filtered_trees['city'] == city]
        
        # Create map figure
        fig = px.scatter_mapbox(
            filtered_trees.sample(min(1000, len(filtered_trees))),  # Sample to improve performance
            lat="latitude",
            lon="longitude",
            color="health",
            size="dbh",
            size_max=15,
            zoom=4,
            mapbox_style="carto-positron",
            title="Tree Locations",
            hover_data=["species", "dbh", "health"]
        )
        
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return html.Div([
            dbc.Card([
                dbc.CardHeader("Tree Distribution Map"),
                dbc.CardBody([
                    dcc.Graph(figure=fig)
                ])
            ])
        ])

# Create dashboard instance
dashboard = TreeDashboard()

# Set app layout
app.layout = dashboard.create_layout()

# Callback to update summary cards
@app.callback(
    Output("summary-cards", "children"),
    [Input("state-dropdown", "value"),
     Input("city-dropdown", "value")]
)
def update_summary_cards(state, city):
    return dashboard.create_summary_cards(state, city)

# Callback to update charts
@app.callback(
    Output("charts-container", "children"),
    [Input("state-dropdown", "value"),
     Input("city-dropdown", "value")]
)
def update_charts(state, city):
    chart_columns = dashboard.create_charts(state, city)
    return dbc.Row(chart_columns) if chart_columns else html.Div("No data available for the selected filters.")

# Callback to update map
@app.callback(
    Output("map-container", "children"),
    [Input("state-dropdown", "value"),
     Input("city-dropdown", "value"),
     Input("canopy-slider", "value")]
)
def update_map(state, city, canopy_range):
    return dashboard.create_map(state, city, canopy_range)

# Callback to update city options based on selected state
@app.callback(
    Output("city-dropdown", "options"),
    [Input("state-dropdown", "value")]
)
def update_city_options(selected_state):
    if selected_state:
        # Filter cities by state if we have that data
        # For now, return all cities
        return dashboard.city_options
    return dashboard.city_options

# Apply EnhanceX to the app
app = apply_enhancex_to_app(app)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)