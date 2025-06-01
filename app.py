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

# Load environment variables
load_dotenv()

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="AroundTheTrees Dashboard"
)

# For Gunicorn deployment
server = app.server

# Define charitable organizations
CHARITABLE_ORGS = [
    {
        'name': 'TreePeople',
        'description': 'Inspiring and supporting the people of Southern California to plant and care for trees.',
        'impact': 'Over 3 million trees planted with 3 million volunteers across Southern California',
        'donate_url': 'https://treepeople.org/?campaign=430396',
        'learn_more': 'https://treepeople.org/about-us/'
    },
    {
        'name': 'Arbor Day Foundation',
        'description': 'A global nonprofit inspiring people to plant, nurture, and celebrate trees.',
        'impact': 'Over 500 million trees planted across 60+ countries',
        'donate_url': 'https://donate.arborday.org/',
        'learn_more': 'https://www.arborday.org/about'
    },
    {
        'name': 'Trees for the Future',
        'description': 'Training farmers in sustainable agroforestry to restore ecosystems and change lives.',
        'impact': 'UN World Restoration Flagship organization helping farmers plant Forest Gardens',
        'donate_url': 'https://trees.org/?form=FUNRTALHRNR',
        'learn_more': 'https://trees.org/about-us/'
    }
]

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
        self.charitable_orgs = [
            {
                'name': 'TreePeople',
                'description': 'Inspiring and supporting the people of Southern California to plant and care for trees.',
                'impact': 'Over 3 million trees planted with 3 million volunteers across Southern California',
                'donate_url': 'https://treepeople.org/?campaign=430396',
                'learn_more': 'https://treepeople.org/about-us/'
            },
            {
                'name': 'Arbor Day Foundation',
                'description': 'A global nonprofit inspiring people to plant, nurture, and celebrate trees.',
                'impact': 'Over 500 million trees planted across 60+ countries',
                'donate_url': 'https://donate.arborday.org/',
                'learn_more': 'https://www.arborday.org/about'
            },
            {
                'name': 'Trees for the Future',
                'description': 'Training farmers in sustainable agroforestry to restore ecosystems and change lives.',
                'impact': 'UN World Restoration Flagship organization helping farmers plant Forest Gardens',
                'donate_url': 'https://trees.org/?form=FUNRTALHRNR',
                'learn_more': 'https://trees.org/about-us/'
            }
        ]
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
            # Charitable Organizations Banner
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Make an Impact - Support Tree Planting Organizations", className="text-center mt-4"),
                            html.Div([
                                dbc.Row([
                                    *[dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H5(org["name"], className="card-title"),
                                                html.P(org["description"], className="card-text small"),
                                                html.P([
                                                    "Impact: ",
                                                    html.Strong(org["impact"])
                                                ], className="card-text text-success"),
                                                html.Div([
                                                    dbc.Button(
                                                        "🌱 Donate",
                                                        href=org["donate_url"],
                                                        target="_blank",
                                                        color="success",
                                                        size="sm",
                                                        className="me-2",
                                                    ),
                                                    dbc.Button(
                                                        "Learn More",
                                                        href=org["learn_more"],
                                                        target="_blank",
                                                        color="secondary",
                                                        size="sm",
                                                    ),
                                                ], className="d-flex justify-content-center")
                                            ])
                                        ], className="h-100 shadow-sm")
                                    ], md=4) for org in CHARITABLE_ORGS]
                                ], className="g-4 mb-4")
                            ])
                        ], className="py-3")
                    ])
                ])
            ], fluid=True, className="bg-light border-bottom"),            # Main content
            dbc.Container([
                # Charitable Organizations Banner
                dbc.Row([
                    dbc.Col([
                        dbc.Alert([
                            html.H4("Support Tree Planting Organizations", className="text-center alert-heading"),
                            html.Hr(),
                            html.Div([
                                html.A("🌱 TreePeople", href="https://treepeople.org/?campaign=430396", target="_blank", className="btn btn-success me-2"),
                                html.A("🌲 Arbor Day Foundation", href="https://donate.arborday.org/", target="_blank", className="btn btn-success me-2"),
                                html.A("🌳 Trees for the Future", href="https://trees.org/?form=FUNRTALHRNR", target="_blank", className="btn btn-success")
                            ], className="text-center")
                        ], color="light", className="mb-4")
                    ], width=12)
                ]),
                # Impact and Donation Section
                dbc.Row([
                    dbc.Col([
                        html.H3("Make an Impact", className="text-center mt-4 mb-4"),
                        html.P(
                            "Support organizations making a difference in tree planting and forest conservation.",
                            className="text-center mb-4"
                        ),
                        dbc.Row([
                            *[
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4(org['name'], className="card-title"),
                                            html.P(org['description'], className="card-text"),
                                            html.P(f"Impact: {org['impact']}", className="card-text text-success"),
                                            dbc.Button(
                                                "Donate", 
                                                href=org['donate_url'],
                                                color="success",
                                                className="me-2",
                                                target="_blank"  # Opens in new tab
                                            ),
                                            dbc.Button(
                                                "Learn More",
                                                href=org['learn_more'],
                                                color="secondary",
                                                target="_blank"  # Opens in new tab
                                            ),
                                        ])
                                    ], className="h-100 mb-4")
                                ], md=4)
                                for org in self.charitable_orgs
                            ]
                        ])
                    ])
                ],

                # Existing controls section
                dbc.Row([
                    # Controls
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
                ],

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

            ], fluid=True),            # Footer
            dbc.Container([
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.P(
                            ["Data sources: Dryad (63-city tree inventory), ",
                             "USDA Forest Service Tree Canopy Cover dataset"],
                            className="text-muted"
                        ),
                        html.Div([
                            html.P("Make an impact - Support tree planting organizations:", className="text-center mt-4"),
                            html.Div([
                                dbc.Button("🌱 Support TreePeople", href="https://treepeople.org/?campaign=430396", target="_blank", color="success", className="me-2 mb-2"),
                                dbc.Button("🌲 Support Arbor Day Foundation", href="https://donate.arborday.org/", target="_blank", color="success", className="me-2 mb-2"),
                                dbc.Button("🌳 Support Trees for the Future", href="https://trees.org/?form=FUNRTALHRNR", target="_blank", color="success", className="mb-2")
                            ], className="text-center")
                        ])
                    ])
                ])
            ], fluid=True)
        ])

    def create_summary_cards(
        self, selected_state: Optional[str], selected_city: Optional[str]
    ) -> List[dbc.Card]:
        """Create summary statistics cards"""
        if selected_city:
            city_data = self.df_trees[self.df_trees['city'] == selected_city]
            return [
                dbc.Card([
                    dbc.CardBody([
                        html.H4(selected_city, className="card-title"),
                        html.P(f"Total Trees: {len(city_data):,}"),
                        html.P(f"Unique Species: {city_data['species'].nunique():,}"),
                        html.P(f"Average DBH: {city_data['dbh'].mean():.1f} cm")
                    ])
                ], className="mb-3")
            ]
        elif selected_state:
            state_data = self.df_canopy[self.df_canopy['state'] == selected_state]
            return [
                dbc.Card([
                    dbc.CardBody([
                        html.H4(selected_state, className="card-title"),
                        html.P(f"Average Canopy: {state_data['canopy_pct'].mean():.1f}%"),
                        html.P(f"Counties: {len(state_data):,}"),
                        html.P(f"Max Canopy: {state_data['canopy_pct'].max():.1f}%")
                    ])
                ], className="mb-3")
            ]
        return []

    def create_species_chart(self, city_data: pd.DataFrame) -> dcc.Graph:
        """Create species distribution chart"""
        species_counts = city_data['species'].value_counts().head(10)
        fig = px.bar(
            x=species_counts.values,
            y=species_counts.index,
            orientation='h',
            title='Top 10 Species Distribution',
            labels={'x': 'Count', 'y': 'Species'}
        )
        fig.update_layout(height=400)
        return dcc.Graph(figure=fig)

    def create_health_chart(self, city_data: pd.DataFrame) -> dcc.Graph:
        """Create health distribution chart"""
        health_counts = city_data['health'].value_counts()
        fig = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            title='Tree Health Distribution'
        )
        fig.update_layout(height=400)
        return dcc.Graph(figure=fig)

    def create_map(
        self, selected_state: Optional[str], selected_city: Optional[str],
        canopy_range: List[float]
    ) -> dcc.Graph:
        """Create the interactive map"""
        if selected_city:
            # City-level tree point map
            city_data = self.df_trees[self.df_trees['city'] == selected_city]
            fig = px.scatter_mapbox(
                city_data,
                lat='latitude',
                lon='longitude',
                color='health',
                hover_data=['species', 'dbh'],
                zoom=11,
                height=600,
                title=f'Tree Locations in {selected_city}'
            )
        elif selected_state:
            # State-level canopy choropleth
            state_data = self.df_canopy[
                (self.df_canopy['state'] == selected_state) &
                (self.df_canopy['canopy_pct'] >= canopy_range[0]) &
                (self.df_canopy['canopy_pct'] <= canopy_range[1])
            ]
            fig = px.choropleth_mapbox(
                state_data,
                geojson=None,  # TODO: Add actual county geometries
                locations=state_data.index,
                color='canopy_pct',
                hover_name='county',
                color_continuous_scale="Greens",
                range_color=[0, 100],
                zoom=6,
                height=600,
                title=f'Canopy Coverage in {selected_state}'
            )
        else:
            # Default US map
            fig = go.Figure(go.Scattermapbox())
            fig.update_layout(height=600)

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_accesstoken=os.getenv('MAPBOX_TOKEN')
        )
        return dcc.Graph(figure=fig)

# Initialize the dashboard
dashboard = TreeDashboard()

# Set the layout
app.layout = dashboard.create_layout()

@app.callback(
    [Output('summary-cards', 'children'),
     Output('charts-container', 'children'),
     Output('map-container', 'children')],
    [Input('state-dropdown', 'value'),
     Input('city-dropdown', 'value'),
     Input('canopy-slider', 'value')]
)
def update_dashboard(
    selected_state: Optional[str],
    selected_city: Optional[str],
    canopy_range: List[float]
) -> Tuple[List[dbc.Card], List[dcc.Graph], dcc.Graph]:
    """Update all dashboard components based on selection"""
    # Create summary cards
    summary_cards = dashboard.create_summary_cards(selected_state, selected_city)

    # Create charts if a city is selected
    charts = []
    if selected_city:
        city_data = dashboard.df_trees[dashboard.df_trees['city'] == selected_city]
        charts = [
            dbc.Row([
                dbc.Col(dashboard.create_species_chart(city_data), md=6),
                dbc.Col(dashboard.create_health_chart(city_data), md=6)
            ])
        ]

    # Create map
    map_component = dashboard.create_map(selected_state, selected_city, canopy_range)

    return summary_cards, charts, map_component

if __name__ == '__main__':
    app.run_server(debug=True)
