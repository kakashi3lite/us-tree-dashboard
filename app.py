import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from src.charitable_orgs import CHARITABLE_ORGS
from datetime import datetime, timedelta
import json
import logging
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING_CONFIG['level']),
    format=config.LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap"
    ],
    title=config.APP_TITLE
)

# For Gunicorn deployment
server = app.server

# Enhanced navigation bar with plant-lover theme
navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src="/assets/tree-icon.png", height="30px"), width="auto"),
                dbc.Col(dbc.NavbarBrand(f"🌿 {config.APP_NAME} - Interactive Biodiversity Dashboard", className="ms-2 fw-bold")),
            ], align="center", className="g-0"),
            href="#",
            style={"textDecoration": "none", "color": "inherit"},
        ),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("🌿 Explore Trees", href="#", className="text-light fw-semibold")),
            dbc.NavItem(dbc.NavLink("🌱 Plant Care", href="#", className="text-light fw-semibold")),
            dbc.NavItem(dbc.NavLink("💚 Get Involved", href="#", className="text-light fw-semibold")),
            dbc.NavItem(dbc.NavLink("📊 Data API", href="#api", className="text-light fw-semibold", disabled=not config.INTEGRATION_CONFIG['api_enabled'])),
        ], navbar=True, className="ms-auto"),
    ]),
    color="success",
    dark=True,
    sticky="top",
    className="shadow-sm",
    style={
        "background": "linear-gradient(90deg, #2E7D32 0%, #388E3C 50%, #4CAF50 100%)",
        "borderBottom": "3px solid #1B5E20"
    }
)

# Load charitable organizations from config or use defaults
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

# Use configuration for data directory
DATA_DIR = config.DATA_DIR
DATA_DIR.mkdir(exist_ok=True)

class PlantsWorldDashboard:
    """Main dashboard class for PlantsWorld - scalable and modular design"""
    
    def __init__(self):
        self.df_trees = None
        self.df_canopy = None
        self.gdf_counties = None
        self.state_options = []
        self.city_options = []
        self.charitable_orgs = CHARITABLE_ORGS.copy()
        self.datasets = self.load_datasets()
        self.load_data()
        
    def load_datasets(self) -> dict:
        """Load all available datasets using configuration"""
        datasets = {}
        
        # Load datasets based on configuration
        for dataset_key, dataset_config in config.DATASETS_CONFIG.items():
            try:
                latest_file = config.get_latest_dataset_file(dataset_key)
                if latest_file and latest_file.exists():
                    df = pd.read_csv(latest_file)
                    # Limit records if configured
                    max_records = config.UPDATE_CONFIG['max_records_per_dataset']
                    if len(df) > max_records:
                        df = df.head(max_records)
                        logger.info(f"Limited {dataset_key} to {max_records} records")
                    
                    datasets[dataset_key] = df
                    logger.info(f"Loaded {dataset_config['display_name']}: {len(df)} records from {latest_file.name}")
                else:
                    logger.warning(f"No files found for {dataset_config['display_name']}")
                    datasets[dataset_key] = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading {dataset_config['display_name']}: {e}")
                datasets[dataset_key] = pd.DataFrame()
        
        # If no real data found, create sample data
        if all(df.empty for df in datasets.values()):
            logger.info("No datasets found, creating sample data...")
            datasets = self.create_sample_datasets()
        
        return datasets
    
    def create_sample_datasets(self) -> dict:
        """Create sample datasets for demonstration"""
        logger.info("Creating sample datasets...")
        
        # Sample plant families data
        plant_families = pd.DataFrame({
            'family': ['Rosaceae', 'Asteraceae', 'Fabaceae', 'Poaceae', 'Orchidaceae'],
            'common_name': ['Rose Family', 'Sunflower Family', 'Legume Family', 'Grass Family', 'Orchid Family'],
            'estimated_species': [4828, 32913, 19500, 12074, 28000],
            'distribution': ['Worldwide', 'Worldwide', 'Worldwide', 'Worldwide', 'Worldwide']
        })
        
        # Sample biodiversity hotspots
        biodiversity_hotspots = pd.DataFrame({
            'hotspot_name': ['Amazon Rainforest', 'Congo Basin', 'Southeast Asian Rainforests', 'Atlantic Forest', 'Madagascar'],
            'country': ['Brazil', 'Democratic Republic of Congo', 'Indonesia', 'Brazil', 'Madagascar'],
            'latitude': [-3.4653, -0.2280, -0.7893, -14.2350, -18.7669],
            'longitude': [-62.2159, 15.8277, 113.9213, -51.9253, 46.8691],
            'endemic_species': [40000, 10000, 25000, 20000, 12000],
            'area_km2': [6700000, 3700000, 2500000, 1233875, 587041],
            'threat_level': ['High', 'High', 'Critical', 'Critical', 'Critical']
        })
        
        # Sample conservation status data
        conservation_status = pd.DataFrame({
            'species': ['Welwitschia mirabilis', 'Dionaea muscipula', 'Nepenthes rajah', 'Baobab tree', 'Giant Sequoia'],
            'common_name': ['Welwitschia', 'Venus Flytrap', 'Rajah Pitcher Plant', 'Baobab', 'Giant Sequoia'],
            'conservation_status': ['Vulnerable', 'Vulnerable', 'Critically Endangered', 'Least Concern', 'Endangered'],
            'population_trend': ['Decreasing', 'Decreasing', 'Decreasing', 'Stable', 'Increasing'],
            'main_threats': ['Climate change', 'Habitat loss', 'Over-collection', 'Climate change', 'Fire/Disease']
        })
        
        return {
            'plant_families': plant_families,
            'biodiversity_hotspots': biodiversity_hotspots,
            'conservation_status': conservation_status
        }

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
        """Create the main dashboard layout with plant-lover theme"""
        return html.Div([
            
            # Hero Section
            html.Div([
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H1([
                                f"🌿 Welcome to {config.APP_NAME} ",
                                html.Span("🌱", className="floating-leaf")
                            ], className="display-3 fw-bold mb-4", style={"textAlign": "center"}),
                            html.P(
                                config.APP_DESCRIPTION + ". Explore plant distributions, "
                                "learn about species diversity, and join the movement to protect our green planet! 🌍",
                                className="lead text-center mb-4", style={"fontSize": "1.3rem"}
                            ),
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-seedling me-2"), "Start Exploring"],
                                    color="success", size="lg", className="me-3",
                                    style={"borderRadius": "25px", "padding": "12px 30px"}
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-heart me-2"), "Plant Care Tips"],
                                    color="light", size="lg", outline=True,
                                    style={"borderRadius": "25px", "padding": "12px 30px", "color": "white", "borderColor": "white"}
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-globe me-2"), "API Access"],
                                    color="info", size="lg", outline=True, className="ms-2",
                                    style={"borderRadius": "25px", "padding": "12px 30px", "color": "white", "borderColor": "white"},
                                    disabled=not config.INTEGRATION_CONFIG['api_enabled']
                                ) if config.INTEGRATION_CONFIG['api_enabled'] else html.Div()
                            ], className="text-center")
                        ], md=12)
                    ])
                ])
            ], className="hero-section"),
            
            # Plant Care Tips Section
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H2("🌱 Daily Plant Wisdom", className="text-center mb-4 nature-gradient fw-bold"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4([html.I(className="fas fa-tint text-primary me-2"), "Watering Tips"], className="card-title"),
                                        html.P("💧 Water your plants early morning or late evening to reduce evaporation. "
                                              "Check soil moisture by inserting your finger 1-2 inches deep!", className="card-text")
                                    ])
                                ], className="plant-card mb-3")
                            ], md=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4([html.I(className="fas fa-sun text-warning me-2"), "Light Love"], className="card-title"),
                                        html.P("☀️ Most houseplants thrive in bright, indirect light. "
                                              "Rotate your plants weekly for even growth and happy leaves!", className="card-text")
                                    ])
                                ], className="plant-card mb-3")
                            ], md=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4([html.I(className="fas fa-leaf text-success me-2"), "Growth Boost"], className="card-title"),
                                        html.P("🌿 Talk to your plants! Studies show plants respond to vibrations. "
                                              "Plus, it's a great way to check for pests and health!", className="card-text")
                                    ])
                                ], className="plant-card mb-3")
                            ], md=4)
                        ])
                    ], md=12)
                ], className="mb-5"),
                
                # Interactive Controls Section
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4([html.I(className="fas fa-map-marked-alt text-success me-2"), "🗺️ Explore Tree Data"], 
                                       className="mb-0 fw-bold")
                            ]),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("🏛️ Select State:", className="fw-bold text-success"),
                                        dcc.Dropdown(
                                            id='state-dropdown',
                                            options=self.state_options,
                                            placeholder="🌲 Choose your state adventure...",
                                            className="mb-3",
                                            style={"borderRadius": "10px"}
                                        ),
                                    ], md=6),
                                    dbc.Col([
                                        html.Label("🏙️ Select City:", className="fw-bold text-success"),
                                        dcc.Dropdown(
                                            id='city-dropdown',
                                            options=self.city_options,
                                            placeholder="🌳 Pick your urban forest...",
                                            className="mb-3",
                                            style={"borderRadius": "10px"}
                                        ),
                                    ], md=6),
                                ]),
                                html.Label("🌿 Canopy Coverage Range:", className="fw-bold text-success"),
                                dcc.RangeSlider(
                                    id='canopy-slider',
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=[0, 100],
                                    marks={
                                        0: {'label': '🌱 0%', 'style': {'color': '#4CAF50'}},
                                        25: {'label': '🌿 25%', 'style': {'color': '#4CAF50'}},
                                        50: {'label': '🌳 50%', 'style': {'color': '#4CAF50'}},
                                        75: {'label': '🌲 75%', 'style': {'color': '#4CAF50'}},
                                        100: {'label': '🌴 100%', 'style': {'color': '#4CAF50'}}
                                    },
                                    className="mb-4"
                                ),
                            ])
                        ], className="plant-card")
                    ], md=12),
                ]),

                # Live Update Indicator
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Badge(
                                [html.I(className="fas fa-circle me-1"), "Live Data"],
                                color="success",
                                className="me-2"
                            ),
                            html.Span(id="live-indicator", className="text-muted small"),
                            html.Span(" | ", className="text-muted mx-2"),
                            html.Span(id="data-count", className="text-success fw-bold"),
                            html.Span(" | Next update in: ", className="text-muted mx-2"),
                            html.Span(id="countdown", className="text-primary fw-bold")
                        ], className="text-center p-2", style={
                            "background": "rgba(76, 175, 80, 0.1)",
                            "borderRadius": "15px",
                            "border": "1px solid rgba(76, 175, 80, 0.3)"
                        })
                    ], md=12)
                ], className="mt-4"),

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
                            children=html.Div(id='charts-container'),
                            color="#4CAF50"
                        )
                    ], md=12)
                ], className="mt-4"),
                
                # Charitable Organizations Section
                dbc.Row([
                    dbc.Col([
                        html.H2("💚 Support Tree Conservation", className="text-center mb-4 nature-gradient fw-bold"),
                        html.P("Join these amazing organizations in their mission to protect and plant trees worldwide! 🌍", 
                               className="text-center mb-4 lead"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5([html.I(className="fas fa-tree text-success me-2"), org['name']], className="card-title"),
                                        html.P(org['description'], className="card-text"),
                                        dbc.Button(
                                            [html.I(className="fas fa-external-link-alt me-2"), "Learn More"],
                                            href=org['learn_more'],
                                            target="_blank",
                                            color="success",
                                            outline=True,
                                            size="sm",
                                            style={"borderRadius": "15px"}
                                        )
                                    ])
                                ], className="plant-card mb-3 h-100")
                            ], md=4) for org in self.charitable_orgs[:6]  # Show first 6 organizations
                        ])
                    ], md=12)
                ], className="mt-5 mb-5"),
                
                # Interval component for live updates
                dcc.Interval(
                    id='interval-component',
                    interval=30*1000,  # Update every 30 seconds
                    n_intervals=0
                ),

                # Map
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id="loading-map",
                            type="circle",
                            children=html.Div(id='map-container'),
                            color="#4CAF50"
                        )
                    ], md=12)
                ], className="mt-4 mb-4"),
            ], fluid=True),
            
            # Footer
            html.Footer([
                html.Div([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.H5(f"🌿 {config.APP_NAME}", className="text-white mb-3"),
                                html.P("Connecting nature lovers with data-driven insights to protect our planet's green heritage.", 
                                       className="text-light mb-3"),
                                html.Div([
                                    dbc.Button(
                                        html.I(className="fab fa-github"),
                                        color="light", outline=True, size="sm", className="me-2",
                                        style={"borderRadius": "50%", "width": "40px", "height": "40px"}
                                    ),
                                    dbc.Button(
                                        html.I(className="fab fa-twitter"),
                                        color="light", outline=True, size="sm", className="me-2",
                                        style={"borderRadius": "50%", "width": "40px", "height": "40px"}
                                    ),
                                    dbc.Button(
                                        html.I(className="fas fa-envelope"),
                                        color="light", outline=True, size="sm",
                                        style={"borderRadius": "50%", "width": "40px", "height": "40px"}
                                    )
                                ])
                            ], md=4),
                            dbc.Col([
                                html.H6("🌱 Quick Links", className="text-white mb-3"),
                                html.Ul([
                                    html.Li(html.A("Plant Care Guide", href="#", className="text-light text-decoration-none")),
                                    html.Li(html.A("Tree Species Database", href="#", className="text-light text-decoration-none")),
                                    html.Li(html.A("Conservation Tips", href="#", className="text-light text-decoration-none")),
                                    html.Li(html.A("API Documentation", href="#", className="text-light text-decoration-none",
                                                  style={"display": "block" if config.INTEGRATION_CONFIG['api_enabled'] else "none"})),
                                    html.Li(html.A("Community Forum", href="#", className="text-light text-decoration-none"))
                                ], className="list-unstyled")
                            ], md=4),
                            dbc.Col([
                                html.H6("🌍 Get Involved", className="text-white mb-3"),
                                html.P("Every tree counts! Join our community of plant lovers and make a difference.", 
                                       className="text-light mb-3"),
                                dbc.Button(
                                    [html.I(className="fas fa-seedling me-2"), "Plant a Tree Today"],
                                    color="success", size="sm",
                                    style={"borderRadius": "20px"}
                                )
                            ], md=4)
                        ]),
                        html.Hr(className="my-4", style={"borderColor": "rgba(255,255,255,0.3)"}),
                        dbc.Row([
                            dbc.Col([
                                html.P(f"© {datetime.now().year} {config.APP_NAME}. Made with 💚 for our planet.", 
                                       className="text-center text-light mb-0")
                            ], md=12)
                        ])
                    ])
                ], style={
                    "background": "linear-gradient(135deg, #2E7D32 0%, #388E3C 50%, #4CAF50 100%)",
                    "padding": "40px 0 20px 0",
                    "marginTop": "50px"
                })
            ])
        ])

    def create_plant_themed_summary_cards(
        self, selected_state: Optional[str], selected_city: Optional[str]
    ) -> html.Div:
        """Create plant-lover themed summary statistics cards"""
        if selected_city:
            city_data = self.df_trees[self.df_trees['city'] == selected_city]
            total_trees = len(city_data)
            avg_dbh = city_data['dbh'].mean()
            unique_species = city_data['species'].nunique()
            health_score = min(95, max(60, avg_dbh * 2)) if not city_data.empty else 0
            
            cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-tree fa-2x text-success mb-2"),
                                html.H3(f"{total_trees:,}", className="card-title text-success fw-bold"),
                                html.P("🌳 Trees Discovered", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-success")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-ruler fa-2x text-primary mb-2"),
                                html.H3(f"{avg_dbh:.1f} cm", className="card-title text-primary fw-bold"),
                                html.P("📏 Average Trunk Size", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-primary")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-seedling fa-2x text-warning mb-2"),
                                html.H3(f"{unique_species}", className="card-title text-warning fw-bold"),
                                html.P("🌱 Species Diversity", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-warning")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-heart fa-2x text-danger mb-2"),
                                html.H3(f"{health_score:.1f}%", className="card-title text-danger fw-bold"),
                                html.P("💚 Health Score", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-danger")
                ], md=3),
            ], className="mb-4")
            
            return html.Div([cards])
            
        elif selected_state:
            state_data = self.df_canopy[self.df_canopy['state'] == selected_state]
            avg_canopy = state_data['canopy_pct'].mean()
            counties = len(state_data)
            max_canopy = state_data['canopy_pct'].max()
            
            cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-leaf fa-2x text-info mb-2"),
                                html.H3(f"{avg_canopy:.1f}%", className="card-title text-info fw-bold"),
                                html.P("🍃 Average Canopy", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-info")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-map fa-2x text-success mb-2"),
                                html.H3(f"{counties:,}", className="card-title text-success fw-bold"),
                                html.P("🗺️ Counties", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-success")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-crown fa-2x text-warning mb-2"),
                                html.H3(f"{max_canopy:.1f}%", className="card-title text-warning fw-bold"),
                                html.P("👑 Max Canopy", className="card-text text-muted")
                            ], className="text-center")
                        ])
                    ], className="plant-card border-warning")
                ], md=4),
            ], className="mb-4")
            
            return html.Div([cards])
            
        return html.Div([])

    def create_plant_themed_charts(self, df_trees, df_canopy):
        """Create plant-lover themed charts"""
        if df_trees.empty:
            return html.Div([
                dbc.Alert(
                    [html.I(className="fas fa-seedling me-2"), "🌱 No tree data available for the selected filters. Try adjusting your selection!"],
                    color="info",
                    className="text-center"
                )
            ])
        
        # Species distribution chart with plant theme
        species_counts = df_trees['species'].value_counts().head(10)
        species_fig = px.bar(
            x=species_counts.values,
            y=species_counts.index,
            orientation='h',
            title="🌳 Top Tree Species in Your Selection",
            color=species_counts.values,
            color_continuous_scale='Greens'
        )
        species_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Poppins',
            title_font_size=16,
            title_font_color='#2E7D32'
        )
        
        # DBH distribution with plant theme
        dbh_fig = px.histogram(
            df_trees,
            x='dbh',
            title="🌲 Tree Trunk Size Distribution (DBH)",
            color_discrete_sequence=['#4CAF50']
        )
        dbh_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Poppins',
            title_font_size=16,
            title_font_color='#2E7D32'
        )
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🌿 Species Diversity", className="mb-0 text-success")),
                    dbc.CardBody([
                        dcc.Graph(figure=species_fig)
                    ])
                ], className="plant-card")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📏 Tree Size Analysis", className="mb-0 text-success")),
                    dbc.CardBody([
                        dcc.Graph(figure=dbh_fig)
                    ])
                ], className="plant-card")
            ], md=6)
        ])
    
    def create_plant_themed_map(self, df_trees, df_canopy):
        """Create plant-lover themed map"""
        if df_trees.empty:
            return html.Div([
                dbc.Alert(
                    [html.I(className="fas fa-map me-2"), "🗺️ No location data available. Select a region to explore!"],
                    color="warning",
                    className="text-center"
                )
            ])
        
        # Create a simple scatter map with plant theme
        fig = px.scatter_mapbox(
            df_trees,
            lat='latitude',
            lon='longitude',
            color='dbh',
            size='dbh',
            hover_data=['species', 'dbh'],
            color_continuous_scale='Greens',
            title="🌍 Tree Locations & Sizes",
            mapbox_style='open-street-map',
            zoom=10
        )
        
        fig.update_layout(
            height=500,
            font_family='Poppins',
            title_font_size=16,
            title_font_color='#2E7D32'
        )
        
        return dbc.Card([
            dbc.CardHeader(html.H5("🗺️ Interactive Tree Map", className="mb-0 text-success")),
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ], className="plant-card")

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

# Initialize dashboard
dashboard = PlantsWorldDashboard()

# Main app layout
app.layout = dashboard.create_layout()

# Setup callbacks will be added here

# Callback to update stats
@app.callback(
    [Output("total-species", "children"),
     Output("total-families", "children"),
     Output("total-hotspots", "children"),
     Output("threatened-species", "children")],
    [Input("family-dropdown", "value"),
     Input("conservation-filter", "value")]
)
def update_stats(family, conservation):
    # Calculate stats from sample datasets
    plant_families = dashboard.datasets.get('plant_families', pd.DataFrame())
    biodiversity_hotspots = dashboard.datasets.get('biodiversity_hotspots', pd.DataFrame())
    conservation_status = dashboard.datasets.get('conservation_status', pd.DataFrame())
    
    total_species = plant_families['estimated_species'].sum() if not plant_families.empty else 0
    total_families = len(plant_families) if not plant_families.empty else 0
    total_hotspots = len(biodiversity_hotspots) if not biodiversity_hotspots.empty else 0
    threatened = len(conservation_status[conservation_status['conservation_status'].isin(['Critically Endangered', 'Endangered', 'Vulnerable'])]) if not conservation_status.empty else 0
    
    return f"{total_species:,}", str(total_families), str(total_hotspots), str(threatened)

# Callback to update biodiversity map
@app.callback(
    Output("biodiversity-map", "figure"),
    [Input("family-dropdown", "value"),
     Input("conservation-filter", "value")]
)
def update_biodiversity_map(family, conservation):
    # Get biodiversity hotspots data
    hotspots = dashboard.datasets.get('biodiversity_hotspots', pd.DataFrame())
    
    if hotspots.empty:
        # Create empty map if no data
        fig = go.Figure(go.Scattermapbox())
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=1,
            mapbox_center={"lat": 0, "lon": 0},
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=500
        )
        return fig
    
    # Create biodiversity hotspots map
    fig = px.scatter_mapbox(
        hotspots,
        lat='latitude',
        lon='longitude',
        size='endemic_species',
        color='threat_level',
        hover_name='hotspot_name',
        hover_data=['country', 'endemic_species'],
        color_discrete_map={
            'Critical': '#d32f2f',
            'High': '#f57c00',
            'Medium': '#fbc02d',
            'Low': '#388e3c'
        },
        zoom=1,
        title="Global Biodiversity Hotspots"
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=500
    )
    
    return fig

# Callback to update plant families chart
@app.callback(
    Output("families-chart", "figure"),
    [Input("family-dropdown", "value")]
)
def update_families_chart(selected_family):
    plant_families = dashboard.datasets.get('plant_families', pd.DataFrame())
    
    if plant_families.empty:
        return go.Figure()
    
    fig = px.bar(
        plant_families,
        x='family',
        y='estimated_species',
        title='Plant Families by Species Count',
        color='estimated_species',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        xaxis_title='Plant Family',
        yaxis_title='Estimated Species Count',
        height=400
    )
    
    return fig

# Callback to update conservation chart
@app.callback(
    Output("conservation-chart", "figure"),
    [Input("conservation-filter", "value")]
)
def update_conservation_chart(conservation_filter):
    conservation_data = dashboard.datasets.get('conservation_status', pd.DataFrame())
    
    if conservation_data.empty:
        return go.Figure()
    
    status_counts = conservation_data['conservation_status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Conservation Status Distribution',
        color_discrete_map={
            'Critically Endangered': '#d32f2f',
            'Endangered': '#f57c00',
            'Vulnerable': '#fbc02d',
            'Least Concern': '#388e3c'
        }
    )
    
    fig.update_layout(height=400)
    
    return fig

# Callback to populate family dropdown
@app.callback(
    Output("family-dropdown", "options"),
    [Input("family-dropdown", "id")]  # Dummy input to trigger on load
)
def populate_family_dropdown(_):
    plant_families = dashboard.datasets.get('plant_families', pd.DataFrame())
    
    if plant_families.empty:
        return []
    
    return [{'label': row['common_name'], 'value': row['family']} 
            for _, row in plant_families.iterrows()]

# Callback to update data table
@app.callback(
    Output("data-table-container", "children"),
    [Input("btn-families", "n_clicks"),
     Input("btn-conservation", "n_clicks"),
     Input("btn-hotspots", "n_clicks")]
)
def update_data_table(btn_families, btn_conservation, btn_hotspots):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Default to plant families
        dataset = dashboard.datasets.get('plant_families', pd.DataFrame())
        title = "Plant Families Data"
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "btn-families":
            dataset = dashboard.datasets.get('plant_families', pd.DataFrame())
            title = "Plant Families Data"
        elif button_id == "btn-conservation":
            dataset = dashboard.datasets.get('conservation_status', pd.DataFrame())
            title = "Conservation Status Data"
        elif button_id == "btn-hotspots":
            dataset = dashboard.datasets.get('biodiversity_hotspots', pd.DataFrame())
            title = "Biodiversity Hotspots Data"
        else:
            dataset = pd.DataFrame()
            title = "No Data Available"
    
    if dataset.empty:
        return html.Div([
            html.H5(title, className="mb-3"),
            html.P("No data available for this category.", className="text-muted")
        ])
    
    return html.Div([
        html.H5(title, className="mb-3"),
        dash_table.DataTable(
            data=dataset.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dataset.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            page_size=10,
            sort_action="native",
            filter_action="native"
        )
    ])

# Callback to update species details
@app.callback(
    Output("species-details", "children"),
    [Input("family-dropdown", "value"),
     Input("conservation-filter", "value")]
)
def update_species_details(selected_family, conservation_filter):
    if not selected_family:
        return html.Div([
            html.P("Please select a plant family to explore species details.", 
                   className="text-muted text-center mt-4")
        ])
    
    plant_families = dashboard.datasets.get('plant_families', pd.DataFrame())
    conservation_data = dashboard.datasets.get('conservation_status', pd.DataFrame())
    
    if plant_families.empty:
        return html.Div([
            html.P("No plant family data available.", className="text-muted text-center mt-4")
        ])
    
    # Get selected family info
    family_info = plant_families[plant_families['family'] == selected_family]
    
    if family_info.empty:
        return html.Div([
            html.P("Family information not found.", className="text-muted text-center mt-4")
        ])
    
    family_row = family_info.iloc[0]
    
    # Filter conservation data if applicable
    filtered_conservation = conservation_data
    if conservation_filter != 'all' and not conservation_data.empty:
        filtered_conservation = conservation_data[
            conservation_data['conservation_status'] == conservation_filter
        ]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5(f"🌿 {family_row['common_name']} ({family_row['family']})", className="mb-0")
                ]),
                dbc.CardBody([
                    html.P(f"**Estimated Species:** {family_row['estimated_species']:,}", className="mb-2"),
                    html.P(f"**Distribution:** {family_row['distribution']}", className="mb-2"),
                    html.Hr(),
                    html.H6("Family Characteristics:", className="fw-bold"),
                    html.Ul([
                        html.Li("Diverse morphological features"),
                        html.Li("Wide ecological adaptations"),
                        html.Li("Important economic and ecological value")
                    ])
                ])
            ], className="shadow-sm")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🔬 Research Insights", className="mb-0")
                ]),
                dbc.CardBody([
                    html.P("**Recent Studies:**", className="fw-bold mb-2"),
                    html.Ul([
                        html.Li("Phylogenetic relationships within the family"),
                        html.Li("Climate change adaptation strategies"),
                        html.Li("Conservation genomics applications")
                    ]),
                    html.Hr(),
                    html.P("**Conservation Priority:**", className="fw-bold mb-2"),
                    dbc.Badge(
                        "High" if family_row['estimated_species'] > 10000 else "Medium",
                        color="danger" if family_row['estimated_species'] > 10000 else "warning",
                        className="me-2"
                    ),
                    html.P("Based on species diversity and threat assessment", 
                           className="text-muted small mt-2")
                ])
            ], className="shadow-sm")
        ], md=6)
    ])

if __name__ == '__main__':
    # Validate configuration
    config.validate_config()
    
    # Initialize API if enabled
    if config.INTEGRATION_CONFIG['api_enabled']:
        try:
            import api
            api.init_api(app)
            logging.info("API initialized successfully")
        except ImportError as e:
            logging.warning(f"API module not available: {e}")
        except Exception as e:
            logging.error(f"Failed to initialize API: {e}")
    
    # Run the app
    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT
    )
