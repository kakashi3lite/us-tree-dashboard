import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dash  # Added after refactor ensuring availability before use
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State


import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
try:  # Optional dependency (may be absent in minimal test env)
    from flask_caching import Cache  # type: ignore
except ImportError:  # pragma: no cover - fallback only when package missing
    class Cache:  # Minimal no-op cache shim
        def __init__(self, app, config=None):
            self.app = app
            self.config = config or {}

        def cached(self, *dargs, **dkwargs):
            def decorator(func):
                return func
            return decorator

        # Provide alias used in code (@cache.memoize)
        memoize = cached

        def clear(self):  # pragma: no cover - trivial
            return None
try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - fallback if not installed
    import json  # noqa: F401
    class _OrjsonShim:  # simple shim with dumps returning bytes-like interface
        @staticmethod
        def dumps(data, option=None):
            return json.dumps(data).encode()
    orjson = _OrjsonShim()  # type: ignore

# Import charitable organizations with error handling
try:
    from src.charitable_orgs import CHARITABLE_ORGS
except ImportError as e:
    logging.warning(f"Could not import charitable orgs: {e}")
    CHARITABLE_ORGS = []

# Import VOSviewer integration components
try:
    from src.components import create_vosviewer_network, register_vosviewer_callbacks
    from src.vosviewer_integration import integrate_vosviewer_into_dashboard, create_vosviewer_navbar_item
    HAS_VOSVIEWER = True
except ImportError as e:
    logging.warning(f"Could not import VOSviewer components: {e}")
    HAS_VOSVIEWER = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tracing (OpenTelemetry) loaded AFTER logger configured
# ---------------------------------------------------------------------------
try:  # Optional tracing dependencies
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - dependency absent
    _OTEL_AVAILABLE = False

if _OTEL_AVAILABLE and os.getenv("ENABLE_TRACING", "0") == "1":  # pragma: no cover - optional path
    try:
        resource = Resource.create({"service.name": "us-tree-dashboard"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        logger.info("✅ OpenTelemetry tracing initialized")
    except Exception as _otel_err:
        logger.warning(f"OpenTelemetry initialization skipped: {_otel_err}")
elif not _OTEL_AVAILABLE:
    logger.debug("OpenTelemetry packages not installed; tracing disabled")

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    title="US Tree Dashboard"
)

# For Gunicorn deployment
server = app.server

# ---------------------------------------------------------------------------
# Performance: Caching layer (SimpleCache by default; configurable via env)
# ---------------------------------------------------------------------------
cache_config = {
    "CACHE_TYPE": os.getenv("CACHE_TYPE", "SimpleCache"),
    "CACHE_DEFAULT_TIMEOUT": int(os.getenv("CACHE_DEFAULT_TIMEOUT", "300")),
}
cache = Cache(server, config=cache_config)

def _orjson_dumps(data):
    return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()

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
                    dbc.NavItem(dbc.NavLink("Networks", href="#networks")) if HAS_VOSVIEWER else None,
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
        self._data_loaded = False
        self.load_data()

    def load_data(self):
        """Load and prepare all necessary datasets with error handling"""
        try:
            # Try to load real data first
            self._load_real_data()
        except Exception as e:
            logger.warning(f"Could not load real data: {e}. Using sample data.")
            self.create_sample_data()
        finally:
            self.prepare_dropdown_options()
            self._data_loaded = True

    def _load_real_data(self):
        """Attempt to load real datasets from data directory"""
        data_dir = Path(__file__).parent / "data"
        
        # Load tree data from GBIF plants if available
        gbif_file = data_dir / "gbif_plants_20250614.csv"
        if gbif_file.exists():
            gbif_df = pd.read_csv(gbif_file)
            # Transform GBIF data to tree format
            self.df_trees = self._transform_gbif_to_trees(gbif_df)
            logger.info(f"Loaded {len(self.df_trees)} records from GBIF data")
        else:
            raise FileNotFoundError("GBIF plants data not found")
            
        # Load families data for canopy approximation
        families_file = data_dir / "plant_families_20250614.csv"
        if families_file.exists():
            families_df = pd.read_csv(families_file)
            self.df_canopy = self._transform_families_to_canopy(families_df)
            logger.info(f"Generated {len(self.df_canopy)} canopy records from families data")
        else:
            raise FileNotFoundError("Plant families data not found")

    def _transform_gbif_to_trees(self, gbif_df: pd.DataFrame) -> pd.DataFrame:
        """Transform GBIF data to tree dashboard format"""
        # Filter for US data if country column exists, otherwise use all data
        if 'country' in gbif_df.columns:
            us_data = gbif_df[gbif_df['country'] == 'United States'].copy()
            if us_data.empty:
                # If no US data, use all data as sample
                us_data = gbif_df.copy()
        else:
            us_data = gbif_df.copy()
        
        # Ensure we have some data to work with
        if us_data.empty:
            # Create minimal sample data if no real data
            us_data = pd.DataFrame({
                'scientificName': ['Quercus alba', 'Acer rubrum', 'Pinus strobus'] * 100,
                'stateProvince': ['California', 'Texas', 'Florida'] * 100,
                'decimalLatitude': [34.0] * 300,
                'decimalLongitude': [-118.0] * 300
            })
        
        # Take a sample if dataset is too large
        if len(us_data) > 5000:
            us_data = us_data.sample(n=5000, random_state=42)
        
        # Create tree data structure
        tree_data = {
            'city': us_data['stateProvince'].fillna('Unknown') if 'stateProvince' in us_data.columns else ['Unknown'] * len(us_data),
            'species': us_data['scientificName'].fillna('Unknown Species') if 'scientificName' in us_data.columns else ['Unknown Species'] * len(us_data),
            'health': np.random.choice(['Good', 'Fair', 'Poor'], len(us_data), p=[0.6, 0.3, 0.1]),
            'latitude': us_data['decimalLatitude'].fillna(39.8283) if 'decimalLatitude' in us_data.columns else [39.8283] * len(us_data),
            'longitude': us_data['decimalLongitude'].fillna(-98.5795) if 'decimalLongitude' in us_data.columns else [-98.5795] * len(us_data),
            'dbh': np.random.uniform(10, 100, len(us_data))
        }
        
        return pd.DataFrame(tree_data)

    def _transform_families_to_canopy(self, families_df: pd.DataFrame) -> pd.DataFrame:
        """Transform families data to canopy coverage format"""
        # Create synthetic canopy data based on families
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
        canopy_data = []
        
        for state in states:
            for i in range(10):  # 10 counties per state
                canopy_data.append({
                    'state': state,
                    'county': f'{state}_County_{i+1}',
                    'canopy_pct': np.random.uniform(15, 75)
                })
        
        return pd.DataFrame(canopy_data)

    @cache.cached(timeout=600)
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

    @cache.cached(timeout=600)
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
        """Create the main dashboard layout with improved error handling and responsiveness"""
        try:
            return html.Div([
                # Header with improved navigation
                dbc.Navbar(
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.A([
                                    html.I(className="fas fa-tree me-2"),
                                    html.Span("🌳 AroundTheTrees", className="navbar-brand")
                                ], href="#", style={"textDecoration": "none", "color": "white"})
                            ], width="auto"),
                            dbc.Col([
                                dbc.Nav([
                                    dbc.NavItem(dbc.NavLink("Dashboard", href="#dashboard", className="nav-link")),
                                    dbc.NavItem(dbc.NavLink("About", href="#about", className="nav-link")),
                                    dbc.NavItem(dbc.NavLink("Support", href="#support", className="nav-link")),
                                ], navbar=True, className="ms-auto")
                            ])
                        ], className="w-100 d-flex justify-content-between align-items-center")
                    ], fluid=True),
                    color="success",
                    dark=True,
                    className="mb-4"
                ),
                
                # Main content with better error boundaries
                dbc.Container([
                    # Status indicator
                    dbc.Alert(
                        id="status-alert",
                        children=self._get_status_message(),
                        color="info" if self._data_loaded else "warning",
                        dismissable=True,
                        className="mb-3"
                    ),
                    
                    # Controls section with improved layout
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("🎯 Dashboard Controls", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select State:", className="form-label fw-bold"),
                                    dcc.Dropdown(
                                        id='state-dropdown',
                                        options=self.state_options,
                                        placeholder="Choose a state...",
                                        className="mb-3",
                                        clearable=True
                                    ),
                                ], md=6),
                                dbc.Col([
                                    html.Label("Select City/Region:", className="form-label fw-bold"),
                                    dcc.Dropdown(
                                        id='city-dropdown',
                                        options=self.city_options,
                                        placeholder="Choose a city...",
                                        className="mb-3",
                                        clearable=True
                                    ),
                                ], md=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Canopy Coverage Range:", className="form-label fw-bold"),
                                    dcc.RangeSlider(
                                        id='canopy-slider',
                                        min=0,
                                        max=100,
                                        step=5,
                                        value=[0, 100],
                                        marks={
                                            0: {'label': '0%', 'style': {'color': '#666'}},
                                            25: {'label': '25%', 'style': {'color': '#666'}},
                                            50: {'label': '50%', 'style': {'color': '#666'}},
                                            75: {'label': '75%', 'style': {'color': '#666'}},
                                            100: {'label': '100%', 'style': {'color': '#666'}}
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-1"
                                    ),
                                    html.Div(id="canopy-range-label", className="text-muted small mb-3")
                                ], md=12)
                            ])
                        ])
                    ], className="mb-4"),

                    # Summary Cards with loading states
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-summary",
                                type="border",
                                color="#28a745",
                                children=html.Div(id='summary-cards')
                            )
                        ], md=12)
                    ], className="mb-4"),

                    # Charts section with improved loading
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("📊 Data Visualizations", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-charts",
                                type="cube",
                                color="#28a745",
                                children=html.Div(id='charts-container')
                            )
                        ])
                    ], className="mb-4"),

                    # Map section with improved loading
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("🗺️ Interactive Map", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-map",
                                type="graph",
                                color="#28a745",
                                children=html.Div(id='map-container')
                            )
                        ])
                    ], className="mb-4"),

                    # Network visualization section (conditionally added)
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4(html.Span([
                                html.I(className="fas fa-project-diagram me-2"), 
                                "🔍 Network Visualization"
                            ]), className="mb-0", id="networks")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-network",
                                type="default",
                                color="#28a745",
                                children=html.Div(id="network-container")
                            )
                        ])
                    ], className="mb-4") if HAS_VOSVIEWER else None,
                ], fluid=True),

                # Enhanced Footer
                dbc.Container([
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("🌱 Support Environmental Organizations", className="text-center mb-3"),
                                html.P("Help make a difference by supporting tree planting and conservation efforts:", 
                                      className="text-center text-muted mb-3"),
                                html.Div([
                                    dbc.Button([
                                        html.I(className="fas fa-seedling me-2"),
                                        "TreePeople"
                                    ], href="https://treepeople.org/?campaign=430396", 
                                       target="_blank", color="success", size="sm", className="me-2 mb-2"),
                                    dbc.Button([
                                        html.I(className="fas fa-tree me-2"),
                                        "Arbor Day Foundation"
                                    ], href="https://donate.arborday.org/", 
                                       target="_blank", color="success", size="sm", className="me-2 mb-2"),
                                    dbc.Button([
                                        html.I(className="fas fa-leaf me-2"),
                                        "Trees for the Future"
                                    ], href="https://trees.org/?form=FUNRTALHRNR", 
                                       target="_blank", color="success", size="sm", className="mb-2")
                                ], className="text-center")
                            ], className="mb-4"),
                            
                            html.Div([
                                html.P([
                                    "Data sources: ", 
                                    html.Strong("GBIF (Global Biodiversity Information Facility)"),
                                    ", Plant Families Database, Conservation Status Data"
                                ], className="text-muted text-center mb-2"),
                                html.P([
                                    "Built with ❤️ using ", 
                                    html.A("Dash", href="https://plotly.com/dash/", target="_blank", className="text-success"),
                                    " and ",
                                    html.A("Plotly", href="https://plotly.com/", target="_blank", className="text-success")
                                ], className="text-muted text-center")
                            ])
                        ])
                    ])
                ], fluid=True, className="py-4 bg-light")
            ])
        except Exception as e:
            logger.error(f"Error creating layout: {e}")
            return self._create_error_layout()

    def _get_status_message(self) -> str:
        """Get current dashboard status message"""
        if self._data_loaded:
            tree_count = len(self.df_trees) if self.df_trees is not None else 0
            canopy_count = len(self.df_canopy) if self.df_canopy is not None else 0
            return f"✅ Dashboard loaded successfully! Displaying {tree_count:,} tree records and {canopy_count} canopy measurements."
        else:
            return "⚠️ Dashboard is loading data. Some features may be limited."

    def _create_error_layout(self) -> html.Div:
        """Create an error layout when main layout fails"""
        return html.Div([
            dbc.Container([
                dbc.Alert([
                    html.H4("⚠️ Dashboard Error", className="alert-heading"),
                    html.P("We're experiencing technical difficulties. Please refresh the page or contact support if the problem persists."),
                    html.Hr(),
                    html.P("Error occurred while loading the dashboard layout.", className="mb-0")
                ], color="danger", className="mt-5")
            ])
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

    def create_charts(self, selected_state: Optional[str], selected_city: Optional[str]) -> List[dbc.Col]:
        """Create charts based on selected filters"""
        if selected_city:
            city_data = self.df_trees[self.df_trees['city'] == selected_city]
            if not city_data.empty:
                return [
                    dbc.Col([
                        html.H4("Species Distribution", className="text-center mb-3"),
                        self.create_species_chart(city_data)
                    ], md=6),
                    dbc.Col([
                        html.H4("Health Distribution", className="text-center mb-3"),
                        self.create_health_chart(city_data)
                    ], md=6)
                ]
        elif selected_state:
            state_data = self.df_canopy[self.df_canopy['state'] == selected_state]
            if not state_data.empty:
                return [
                    dbc.Col([
                        html.H4(f"Canopy Coverage in {selected_state}", className="text-center mb-3"),
                        self.create_canopy_chart(state_data)
                    ], md=12)
                ]
        
        # Default overview charts
        return [
            dbc.Col([
                html.H4("National Overview", className="text-center mb-3"),
                self.create_overview_chart()
            ], md=12)
        ]

    def create_species_chart(self, city_data: pd.DataFrame) -> dcc.Graph:
        """Create species distribution chart with error handling"""
        try:
            if city_data.empty:
                return self._create_empty_chart("No species data available")
                
            species_counts = city_data['species'].value_counts().head(10)
            if species_counts.empty:
                return self._create_empty_chart("No species found")
                
            fig = px.bar(
                x=species_counts.values,
                y=species_counts.index,
                orientation='h',
                title='Top 10 Species Distribution',
                labels={'x': 'Count', 'y': 'Species'},
                color_discrete_sequence=['#28a745']
            )
            fig.update_layout(
                height=400,
                font_family="Arial, sans-serif",
                title_x=0.5
            )
            return dcc.Graph(figure=fig)
        except Exception as e:
            logger.error(f"Error creating species chart: {e}")
            return self._create_error_chart("Error loading species data")

    def create_health_chart(self, city_data: pd.DataFrame) -> dcc.Graph:
        """Create health distribution chart with error handling"""
        try:
            if city_data.empty:
                return self._create_empty_chart("No health data available")
                
            health_counts = city_data['health'].value_counts()
            if health_counts.empty:
                return self._create_empty_chart("No health data found")
                
            fig = px.pie(
                values=health_counts.values,
                names=health_counts.index,
                title='Tree Health Distribution',
                color_discrete_map={
                    'Good': '#28a745',
                    'Fair': '#ffc107', 
                    'Poor': '#dc3545'
                }
            )
            fig.update_layout(
                height=400,
                font_family="Arial, sans-serif",
                title_x=0.5
            )
            return dcc.Graph(figure=fig)
        except Exception as e:
            logger.error(f"Error creating health chart: {e}")
            return self._create_error_chart("Error loading health data")

    def create_canopy_chart(self, state_data: pd.DataFrame) -> dcc.Graph:
        """Create canopy coverage chart for a state with error handling"""
        try:
            if state_data.empty:
                return self._create_empty_chart("No canopy data available")
                
            fig = px.histogram(
                state_data,
                x='canopy_pct',
                nbins=20,
                title='Canopy Coverage Distribution',
                labels={'canopy_pct': 'Canopy Coverage (%)', 'count': 'Number of Counties'},
                color_discrete_sequence=['#28a745']
            )
            fig.update_layout(
                height=400,
                font_family="Arial, sans-serif",
                title_x=0.5
            )
            return dcc.Graph(figure=fig)
        except Exception as e:
            logger.error(f"Error creating canopy chart: {e}")
            return self._create_error_chart("Error loading canopy data")

    def create_overview_chart(self) -> dcc.Graph:
        """Create national overview chart with error handling"""
        try:
            # Create overview from actual data if available
            if self._data_loaded and self.df_trees is not None and not self.df_trees.empty:
                state_counts = self.df_trees.groupby('city').size().head(10)
                fig = px.bar(
                    x=state_counts.values,
                    y=state_counts.index,
                    orientation='h',
                    title='Tree Count by Region (Top 10)',
                    labels={'x': 'Number of Trees', 'y': 'Region'},
                    color_discrete_sequence=['#28a745']
                )
            else:
                # Fallback sample data
                overview_data = pd.DataFrame({
                    'State': ['CA', 'TX', 'FL', 'NY', 'PA'],
                    'Trees': [150000, 120000, 100000, 95000, 85000]
                })
                fig = px.bar(
                    overview_data,
                    x='State',
                    y='Trees',
                    title='Tree Count by State (Sample Data)',
                    labels={'Trees': 'Number of Trees'},
                    color_discrete_sequence=['#28a745']
                )
            
            fig.update_layout(
                height=400,
                font_family="Arial, sans-serif",
                title_x=0.5
            )
            return dcc.Graph(figure=fig)
        except Exception as e:
            logger.error(f"Error creating overview chart: {e}")
            return self._create_error_chart("Error loading overview data")

    def _create_empty_chart(self, message: str) -> dcc.Graph:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{"text": message, "showarrow": False}]
        )
        return dcc.Graph(figure=fig)

    def _create_error_chart(self, message: str) -> dcc.Graph:
        """Create an error chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            height=400,
            xaxis={"visible": False},
            yaxis={"visible": False}
        )
        return dcc.Graph(figure=fig)

    @cache.memoize(timeout=120)
    def create_map(
        self, selected_state: Optional[str], selected_city: Optional[str],
        canopy_range: List[float]
    ) -> dcc.Graph:
        """Create the interactive map with comprehensive error handling"""
        try:
            mapbox_token = os.getenv('MAPBOX_TOKEN')
            mapbox_style = "open-street-map"  # Default fallback style
            
            if selected_city and self.df_trees is not None:
                # City-level tree point map
                city_data = self.df_trees[self.df_trees['city'] == selected_city]
                if not city_data.empty and 'latitude' in city_data.columns and 'longitude' in city_data.columns:
                    # Filter out invalid coordinates
                    city_data = city_data[
                        (city_data['latitude'].notna()) & 
                        (city_data['longitude'].notna()) &
                        (city_data['latitude'].between(-90, 90)) &
                        (city_data['longitude'].between(-180, 180))
                    ]
                    
                    if not city_data.empty:
                        render_mode = "webgl" if len(city_data) > 2000 else "auto"
                        fig = px.scatter_mapbox(
                            city_data,
                            lat='latitude',
                            lon='longitude',
                            color='health',
                            hover_data={'species': True, 'dbh': True, 'latitude': False, 'longitude': False},
                            zoom=8,
                            height=600,
                            title=f'Tree Locations in {selected_city}',
                            color_discrete_map={
                                'Good': '#28a745',
                                'Fair': '#ffc107',
                                'Poor': '#dc3545'
                            },
                            render_mode=render_mode
                        )
                        # Center the map on the data
                        fig.update_layout(
                            mapbox_center_lat=city_data['latitude'].mean(),
                            mapbox_center_lon=city_data['longitude'].mean()
                        )
                    else:
                        return self._create_empty_map("No valid coordinates found for this city")
                else:
                    return self._create_empty_map(f"No data available for {selected_city}")
                    
            elif selected_state and self.df_canopy is not None:
                # State-level canopy visualization
                state_data = self.df_canopy[
                    (self.df_canopy['state'] == selected_state) &
                    (self.df_canopy['canopy_pct'] >= canopy_range[0]) &
                    (self.df_canopy['canopy_pct'] <= canopy_range[1])
                ]
                
                if not state_data.empty:
                    # Create a simple scatter plot for counties (since we don't have actual geometries)
                    # Generate random coordinates within US bounds for visualization
                    np.random.seed(42)  # For reproducible results
                    state_coords = self._get_state_coordinates(selected_state)
                    
                    lats = np.random.uniform(
                        state_coords['lat'] - 2, 
                        state_coords['lat'] + 2, 
                        len(state_data)
                    )
                    lons = np.random.uniform(
                        state_coords['lon'] - 3, 
                        state_coords['lon'] + 3, 
                        len(state_data)
                    )
                    
                    render_mode = "webgl" if len(state_data) > 1000 else "auto"
                    fig = px.scatter_mapbox(
                        state_data,
                        lat=lats,
                        lon=lons,
                        color='canopy_pct',
                        hover_name='county',
                        color_continuous_scale="Greens",
                        range_color=[0, 100],
                        zoom=6,
                        height=600,
                        title=f'Canopy Coverage in {selected_state}',
                        labels={'color': 'Canopy %'},
                        render_mode=render_mode
                    )
                    fig.update_layout(
                        mapbox_center_lat=state_coords['lat'],
                        mapbox_center_lon=state_coords['lon']
                    )
                else:
                    return self._create_empty_map(f"No data available for {selected_state} in the selected range")
            else:
                # Default US overview map
                fig = self._create_default_us_map()

            # Configure mapbox
            fig.update_layout(
                mapbox_style=mapbox_style,
                mapbox_accesstoken=mapbox_token,
                font_family="Arial, sans-serif",
                title_x=0.5,
                margin={"r":0,"t":40,"l":0,"b":0}
            )
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            logger.error(f"Error creating map: {e}")
            return self._create_empty_map("Error loading map data")

    def _get_state_coordinates(self, state: str) -> Dict[str, float]:
        """Get approximate center coordinates for US states"""
        state_coords = {
            'CA': {'lat': 36.7783, 'lon': -119.4179},
            'TX': {'lat': 31.9686, 'lon': -99.9018},
            'FL': {'lat': 27.7663, 'lon': -81.6868},
            'NY': {'lat': 42.1657, 'lon': -74.9481},
            'PA': {'lat': 41.2033, 'lon': -77.1945},
            'IL': {'lat': 40.3363, 'lon': -89.0022},
            'OH': {'lat': 40.3888, 'lon': -82.7649},
            'GA': {'lat': 33.76, 'lon': -84.39},
            'NC': {'lat': 35.771, 'lon': -78.638},
            'MI': {'lat': 44.182, 'lon': -84.506}
        }
        return state_coords.get(state, {'lat': 39.8283, 'lon': -98.5795})  # Default to US center

    def _create_default_us_map(self) -> go.Figure:
        """Create a default US map with sample data"""
        # Sample major US cities
        cities_data = pd.DataFrame({
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'lat': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
            'lon': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
            'trees': [50000, 45000, 40000, 35000, 30000]
        })
        
        fig = px.scatter_mapbox(
            cities_data,
            lat='lat',
            lon='lon',
            size='trees',
            hover_name='city',
            hover_data={'trees': True, 'lat': False, 'lon': False},
            zoom=3,
            height=600,
            title='Major US Cities Tree Distribution (Sample)',
            color_discrete_sequence=['#28a745']
        )
        
        fig.update_layout(
            mapbox_center_lat=39.8283,
            mapbox_center_lon=-98.5795
        )
        
        return fig

    def _create_empty_map(self, message: str) -> dcc.Graph:
        """Create an empty map with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=600,
            xaxis={"visible": False},
            yaxis={"visible": False}
        )
        return dcc.Graph(figure=fig)

# Initialize dashboard
dashboard = TreeDashboard()

# Main app layout
app.layout = dashboard.create_layout()

# Register VOSviewer callbacks if available
if HAS_VOSVIEWER:
    try:
        from src.components import register_vosviewer_callbacks
        register_vosviewer_callbacks(app)
        logger.info("✅ VOSviewer callbacks registered successfully")
    except Exception as e:
        logger.error(f"❌ Failed to register VOSviewer callbacks: {e}")

@cache.memoize(timeout=90)
def _render_summary_cards(state, city):
    cards = dashboard.create_summary_cards(state, city)
    if cards:
        return dbc.Row([dbc.Col(card, md=4) for card in cards[:3]])
    return dbc.Alert(
        "Select a state or city to view summary statistics.",
        color="info",
        className="text-center"
    )

@app.callback(
    Output("summary-cards", "children"),
    [Input("state-dropdown", "value"),
     Input("city-dropdown", "value")],
    prevent_initial_call=False
)
def update_summary_cards(state, city):
    try:
        return _render_summary_cards(state, city)
    except Exception as e:
        logger.error(f"Error updating summary cards: {e}")
        return dbc.Alert("Error loading summary data. Please try again.", color="danger", className="text-center")

# Clientside callback for canopy range label (avoids server round-trip)
app.clientside_callback(
    "function(range){ if(!range){return ''}; return 'Range: '+ range[0] + '% - ' + range[1] + '%'; }",
    Output("canopy-range-label", "children"),
    Input("canopy-slider", "value")
)

# Callback to update charts
@app.callback(
    Output("charts-container", "children"),
    [Input("state-dropdown", "value"),
     Input("city-dropdown", "value")],
    prevent_initial_call=False
)
def update_charts(state, city):
    try:
        chart_columns = dashboard.create_charts(state, city)
        if chart_columns:
            return dbc.Row(chart_columns)
        else:
            return dbc.Alert(
                "No data available for the selected filters.",
                color="warning",
                className="text-center"
            )
    except Exception as e:
        logger.error(f"Error updating charts: {e}")
        return dbc.Alert(
            "Error loading chart data. Please try again.",
            color="danger",
            className="text-center"
        )

# Callback to update map
@app.callback(
    Output("map-container", "children"),
    [Input("state-dropdown", "value"),
     Input("city-dropdown", "value"),
     Input("canopy-slider", "value")],
    prevent_initial_call=False
)
def update_map(state, city, canopy_range):
    try:
        if canopy_range is None:
            canopy_range = [0, 100]
        return dashboard.create_map(state, city, canopy_range)
    except Exception as e:
        logger.error(f"Error updating map: {e}")
        return dbc.Alert(
            "Error loading map data. Please try again.",
            color="danger",
            className="text-center"
        )

# Callback to update city options based on selected state
@app.callback(
    Output("city-dropdown", "options"),
    [Input("state-dropdown", "value")],
    prevent_initial_call=False
)
def update_city_options(selected_state):
    try:
        if selected_state and dashboard.df_trees is not None:
            # Filter cities by state if we have mapping data
            # For now, return all cities since we don't have state-city mapping
            return dashboard.city_options
        return dashboard.city_options
    except Exception as e:
        logger.error(f"Error updating city options: {e}")
        return []

# Callback to populate VOSviewer network container
@app.callback(
    Output("network-container", "children"),
    [Input("tabs", "active_tab")],
    prevent_initial_call=False
)
def render_network_visualization(active_tab):
    if not HAS_VOSVIEWER:
        return html.Div("VOSviewer components not available", className="text-center p-5")
    
    if active_tab == "networks-tab":
        try:
            # Create the network visualization component
            from src.components import create_vosviewer_network
            return create_vosviewer_network()
        except Exception as e:
            logger.error(f"Error rendering VOSviewer: {e}")
            return html.Div(f"Error loading network visualization: {str(e)}", className="text-danger p-5")
    return html.Div()

# ---------------------------------------------------------------------------
# Observability & Health Endpoints (Prometheus metrics + basic health)
# ---------------------------------------------------------------------------
try:  # Optional metrics dependencies
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST  # type: ignore
    import time
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _METRICS_AVAILABLE = False

if _METRICS_AVAILABLE:
    REQUEST_COUNTER = Counter('dashboard_requests_total', 'Total dashboard HTTP requests', ['endpoint', 'method'])
    REQUEST_LATENCY = Histogram('dashboard_request_latency_seconds', 'Latency for dashboard HTTP requests', ['endpoint'])
    ACTIVE_USERS_GAUGE = Gauge('dashboard_active_users', 'Active user sessions (approximate)')
    LAST_HEALTH_CHECK = Gauge('dashboard_last_health_check_timestamp', 'Last health check UNIX timestamp')

    @server.before_request
    def _before_request_metrics():  # pragma: no cover
        from flask import g  # type: ignore
        g._start_time = time.time()

    @server.after_request
    def _after_request_metrics(response):  # pragma: no cover
        from flask import request, g  # type: ignore
        try:
            elapsed = time.time() - getattr(g, '_start_time', time.time())
            REQUEST_COUNTER.labels(endpoint=request.path, method=request.method).inc()
            REQUEST_LATENCY.labels(endpoint=request.path).observe(elapsed)
        except Exception as metric_err:  # noqa: BLE001
            logger.debug(f"Metric collection error: {metric_err}")
        return response
else:
    logger.debug("Prometheus metrics not available; skipping instrumentation")

@server.route('/healthz')
def health_root():  # pragma: no cover
    """Lightweight container/platform liveness probe."""
    LAST_HEALTH_CHECK.set_to_current_time()
    return {"status": "ok", "version": "2.1.0"}, 200

@server.route('/api/v1/health')
def health_api():  # pragma: no cover
    """Semantic API health endpoint used by orchestrators & monitors."""
    LAST_HEALTH_CHECK.set_to_current_time()
    issues: list[str] = []
    try:
        if dashboard.df_trees is None or dashboard.df_trees.empty:
            issues.append("trees_dataset_empty")
        if dashboard.df_canopy is None or dashboard.df_canopy.empty:
            issues.append("canopy_dataset_empty")
    except Exception as e:  # noqa: BLE001
        issues.append(f"exception:{e.__class__.__name__}")
    status = "healthy" if not issues else "degraded"
    return {"status": status, "issues": issues, "records": {
        "trees": 0 if dashboard.df_trees is None else len(dashboard.df_trees),
        "canopy": 0 if dashboard.df_canopy is None else len(dashboard.df_canopy)
    }}, 200 if status == "healthy" else 206

@server.route('/metrics')
def metrics():  # pragma: no cover
    """Prometheus metrics scrape endpoint."""
    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    try:
        # Get configuration from environment
        debug_mode = os.getenv('DEBUG', 'True').lower() == 'true'
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 8050))
        
        logger.info(f"Starting dashboard server on {host}:{port} (debug={debug_mode})")
        
        # Start the app
        app.run(
            debug=debug_mode, 
            host=host, 
            port=port,
            dev_tools_hot_reload=debug_mode,
            dev_tools_props_check=debug_mode
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Error starting application: {e}")
        print("Please check your environment configuration and try again.")
