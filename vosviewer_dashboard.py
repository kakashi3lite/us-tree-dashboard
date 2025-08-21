"""
VOSviewer Dashboard Integration Page

Provides a standalone page for VOSviewer network visualization
integrated with the conservation dashboard.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

from src.components import create_vosviewer_network, register_vosviewer_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME],
    title="VOSviewer Conservation Network"
)

# For Gunicorn deployment
server = app.server

# Navigation bar with links
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.I(className="fas fa-tree me-2"),
                dbc.NavbarBrand("VOSviewer Conservation Network", className="ms-2")
            ]),
        ], align="center"),
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Species Network", href="#species")),
                    dbc.NavItem(dbc.NavLink("Hotspots Network", href="#hotspots")),
                    dbc.NavItem(dbc.NavLink("Conservation Status", href="#status")),
                    dbc.NavItem(dbc.NavLink("Back to Dashboard", href="/")),
                ], navbar=True)
            ])
        ])
    ]),
    color="primary",
    dark=True,
    className="mb-4"
)

# Create app layout
app.layout = html.Div([
    # URL management
    dcc.Location(id="url", refresh=False),
    
    # Navigation bar
    navbar,
    
    # Main container
    dbc.Container([
        # Header section
        dbc.Row([
            dbc.Col([
                html.H1("Biodiversity Network Visualization", className="display-4 mb-3"),
                html.P(
                    "Explore relationships between species, biodiversity hotspots, and conservation statuses "
                    "using interactive network visualizations powered by VOSviewer.",
                    className="lead mb-4"
                ),
                html.Hr()
            ])
        ]),
        
        # VOSviewer explanation
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("About VOSviewer Network Visualization", className="card-title"),
                        html.P(
                            "VOSviewer is a software tool for creating and visualizing bibliometric networks. "
                            "In this dashboard, we've adapted it to visualize biodiversity and conservation networks, "
                            "helping you discover relationships and patterns in our ecological data.",
                            className="card-text"
                        ),
                        html.P([
                            "The visualization supports three types of networks:",
                            html.Ul([
                                html.Li("Species Relationships: Shows connections between species based on conservation status and threats"),
                                html.Li("Biodiversity Hotspots: Visualizes connections between global biodiversity hotspots"),
                                html.Li("Conservation Status: Displays relationships between different conservation status categories")
                            ])
                        ]),
                        dbc.Button(
                            "Learn More about VOSviewer",
                            href="https://www.vosviewer.com/",
                            target="_blank",
                            color="info",
                            className="mt-2"
                        )
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # VOSviewer Network Visualization
        dbc.Row([
            dbc.Col([
                # Main visualization component
                create_vosviewer_network()
            ])
        ]),
        
        # Instructions section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Usage Instructions"),
                    dbc.CardBody([
                        html.P([
                            "To use the network visualization:",
                            html.Ol([
                                html.Li("Select a network type (Species, Biodiversity Hotspots, or Conservation Status)"),
                                html.Li("Choose visualization settings like visualization type and color scheme"),
                                html.Li("Adjust node size and distance parameters"),
                                html.Li("Click 'Apply Settings' to generate the visualization"),
                                html.Li("Use mouse to interact with the network: zoom, pan, and click nodes for details"),
                                html.Li("Optionally download the network data for use in the standalone VOSviewer application")
                            ])
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P(
                    "Conservation Dashboard VOSviewer Integration | "
                    "Data sources: GBIF, IUCN Red List, Biodiversity Hotspots Database",
                    className="text-muted text-center"
                )
            ])
        ])
    ], className="mb-5"),
])

# Register VOSviewer callbacks
register_vosviewer_callbacks(app)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
