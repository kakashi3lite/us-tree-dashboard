"""
VOSviewer Integration Module

This module provides functions to integrate VOSviewer network visualization
into any Dash application, particularly for conservation dashboards.
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from .components import create_vosviewer_network, register_vosviewer_callbacks

def integrate_vosviewer_into_dashboard(app, layout_component, navbar_item=True):
    """
    Integrate VOSviewer network visualization into an existing dashboard.
    
    Args:
        app: Dash application instance
        layout_component: Component to add VOSviewer section to
        navbar_item: Whether to add a navbar item for VOSviewer
        
    Returns:
        Updated layout component with VOSviewer integration
    """
    # Add VOSviewer network callbacks
    register_vosviewer_callbacks(app)
    
    # Create VOSviewer section
    vosviewer_section = html.Div([
        html.H2("Network Visualization", id="network-visualization", className="mb-4"),
        html.P(
            "Explore relationships between species, biodiversity hotspots, and conservation statuses "
            "using interactive network visualizations powered by VOSviewer.",
            className="lead mb-4"
        ),
        create_vosviewer_network()
    ], className="mt-5 mb-5")
    
    # Add VOSviewer section to layout
    if isinstance(layout_component, list):
        layout_component.append(vosviewer_section)
    elif hasattr(layout_component, 'children') and isinstance(layout_component.children, list):
        layout_component.children.append(vosviewer_section)
    
    return layout_component

def create_vosviewer_navbar_item():
    """
    Create a navbar item for VOSviewer network visualization.
    
    Returns:
        Dash component for navbar item
    """
    return dbc.NavItem(dbc.NavLink("Network Visualization", href="#network-visualization"))

def create_vosviewer_modal(id_prefix="vosviewer"):
    """
    Create a modal dialog for VOSviewer network visualization.
    
    Args:
        id_prefix: Prefix for component IDs
        
    Returns:
        Modal component with VOSviewer
    """
    return dbc.Modal([
        dbc.ModalHeader("Network Visualization"),
        dbc.ModalBody([
            create_vosviewer_network()
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Close",
                id=f"{id_prefix}-modal-close",
                className="ms-auto"
            )
        )
    ], id=f"{id_prefix}-modal", size="xl")

def create_vosviewer_tab():
    """
    Create a tab for VOSviewer network visualization.
    
    Returns:
        Tab component with VOSviewer
    """
    return dbc.Tab(
        label="Network Visualization",
        tab_id="network-tab",
        children=html.Div(
            className="p-3",
            children=[
                create_vosviewer_network()
            ]
        )
    )

def register_vosviewer_modal_callbacks(app, button_id, modal_id="vosviewer-modal", close_id="vosviewer-modal-close"):
    """
    Register callbacks for VOSviewer modal dialog.
    
    Args:
        app: Dash application instance
        button_id: ID of button to trigger modal
        modal_id: ID of modal component
        close_id: ID of close button
    """
    @app.callback(
        dash.dependencies.Output(modal_id, "is_open"),
        [dash.dependencies.Input(button_id, "n_clicks"),
         dash.dependencies.Input(close_id, "n_clicks")],
        [dash.dependencies.State(modal_id, "is_open")]
    )
    def toggle_modal(n_open, n_close, is_open):
        if n_open or n_close:
            return not is_open
        return is_open
