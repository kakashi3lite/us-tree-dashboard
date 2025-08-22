"""Minimal stub for VOSviewer network component.

This stub provides the basic interface to avoid import errors during testing
while the full implementation with heavy dependencies is developed separately.
"""

import os

# Feature flag for dashboard components  
ENABLE_DASHBOARD_FEATURES = os.getenv('DASHBOARD_FULL_FEATURES', 'false').lower() == 'true'

if ENABLE_DASHBOARD_FEATURES:
    try:
        from dash import html
        DASHBOARD_AVAILABLE = True
    except ImportError:
        DASHBOARD_AVAILABLE = False
else:
    DASHBOARD_AVAILABLE = False


def create_vosviewer_network():  # pragma: no cover - trivial stub
    """Create a minimal VOSviewer network stub."""
    if DASHBOARD_AVAILABLE:
        return html.Div(
            [
                html.H5("VOSviewer Network (stub)"),
                html.P("VOSviewer network visualization temporarily disabled."),
            ],
            id="vosviewer-network-stub",
        )
    else:
        # Return mock component for testing
        return {
            "type": "div",
            "props": {
                "id": "vosviewer-network-stub",
                "children": [
                    {"type": "h5", "props": {"children": "VOSviewer Network (stub)"}},
                    {"type": "p", "props": {"children": "VOSviewer network visualization temporarily disabled."}}
                ]
            }
        }


def register_callbacks(app):  # pragma: no cover - nothing to register
    """Register callbacks for VOSviewer network."""
    return None