"""Minimal stub for EnhanceX dashboard component.

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


def create_enhancex_dashboard():  # pragma: no cover - trivial stub
    """Create a minimal EnhanceX dashboard stub."""
    if DASHBOARD_AVAILABLE:
        return html.Div(
            [
                html.H5("EnhanceX Dashboard (stub)"),
                html.P("EnhanceX dashboard UI temporarily disabled."),
            ],
            id="enhancex-dashboard-stub",
        )
    else:
        # Return mock component for testing
        return {
            "type": "div",
            "props": {
                "id": "enhancex-dashboard-stub",
                "children": [
                    {"type": "h5", "props": {"children": "EnhanceX Dashboard (stub)"}},
                    {"type": "p", "props": {"children": "EnhanceX dashboard UI temporarily disabled."}}
                ]
            }
        }


def register_callbacks(app):  # pragma: no cover - nothing to register
    """Register callbacks for EnhanceX dashboard."""
    return None