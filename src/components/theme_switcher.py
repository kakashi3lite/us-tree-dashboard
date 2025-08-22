"""Minimal theme switcher stub for tests.

Original implementation removed due to corruption; this keeps the public
API (create_theme_switcher, register_callbacks) so imports succeed.
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


def create_theme_switcher():  # pragma: no cover - trivial
    """Create theme switcher stub."""
    if DASHBOARD_AVAILABLE:
        return html.Div(id="theme-switcher", children=[html.Span("Theme: Light")])
    else:
        # Return mock component for testing
        return {
            "type": "div", 
            "props": {
                "id": "theme-switcher",
                "children": [{"type": "span", "props": {"children": "Theme: Light"}}]
            }
        }


def register_callbacks(app):  # pragma: no cover - no callbacks
    """Register callbacks stub."""
    return None