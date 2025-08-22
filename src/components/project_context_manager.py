"""Minimal stub for project context manager component.

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


def create_project_context_manager():  # pragma: no cover - trivial stub
    """Create a minimal project context manager stub."""
    if DASHBOARD_AVAILABLE:
        return html.Div(
            [
                html.H5("Project Context Manager (stub)"),
                html.P("Project context manager temporarily disabled."),
            ],
            id="project-context-manager-stub",
        )
    else:
        # Return mock component for testing
        return {
            "type": "div",
            "props": {
                "id": "project-context-manager-stub",
                "children": [
                    {"type": "h5", "props": {"children": "Project Context Manager (stub)"}},
                    {"type": "p", "props": {"children": "Project context manager temporarily disabled."}}
                ]
            }
        }


def register_callbacks(app):  # pragma: no cover - nothing to register
    """Register callbacks for project context manager."""
    return None


def render_context_item(key, value, index):  # pragma: no cover - stub
    """Render context item stub."""
    return {"key": key, "value": value, "index": index}