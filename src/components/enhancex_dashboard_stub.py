"""Minimal stub for EnhanceX dashboard component.

This stub provides the basic interface to avoid import errors during testing
while the full implementation with heavy dependencies is developed separately.
"""

from dash import html


def create_enhancex_dashboard():  # pragma: no cover - trivial stub
    """Create a minimal EnhanceX dashboard stub."""
    return html.Div(
        [
            html.H5("EnhanceX Dashboard (stub)"),
            html.P("EnhanceX dashboard UI temporarily disabled."),
        ],
        id="enhancex-dashboard-stub",
    )


def register_callbacks(app):  # pragma: no cover - nothing to register
    """Register callbacks for EnhanceX dashboard."""
    return None