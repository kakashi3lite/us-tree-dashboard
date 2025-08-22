"""Minimal notification system stub for tests.

Original implementation requires heavy dependencies; this keeps the public
API so imports succeed during coverage parsing.
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


def create_notification_system():  # pragma: no cover - trivial stub
    """Create a minimal notification system stub."""
    if DASHBOARD_AVAILABLE:
        return html.Div(
            [
                html.H5("Notification System (stub)"),
                html.P("Notification system temporarily disabled."),
            ],
            id="notification-system-stub",
        )
    else:
        # Return mock component for testing
        return {
            "type": "div",
            "props": {
                "id": "notification-system-stub",
                "children": [
                    {"type": "h5", "props": {"children": "Notification System (stub)"}},
                    {"type": "p", "props": {"children": "Notification system temporarily disabled."}}
                ]
            }
        }


def register_callbacks(app):  # pragma: no cover - nothing to register
    """Register callbacks for notification system."""
    return None


def add_notification(notification_type, message, title=None, duration=5000):  # pragma: no cover
    """Add notification stub."""
    return {"type": notification_type, "message": message, "title": title, "duration": duration}