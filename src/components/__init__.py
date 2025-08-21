"""
Dashboard Components Package

This package contains reusable components for the conservation dashboard.
"""

# Import key components for easy access
from .vosviewer_network import create_vosviewer_network, register_callbacks as register_vosviewer_callbacks
from .theme_switcher import create_theme_switcher, register_callbacks as register_theme_callbacks
from .visualization_preferences import create_visualization_preferences, register_callbacks as register_viz_callbacks
from .notification_system import create_notification_system, register_callbacks as register_notification_callbacks
from .project_context_manager import create_project_context_manager, register_callbacks as register_context_callbacks
