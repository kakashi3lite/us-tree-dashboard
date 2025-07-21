# EnhanceX Main Integration Module

import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Import EnhanceX components
from src.memory import EnhanceX
from src.components.enhancex_dashboard import create_enhancex_dashboard, register_enhancex_callbacks
from src.components.theme_switcher import create_theme_switcher, register_theme_callbacks
from src.components.notification_system import create_notification_system, register_notification_callbacks, add_notification
from src.components.project_context_manager import create_project_context_manager, register_context_callbacks
from src.components.visualization_preferences import create_visualization_preferences, register_visualization_callbacks

# Initialize EnhanceX instance
enhancex = None

def initialize_enhancex(app):
    """Initialize the EnhanceX memory management system."""
    global enhancex
    
    # Create data directories if they don't exist
    os.makedirs('data/enhancex', exist_ok=True)
    os.makedirs('data/enhancex/memory', exist_ok=True)
    os.makedirs('data/enhancex/contexts', exist_ok=True)
    os.makedirs('data/enhancex/preferences', exist_ok=True)
    os.makedirs('data/enhancex/sessions', exist_ok=True)
    
    # Initialize EnhanceX with configuration
    enhancex = EnhanceX(
        app_name="US Tree Dashboard",
        memory_dir="data/enhancex/memory",
        contexts_dir="data/enhancex/contexts",
        preferences_dir="data/enhancex/preferences",
        sessions_dir="data/enhancex/sessions",
        session_timeout=3600,  # 1 hour session timeout
        cleanup_interval=86400  # Clean up expired sessions daily
    )
    
    # Add EnhanceX instance to app for global access
    app.enhancex = enhancex
    
    # Register interaction handler for page views
    @enhancex.session_memory.register_interaction_handler('page_view')
    def handle_page_view(interaction_data):
        page = interaction_data.get('page', 'unknown')
        enhancex.store_memory(f"User viewed {page} page", memory_type='navigation')
    
    return enhancex

def extend_app_layout(app, original_layout):
    """Extend the app layout with EnhanceX components."""
    # Create EnhanceX components
    enhancex_dashboard = create_enhancex_dashboard()
    theme_switcher = create_theme_switcher()
    notification_system = create_notification_system()
    
    # Create EnhanceX sidebar
    sidebar_content = html.Div([
        html.Div([
            html.H4("EnhanceX Dashboard", className="mb-3"),
            html.Button(
                html.I(className="fas fa-times"),
                id="enhancex-sidebar-close",
                className="enhancex-sidebar-close"
            )
        ], className="enhancex-sidebar-header"),
        
        html.Div([
            # Theme Switcher
            html.Div([
                html.H5("Theme", className="mb-2"),
                theme_switcher
            ], className="mb-4"),
            
            # Visualization Preferences
            html.Div([
                html.H5("Visualization Preferences", className="mb-2"),
                create_visualization_preferences()
            ], className="mb-4"),
            
            # Project Context Manager
            html.Div([
                html.H5("Project Context", className="mb-2"),
                create_project_context_manager()
            ], className="mb-4")
        ], className="enhancex-sidebar-content")
    ], id="enhancex-sidebar", className="enhancex-sidebar")
    
    # Create sidebar toggle button
    sidebar_toggle = html.Button(
        html.I(className="fas fa-cog"),
        id="enhancex-sidebar-toggle",
        className="enhancex-sidebar-toggle",
        title="EnhanceX Settings"
    )
    
    # Combine original layout with EnhanceX components
    extended_layout = html.Div([
        # Add app-container class to the original layout for theming
        html.Div(original_layout, className="app-container"),
        
        # EnhanceX components
        html.Div([
            sidebar_toggle,
            sidebar_content
        ], className="enhancex-sidebar-container"),
        
        # Notification system
        notification_system,
        
        # Store for EnhanceX state
        dcc.Store(id="enhancex-session-store"),
        dcc.Store(id="enhancex-theme-store"),
        dcc.Store(id="enhancex-preferences-store"),
        
        # Interval for session management
        dcc.Interval(
            id='enhancex-session-interval',
            interval=60000,  # 1 minute
            n_intervals=0
        ),
        
        # Interval for cleanup
        dcc.Interval(
            id='enhancex-cleanup-interval',
            interval=3600000,  # 1 hour
            n_intervals=0
        ),
        
        # Load Font Awesome for icons
        html.Link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
        )
    ])
    
    return extended_layout

def register_enhancex_app_callbacks(app):
    """Register EnhanceX callbacks for the app."""
    # Register component-specific callbacks
    register_theme_callbacks(app)
    register_notification_callbacks(app)
    register_context_callbacks(app)
    register_visualization_callbacks(app)
    
    # Register main EnhanceX dashboard callbacks
    register_enhancex_callbacks(app)
    
    # Sidebar toggle callback
    @app.callback(
        Output("enhancex-sidebar", "className"),
        [Input("enhancex-sidebar-toggle", "n_clicks"),
         Input("enhancex-sidebar-close", "n_clicks")],
        [State("enhancex-sidebar", "className")]
    )
    def toggle_sidebar(n_open, n_close, current_class):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_class
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "enhancex-sidebar-toggle":
            if "active" in current_class:
                return "enhancex-sidebar"
            else:
                return "enhancex-sidebar active"
        elif button_id == "enhancex-sidebar-close":
            return "enhancex-sidebar"
        
        return current_class
    
    # Session management callback
    @app.callback(
        Output("enhancex-session-store", "data"),
        [Input("enhancex-session-interval", "n_intervals")]
    )
    def keep_session_alive(n_intervals):
        if enhancex and enhancex.session_memory:
            enhancex.session_memory.keep_alive()
            return {"session_id": enhancex.session_memory.session_id, "timestamp": enhancex.session_memory.last_activity}
        return {}
    
    # Cleanup callback
    @app.callback(
        Output("enhancex-cleanup-interval", "disabled"),
        [Input("enhancex-cleanup-interval", "n_intervals")]
    )
    def run_cleanup(n_intervals):
        if enhancex:
            enhancex.cleanup_expired_sessions()
        return False

def track_interaction(interaction_type):
    """Decorator to track user interactions with EnhanceX."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if enhancex and enhancex.session_memory:
                # Extract relevant data from args/kwargs based on interaction type
                interaction_data = {}
                if args and hasattr(args[0], 'callback_context'):
                    ctx = args[0].callback_context
                    if ctx.triggered:
                        prop_id = ctx.triggered[0]['prop_id'].split('.')
                        interaction_data['component_id'] = prop_id[0] if len(prop_id) > 0 else None
                        interaction_data['property'] = prop_id[1] if len(prop_id) > 1 else None
                
                enhancex.session_memory.record_interaction(interaction_type, interaction_data)
            return result
        return wrapper
    return decorator

def add_notification_function(app):
    """Add the add_notification function to the app for global access."""
    app.add_notification = lambda message, category='info', timeout=4000: add_notification(
        message, category, timeout
    )

def show_welcome_notification(app):
    """Show a welcome notification when the app starts."""
    @app.callback(
        Output("enhancex-notification-container", "children"),
        [Input("enhancex-session-store", "data")],
        [State("enhancex-notification-container", "children")]
    )
    def welcome_notification(session_data, current_notifications):
        if session_data and enhancex and enhancex.session_memory:
            # Only show welcome message for new sessions
            if enhancex.session_memory.session_duration < 5:  # Less than 5 seconds old
                add_notification(
                    "Welcome to the US Tree Dashboard with EnhanceX! Your session preferences and context will be saved automatically.",
                    "success",
                    6000
                )
        return current_notifications or []

def integrate_enhancex(app, original_layout):
    """Main function to integrate EnhanceX with the Dash app."""
    # Initialize EnhanceX
    initialize_enhancex(app)
    
    # Extend app layout
    app.layout = extend_app_layout(app, original_layout)
    
    # Register callbacks
    register_enhancex_app_callbacks(app)
    
    # Add notification function to app
    add_notification_function(app)
    
    # Show welcome notification
    show_welcome_notification(app)
    
    return app