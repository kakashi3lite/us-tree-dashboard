# EnhanceX Demo Script

import dash
from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import time

# Import EnhanceX components
from src.memory import EnhanceX
from src.components.theme_switcher import create_theme_switcher
from src.components.notification_system import create_notification_system, add_notification
from src.components.project_context_manager import create_project_context_manager
from src.components.visualization_preferences import create_visualization_preferences

# Create a simple Dash app for demonstration
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    title="EnhanceX Demo"
)

# Initialize EnhanceX
enhancex = EnhanceX(
    app_name="EnhanceX Demo",
    memory_dir="data/enhancex/memory",
    contexts_dir="data/enhancex/contexts",
    preferences_dir="data/enhancex/preferences",
    sessions_dir="data/enhancex/sessions"
)

# Add EnhanceX to app for global access
app.enhancex = enhancex

# Create EnhanceX components
theme_switcher = create_theme_switcher()
notification_system = create_notification_system()
project_context_manager = create_project_context_manager()
visualization_preferences = create_visualization_preferences()

# Create demo layout
app.layout = html.Div([
    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-memory me-2"),
                    dbc.NavbarBrand("EnhanceX Demo", className="ms-2")
                ]),
            ], align="center"),
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Main content
    dbc.Container([
        dbc.Row([
            # Left column - Demo controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Demo Controls"),
                    dbc.CardBody([
                        html.H5("Theme Switcher", className="mb-3"),
                        theme_switcher,
                        html.Hr(),
                        
                        html.H5("Notifications", className="mb-3 mt-4"),
                        dbc.Button(
                            "Show Info Notification", 
                            id="show-info-notification", 
                            color="info", 
                            className="me-2 mb-2"
                        ),
                        dbc.Button(
                            "Show Success Notification", 
                            id="show-success-notification", 
                            color="success", 
                            className="me-2 mb-2"
                        ),
                        dbc.Button(
                            "Show Warning Notification", 
                            id="show-warning-notification", 
                            color="warning", 
                            className="me-2 mb-2"
                        ),
                        dbc.Button(
                            "Show Error Notification", 
                            id="show-error-notification", 
                            color="danger", 
                            className="mb-2"
                        ),
                        html.Hr(),
                        
                        html.H5("Session Interactions", className="mb-3 mt-4"),
                        dbc.Button(
                            "Record Interaction", 
                            id="record-interaction", 
                            color="primary", 
                            className="me-2 mb-2"
                        ),
                        dbc.Button(
                            "Show Recent Interactions", 
                            id="show-interactions", 
                            color="secondary", 
                            className="mb-2"
                        ),
                        html.Div(id="interactions-output", className="mt-3")
                    ])
                ], className="mb-4")
            ], md=6),
            
            # Right column - EnhanceX components
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Project Context Manager"),
                    dbc.CardBody([
                        project_context_manager
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Visualization Preferences"),
                    dbc.CardBody([
                        visualization_preferences
                    ])
                ]),
            ], md=6),
        ]),
        
        # Session info
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Session Information"),
                    dbc.CardBody([
                        html.Div(id="session-info")
                    ])
                ])
            ], md=12)
        ], className="mt-4 mb-4"),
        
        # Memory store display
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Memory Store"),
                    dbc.CardBody([
                        html.Div(id="memory-store-display")
                    ])
                ])
            ], md=12)
        ], className="mb-4"),
    ], fluid=True, className="app-container"),
    
    # Notification system
    notification_system,
    
    # Stores and intervals
    dcc.Store(id="enhancex-session-store"),
    dcc.Store(id="enhancex-theme-store"),
    dcc.Store(id="enhancex-preferences-store"),
    
    dcc.Interval(
        id='session-update-interval',
        interval=1000,  # 1 second
        n_intervals=0
    ),
    
    dcc.Interval(
        id='memory-update-interval',
        interval=5000,  # 5 seconds
        n_intervals=0
    ),
    
    # Load Font Awesome for icons
    html.Link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
    )
])

# Notification callbacks
@app.callback(
    Output("enhancex-notification-container", "children"),
    [Input("show-info-notification", "n_clicks"),
     Input("show-success-notification", "n_clicks"),
     Input("show-warning-notification", "n_clicks"),
     Input("show-error-notification", "n_clicks")],
    [State("enhancex-notification-container", "children")]
)
def show_notifications(info_clicks, success_clicks, warning_clicks, error_clicks, current_notifications):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_notifications or []
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "show-info-notification":
        add_notification("This is an information notification", "info")
        enhancex.session_memory.record_interaction("notification", {"type": "info"})
    elif button_id == "show-success-notification":
        add_notification("Operation completed successfully!", "success")
        enhancex.session_memory.record_interaction("notification", {"type": "success"})
    elif button_id == "show-warning-notification":
        add_notification("Warning: This action may have consequences", "warning")
        enhancex.session_memory.record_interaction("notification", {"type": "warning"})
    elif button_id == "show-error-notification":
        add_notification("Error: Something went wrong", "danger")
        enhancex.session_memory.record_interaction("notification", {"type": "error"})
    
    return current_notifications or []

# Session interaction callbacks
@app.callback(
    Output("interactions-output", "children"),
    [Input("record-interaction", "n_clicks"),
     Input("show-interactions", "n_clicks")]
)
def handle_interaction_buttons(record_clicks, show_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "record-interaction":
        interaction_data = {"action": "button_click", "timestamp": time.time()}
        enhancex.session_memory.record_interaction("user_action", interaction_data)
        return html.Div(["Interaction recorded: ", html.Code(str(interaction_data))])
    
    elif button_id == "show-interactions":
        recent_interactions = enhancex.session_memory.get_recent_interactions(5)
        if not recent_interactions:
            return "No interactions recorded yet."
        
        interaction_items = []
        for interaction in recent_interactions:
            interaction_items.append(html.Li([
                html.Strong(f"{interaction.interaction_type}: "),
                html.Span(f"{interaction.timestamp}"),
                html.Br(),
                html.Code(str(interaction.data))
            ]))
        
        return html.Div([
            html.H6("Recent Interactions:"),
            html.Ul(interaction_items)
        ])
    
    return ""

# Session info update
@app.callback(
    Output("session-info", "children"),
    [Input("session-update-interval", "n_intervals")]
)
def update_session_info(n_intervals):
    if enhancex and enhancex.session_memory:
        session_id = enhancex.session_memory.session_id
        start_time = enhancex.session_memory.session_start
        duration = enhancex.session_memory.session_duration
        interaction_count = len(enhancex.session_memory.get_recent_interactions(100))
        
        return html.Div([
            html.P([html.Strong("Session ID: "), html.Code(session_id)]),
            html.P([html.Strong("Started: "), html.Span(str(start_time))]),
            html.P([html.Strong("Duration: "), html.Span(f"{duration:.1f} seconds")]),
            html.P([html.Strong("Interactions: "), html.Span(str(interaction_count))])
        ])
    
    return "Session not initialized"

# Memory store update
@app.callback(
    Output("memory-store-display", "children"),
    [Input("memory-update-interval", "n_intervals")]
)
def update_memory_display(n_intervals):
    if enhancex and enhancex.memory_store:
        memory_entries = enhancex.memory_store.get_all_entries()
        
        if not memory_entries:
            return "No memory entries yet."
        
        memory_items = []
        for entry in memory_entries:
            memory_items.append(html.Div([
                html.Strong(f"{entry.memory_type}: "),
                html.Span(entry.content),
                html.Small(f" ({entry.timestamp})", className="text-muted")
            ], className="mb-2"))
        
        return html.Div(memory_items)
    
    return "Memory store not initialized"

# Keep session alive
@app.callback(
    Output("enhancex-session-store", "data"),
    [Input("session-update-interval", "n_intervals")]
)
def keep_session_alive(n_intervals):
    if enhancex and enhancex.session_memory:
        enhancex.session_memory.keep_alive()
        return {"session_id": enhancex.session_memory.session_id, "timestamp": enhancex.session_memory.last_activity}
    return {}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)