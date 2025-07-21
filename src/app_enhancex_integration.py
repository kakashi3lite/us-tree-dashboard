# EnhanceX Integration with Main App

import dash
from dash import html

# Import EnhanceX main integration module
from src.enhancex_main import integrate_enhancex

def integrate_enhancex_with_main_app(app):
    """
    Integrate EnhanceX with the main Dash application.
    This function should be called after the app is created but before it's run.
    
    Args:
        app: The Dash application instance
        
    Returns:
        The modified Dash application with EnhanceX integrated
    """
    # Store the original layout
    original_layout = app.layout
    
    # Integrate EnhanceX with the app
    app = integrate_enhancex(app, original_layout)
    
    return app

def apply_enhancex_to_app(app):
    """
    Apply EnhanceX to an existing Dash app.
    This is the main function that should be imported and used in app.py.
    
    Example usage in app.py:
    ```python
    from src.app_enhancex_integration import apply_enhancex_to_app
    
    # Create the app
    app = dash.Dash(__name__, ...)
    
    # Define layout
    app.layout = html.Div([...])
    
    # Register callbacks
    @app.callback(...)
    def update_graph(...):
        ...
    
    # Apply EnhanceX
    app = apply_enhancex_to_app(app)
    
    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    
    Args:
        app: The Dash application instance
        
    Returns:
        The modified Dash application with EnhanceX integrated
    """
    return integrate_enhancex_with_main_app(app)