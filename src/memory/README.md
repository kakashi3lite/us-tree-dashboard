# EnhanceX Memory Management System

EnhanceX is a comprehensive memory management system for the US Tree Dashboard application, providing contextual memory capabilities to enhance user experience and application performance.

## Features

### Session Memory
- Track user interactions during a dashboard session
- Maintain session state and user activity
- Record and analyze user behavior patterns
- Support for interaction handlers to respond to user actions

### User Preferences
- Store and retrieve user preferences
- Personalize UI elements based on user preferences
- Support for multiple preference categories
- Default preferences for new users

### Project Context
- Create and manage project contexts
- Store project-specific data and settings
- Track context history and changes
- Support for multiple concurrent projects

### Memory Persistence
- Short-term memory for session data
- Long-term memory for persistent storage
- Automatic cleanup of expired memory entries
- JSON-based storage with configurable paths

## Architecture

The EnhanceX system consists of the following components:

- **EnhanceX**: Main integration class providing a unified interface
- **MemoryStore**: Core storage system for all memory types
- **SessionMemory**: Manages short-term session data and interactions
- **PreferenceTracker**: Handles user preference storage and retrieval
- **ContextManager**: Manages project contexts and history

## Usage

### Basic Usage

```python
from src.memory import EnhanceX

# Initialize EnhanceX
enhancex = EnhanceX()

# Start a session
session_id = enhancex.start_session()

# Set user preferences
enhancex.set_user_preference("ui", "theme", "dark")

# Record user interactions
enhancex.record_interaction(
    "filter_change", 
    {"filter": "city", "value": "New York"}
)

# Update session state
enhancex.update_session_state({
    "current_view": "map",
    "filters": {"city": "New York"}
})

# Create a project context
context_id = enhancex.create_project_context(
    "Tree Analysis",
    "Analysis of tree health in urban areas",
    {"dataset": "urban_trees_2023"}
)

# End session
enhancex.end_session()
```

### Integration with Dash

```python
from dash import Dash, Input, Output, State
from src.memory import EnhanceX

# Initialize EnhanceX
enhancex = EnhanceX()

# Initialize Dash app
app = Dash(__name__)

# Start session on app load
@app.callback(
    Output('session-store', 'data'),
    Input('url', 'pathname')
)
def initialize_session(pathname):
    if pathname == '/':
        session_id = enhancex.start_session({
            'initial_page': pathname
        })
        return {'session_id': session_id}
    return dash.no_update

# Record interactions
@app.callback(
    Output('interaction-log', 'children'),
    Input('filter-dropdown', 'value'),
    State('session-store', 'data')
)
def on_filter_change(value, session_data):
    if session_data and value:
        enhancex.record_interaction(
            'filter_change',
            {'filter': 'main_filter', 'value': value}
        )
        # Update session state
        enhancex.update_session_state({
            'current_filter': value
        })
    return f'Filter set to: {value}'
```

## Advanced Features

### Interaction Handlers

```python
# Define an interaction handler
def handle_filter_change(interaction):
    print(f"Filter changed: {interaction.data}")
    # Update UI components, trigger data reloading, etc.

# Register the handler
enhancex.register_interaction_handler("filter_change", handle_filter_change)
```

### Long-term Memory Storage

```python
# Store analysis results for long-term reference
enhancex.store_long_term_memory(
    "analysis_results_2023",
    {
        "timestamp": time.time(),
        "health_index": 0.82,
        "risk_areas": ["downtown", "industrial_zone"]
    }
)

# Retrieve results later
results = enhancex.retrieve_long_term_memory("analysis_results_2023")
```

## Configuration

EnhanceX can be configured with the following parameters:

- **data_dir**: Directory for storing persistent data (defaults to 'data/memory')
- **session_timeout**: Session timeout in minutes (defaults to 30)

```python
# Custom configuration
enhancex = EnhanceX(
    data_dir="/custom/data/path",
    session_timeout=60  # 1 hour timeout
)
```

## Examples

See the `examples.py` file for complete usage examples.