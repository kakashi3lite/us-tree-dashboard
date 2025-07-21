# EnhanceX Integration for US Tree Dashboard

## Overview

This document provides instructions for integrating the EnhanceX memory management system with the US Tree Dashboard application. EnhanceX enhances the user experience by providing contextual memory, user preference tracking, session management, and a notification system.

## Features

EnhanceX adds the following features to the dashboard:

- **Theme Switching**: Users can switch between light, dark, and high-contrast themes
- **Visualization Preferences**: Customize chart types, colors, and display options
- **Project Context Management**: Save and restore dashboard states and contexts
- **Notification System**: In-app notifications for important events and updates
- **Session Memory**: Track user interactions and maintain session state
- **User Preferences**: Remember user settings across sessions

## Installation

1. The EnhanceX system is already included in the codebase under the `src/memory` directory
2. The integration files are located in:
   - `src/enhancex_main.py`: Main integration module
   - `src/app_enhancex_integration.py`: Application integration helpers
   - `src/components/`: EnhanceX UI components
   - `assets/enhancex.css`: Styling for EnhanceX components

## Usage

### Running with EnhanceX

To run the dashboard with EnhanceX integration:

```bash
python app_with_enhancex.py
```

This will start the dashboard with all EnhanceX features enabled.

### Integrating with Existing App

If you want to integrate EnhanceX with your existing app.py:

1. Import the integration module:
   ```python
   from src.app_enhancex_integration import apply_enhancex_to_app
   ```

2. Apply EnhanceX to your app after defining the layout and callbacks:
   ```python
   # Apply EnhanceX to the app
   app = apply_enhancex_to_app(app)
   ```

## Components

### EnhanceX Sidebar

The EnhanceX sidebar provides access to all EnhanceX features. It can be toggled by clicking the gear icon in the top-right corner of the dashboard.

### Theme Switcher

The theme switcher allows users to select between:
- Light theme (default)
- Dark theme
- High-contrast theme

Theme preferences are saved across sessions.

### Visualization Preferences

Users can customize visualization settings:
- Chart type (bar, line, pie, etc.)
- Color palette
- Grid lines
- Legend position

### Project Context Manager

The project context manager allows users to:
- Save the current dashboard state as a named context
- Load previously saved contexts
- Add notes and descriptions to contexts
- Share contexts with other users

### Notification System

The notification system provides:
- Toast notifications for important events
- Notification history
- Categorized notifications (info, success, warning, error)

## Directory Structure

```
src/
├── memory/                  # Core EnhanceX memory system
│   ├── __init__.py          # Package exports
│   ├── memory_store.py      # Long-term memory storage
│   ├── context_manager.py   # Project context management
│   ├── preference_tracker.py # User preferences
│   ├── session_memory.py    # Session state and interactions
│   ├── enhancex.py          # Main EnhanceX class
│   └── examples.py          # Usage examples
├── components/              # UI Components
│   ├── theme_switcher.py    # Theme selection component
│   ├── visualization_preferences.py # Chart customization
│   ├── project_context_manager.py # Context UI
│   ├── notification_system.py # Notifications
│   └── enhancex_dashboard.py # Main dashboard integration
├── enhancex_main.py         # Main integration module
└── app_enhancex_integration.py # App integration helpers
assets/
└── enhancex.css            # EnhanceX styles
```

## Data Storage

EnhanceX stores data in the following directories:

- `data/enhancex/memory/`: Long-term memory storage
- `data/enhancex/contexts/`: Saved project contexts
- `data/enhancex/preferences/`: User preferences
- `data/enhancex/sessions/`: Active session data

## Customization

### Modifying Themes

Themes can be customized by editing the CSS variables in `assets/enhancex.css`.

### Adding New Features

To add new EnhanceX features:

1. Create a new component in `src/components/`
2. Add the component to the EnhanceX sidebar in `src/enhancex_main.py`
3. Register any required callbacks

## Troubleshooting

### Common Issues

- **Missing data directories**: EnhanceX will create required directories automatically
- **Theme not applying**: Make sure the `app-container` class is applied to the main layout
- **Preferences not saving**: Check file permissions for the data directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.