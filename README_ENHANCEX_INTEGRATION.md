# EnhanceX Integration for US Tree Dashboard

## Overview

This document provides instructions on how to integrate EnhanceX with the US Tree Dashboard application. EnhanceX adds several powerful features to enhance the user experience:

- **Theme Switching**: Allow users to switch between light, dark, and high-contrast themes
- **Visualization Preferences**: Customize chart types, color palettes, grid lines, and legend positions
- **Project Context Management**: Save and load project contexts with key-value pairs
- **Notification System**: Display notifications to users and maintain a notification history
- **Session Memory**: Track user interactions and preferences across sessions

## Files Created

- `app_with_enhancex_integration.py`: A complete implementation of the dashboard with EnhanceX integration

## How to Use

### Option 1: Use the Integrated App

The simplest way to use EnhanceX is to run the integrated application:

```bash
python app_with_enhancex_integration.py
```

This will start the dashboard with all EnhanceX features enabled.

### Option 2: Integrate with Your Existing App

If you want to integrate EnhanceX with your own version of the dashboard, follow these steps:

1. Import the EnhanceX integration module:

```python
from src.app_enhancex_integration import apply_enhancex_to_app
```

2. After defining your app layout and callbacks, apply EnhanceX:

```python
# Apply EnhanceX to the app
app = apply_enhancex_to_app(app)
```

3. Run your app as usual:

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
```

## EnhanceX Components

### Theme Switcher

Allows users to switch between different visual themes:

- Light Theme: Default theme with light background
- Dark Theme: Dark mode for reduced eye strain
- High Contrast: Accessible theme with high contrast colors

### Visualization Preferences

Customize visualization settings:

- Chart Type: Select different chart types for data visualization
- Color Palette: Choose from various color schemes
- Grid Lines: Toggle grid lines on/off
- Legend Position: Set the position of chart legends

### Project Context Manager

Manage project contexts:

- Save current context with a name and description
- Load previously saved contexts
- Add and remove key-value pairs from the context

### Notification System

Display notifications to users:

- Show notifications of different types (info, success, warning, error)
- View notification history
- Track unread notifications

## Directory Structure

The EnhanceX integration uses the following directory structure:

```
src/
  ├── memory/                  # EnhanceX core functionality
  ├── components/              # EnhanceX UI components
  │   ├── enhancex_dashboard.py    # Main dashboard component
  │   ├── theme_switcher.py        # Theme switching component
  │   ├── visualization_preferences.py  # Visualization settings
  │   ├── project_context_manager.py    # Context management
  │   └── notification_system.py        # Notification system
  ├── enhancex_main.py         # Main integration module
  └── app_enhancex_integration.py  # Integration helper
```

## Data Storage

EnhanceX stores data in the following locations:

- `data/enhancex/memory/`: Session memory data
- `data/enhancex/contexts/`: Saved project contexts
- `data/enhancex/preferences/`: User preferences
- `data/enhancex/sessions/`: Session data

## Customization

### Modifying Themes

Themes are defined in `assets/enhancex.css`. You can modify this file to customize the appearance of the themes.

### Adding New Features

To add new EnhanceX features:

1. Create a new component in `src/components/`
2. Integrate it in `src/enhancex_main.py`
3. Update the EnhanceX dashboard in `src/components/enhancex_dashboard.py`

## Troubleshooting

### Missing Data Directories

If you encounter errors related to missing data directories, ensure that the following directories exist:

```bash
mkdir -p data/enhancex/memory
mkdir -p data/enhancex/contexts
mkdir -p data/enhancex/preferences
mkdir -p data/enhancex/sessions
```

### Themes Not Applying

If themes are not applying correctly, check that `assets/enhancex.css` is properly loaded. The file should be in the `assets` directory of your application.

### Preferences Not Saving

If preferences are not being saved, check the permissions of the `data/enhancex/preferences` directory to ensure it is writable.

## Docker Integration

If you're using Docker, make sure to update your `Dockerfile` and `docker-compose.yml` to use the EnhanceX-integrated app:

```yaml
# docker-compose.yml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - '8050:8050'
    command: python app_with_enhancex_integration.py
```

## Contributing

To contribute to the EnhanceX integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the same license as the US Tree Dashboard.