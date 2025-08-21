# VOSviewer Network Visualization for Conservation Dashboard

## Overview

This integration adds powerful network visualization capabilities to the Conservation Dashboard using VOSviewer. It enables users to explore relationships between species, biodiversity hotspots, and conservation statuses through interactive network visualizations.

## Features

- **Multiple Network Types**:
  - **Species Relationship Network**: Visualize connections between species based on conservation status and threats
  - **Biodiversity Hotspot Network**: Explore connections between global biodiversity hotspots
  - **Conservation Status Network**: Understand the relationships between different conservation statuses

- **Customizable Visualizations**:
  - Multiple visualization types (network, density, cluster)
  - Adjustable node size and distance
  - Different color schemes
  - Download network data for external use

- **Integration Options**:
  - As a tab within the EnhanceX dashboard
  - As a standalone page
  - As a modal dialog
  - Embedded within any dashboard section

## Installation

No additional installation is required beyond the existing dashboard dependencies. The VOSviewer integration uses the standard Dash/Plotly components already in use.

## Usage

### Access the Visualization

The VOSviewer network visualization is available through:

1. The EnhanceX dashboard's "Network Visualization" tab
2. The standalone VOSviewer dashboard page (`/vosviewer_dashboard.py`)

### Creating Network Visualizations

1. Select a network type (Species, Biodiversity Hotspots, Conservation Status)
2. Choose visualization settings (visualization type, color scheme)
3. Adjust node size and distance parameters
4. Click "Apply Settings" to generate the visualization
5. Interact with the network using mouse controls (zoom, pan, click on nodes)
6. Optionally download the network data for use in the standalone VOSviewer application

## Data Sources

The network visualizations use data from:

- **Species Data**: `conservation_status_20250614.csv`
- **Biodiversity Hotspots**: `biodiversity_hotspots_20250614.csv`

## Technical Implementation

The implementation consists of:

1. **VOSviewer Network Component** (`src/components/vosviewer_network.py`):
   - Core visualization component
   - Data transformation functions
   - Network generation algorithms
   - Fallback visualization using Plotly

2. **EnhanceX Integration** (`src/components/enhancex_dashboard.py`):
   - Tab integration in EnhanceX dashboard
   - Preference management

3. **Standalone Dashboard** (`vosviewer_dashboard.py`):
   - Dedicated page for VOSviewer visualizations
   - Navigation between different network types

4. **Integration Utilities** (`src/vosviewer_integration.py`):
   - Helper functions for integration into any dashboard
   - Modal dialog implementation
   - Tab creation utilities

## Extending the Visualization

### Adding New Network Types

To add new network types:

1. Update the `VOSVIEWER_OPTIONS` dictionary in `vosviewer_network.py`
2. Add a new case in the `create_network_data()` function
3. Implement the data transformation logic for the new network type

### Customizing Visual Appearance

The visual appearance can be customized by:

1. Modifying the color schemes in the `VOSVIEWER_OPTIONS` dictionary
2. Adjusting the default node size and distance parameters
3. Customizing the fallback visualization in `create_fallback_network_visualization()`

## Troubleshooting

### VOSviewer Online Embedding

If the VOSviewer Online embedding is not working:

1. Check that the data is properly formatted as JSON
2. Verify that the URL encoding is correct
3. Fall back to the Plotly visualization if necessary

### Data Transformation

If you encounter issues with the data transformation:

1. Check that the source data files exist and have the expected format
2. Verify that the columns used in the transformation match the actual data
3. Check for errors in the network generation logic

## Dependencies

- **Dash/Plotly**: For UI components and fallback visualization
- **Pandas/NumPy**: For data processing
- **VOSviewer Online**: For embedded visualization (optional, falls back to Plotly)

## License

This integration is provided under the same license as the main Conservation Dashboard.
