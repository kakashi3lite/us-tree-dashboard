"""
VOSviewer Network Visualization Component

This component integrates VOSviewer for network visualization into the conservation dashboard.
It provides functionality to create and display network visualizations for species relationships,
biodiversity hotspots, and conservation status.
"""

import json
import base64
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..memory import EnhanceX

# Initialize EnhanceX
enhancex = EnhanceX()

# VOSviewer configuration options
VOSVIEWER_OPTIONS = {
    "visualization": [
        {"label": "Network Visualization", "value": "network"},
        {"label": "Density Visualization", "value": "density"},
        {"label": "Cluster Density Visualization", "value": "cluster_density"}
    ],
    "colorScheme": [
        {"label": "Default", "value": "default"},
        {"label": "Viridis", "value": "viridis"},
        {"label": "Magma", "value": "magma"},
        {"label": "Rainbow", "value": "rainbow"}
    ],
    "networkType": [
        {"label": "Species Relationships", "value": "species_relationship"},
        {"label": "Biodiversity Hotspots", "value": "biodiversity_hotspots"},
        {"label": "Conservation Status", "value": "conservation_status"}
    ]
}

def create_vosviewer_network():
    """Create a VOSviewer network visualization component."""
    return html.Div(
        className="vosviewer-container",
        children=[
            html.H5("Network Visualization", className="mb-3"),
            
            # Configuration panel
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # Network type selection
                        dbc.Col([
                            html.Label("Network Type", className="form-label"),
                            dcc.Dropdown(
                                id="vosviewer-network-type",
                                options=VOSVIEWER_OPTIONS["networkType"],
                                value="species_relationship",
                                clearable=False
                            )
                        ], md=4),
                        
                        # Visualization type
                        dbc.Col([
                            html.Label("Visualization Type", className="form-label"),
                            dcc.Dropdown(
                                id="vosviewer-visualization-type",
                                options=VOSVIEWER_OPTIONS["visualization"],
                                value="network",
                                clearable=False
                            )
                        ], md=4),
                        
                        # Color scheme
                        dbc.Col([
                            html.Label("Color Scheme", className="form-label"),
                            dcc.Dropdown(
                                id="vosviewer-color-scheme",
                                options=VOSVIEWER_OPTIONS["colorScheme"],
                                value="default",
                                clearable=False
                            )
                        ], md=4)
                    ]),
                    
                    dbc.Row([
                        # Minimum node size
                        dbc.Col([
                            html.Label("Min Node Size", className="form-label"),
                            dcc.Slider(
                                id="vosviewer-min-node-size",
                                min=1,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(1, 11, 2)}
                            )
                        ], md=4),
                        
                        # Maximum node size
                        dbc.Col([
                            html.Label("Max Node Size", className="form-label"),
                            dcc.Slider(
                                id="vosviewer-max-node-size",
                                min=10,
                                max=50,
                                step=5,
                                value=30,
                                marks={i: str(i) for i in range(10, 51, 10)}
                            )
                        ], md=4),
                        
                        # Node distance
                        dbc.Col([
                            html.Label("Node Distance", className="form-label"),
                            dcc.Slider(
                                id="vosviewer-node-distance",
                                min=1,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(1, 11, 2)}
                            )
                        ], md=4)
                    ], className="mt-3"),
                    
                    # Apply button
                    dbc.Button(
                        "Apply Settings",
                        id="vosviewer-apply-button",
                        color="primary",
                        className="mt-3"
                    ),
                    
                    # Download VOSviewer file button
                    dbc.Button(
                        "Download Network File",
                        id="vosviewer-download-button",
                        color="secondary",
                        className="mt-3 ms-2"
                    ),
                    dcc.Download(id="vosviewer-download")
                ])
            ], className="mb-3"),
            
            # Visualization area - either iframe or plotly fallback
            html.Div([
                # Message when loading
                html.Div(
                    id="vosviewer-loading-message",
                    children=[
                        html.I(className="fas fa-spinner fa-spin me-2"),
                        html.Span("Preparing network visualization...")
                    ],
                    style={"textAlign": "center", "marginTop": "50px"}
                ),
                
                # VOSviewer Online iframe
                html.Iframe(
                    id="vosviewer-iframe",
                    style={"width": "100%", "height": "600px", "border": "none", "display": "none"}
                ),
                
                # Fallback visualization using Plotly
                html.Div(
                    id="vosviewer-fallback-container",
                    children=[
                        dcc.Graph(
                            id="vosviewer-network-graph",
                            style={"height": "600px", "display": "none"}
                        )
                    ]
                )
            ], className="vosviewer-visualization-area"),
            
            # Hidden components for state management
            dcc.Store(id="vosviewer-data"),
            dcc.Store(id="vosviewer-network-state"),
            dcc.Store(id="vosviewer-embed-url")
        ]
    )

def register_callbacks(app):
    """Register callbacks for VOSviewer network visualization."""
    
    # Generate network data when settings are applied
    @app.callback(
        [Output("vosviewer-data", "data"),
         Output("vosviewer-loading-message", "style")],
        [Input("vosviewer-apply-button", "n_clicks")],
        [State("vosviewer-network-type", "value"),
         State("vosviewer-visualization-type", "value"),
         State("vosviewer-color-scheme", "value"),
         State("vosviewer-min-node-size", "value"),
         State("vosviewer-max-node-size", "value"),
         State("vosviewer-node-distance", "value")]
    )
    def generate_network_data(n_clicks, network_type, visualization_type, 
                             color_scheme, min_node_size, max_node_size, node_distance):
        if n_clicks is None:
            # Initial load - show loading message
            return None, {"textAlign": "center", "marginTop": "50px", "display": "block"}
        
        # Show loading message while processing
        loading_style = {"textAlign": "center", "marginTop": "50px", "display": "block"}
        
        try:
            # Generate network data based on type
            data = create_network_data(
                network_type=network_type,
                min_node_size=min_node_size,
                max_node_size=max_node_size,
                node_distance=node_distance
            )
            
            # Save user preference
            enhancex.set_user_preference("visualization", "vosviewer_network_type", network_type)
            enhancex.set_user_preference("visualization", "vosviewer_visualization_type", visualization_type)
            enhancex.set_user_preference("visualization", "vosviewer_color_scheme", color_scheme)
            
            # Record interaction
            enhancex.record_interaction(
                interaction_type="vosviewer_network_generation",
                data={
                    "network_type": network_type,
                    "visualization_type": visualization_type,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            )
            
            # Hide loading message
            loading_style = {"display": "none"}
            
            return data, loading_style
            
        except Exception as e:
            print(f"Error generating network data: {e}")
            # Hide loading message but show error in the UI
            return None, {"display": "none"}
    
    # Generate VOSviewer embed URL
    @app.callback(
        Output("vosviewer-embed-url", "data"),
        [Input("vosviewer-data", "data"),
         Input("vosviewer-visualization-type", "value"),
         Input("vosviewer-color-scheme", "value")]
    )
    def generate_vosviewer_url(data, visualization_type, color_scheme):
        if not data:
            return None
        
        try:
            # Convert data to JSON
            data_json = json.dumps(data)
            
            # Encode data for URL
            encoded_data = base64.b64encode(data_json.encode('utf-8')).decode('utf-8')
            
            # Create VOSviewer Online URL with parameters
            vosviewer_url = (
                f"https://app.vosviewer.com/?json={encoded_data}"
                f"&visualization_type={visualization_type}"
                f"&color_scheme={color_scheme}"
            )
            
            return vosviewer_url
            
        except Exception as e:
            print(f"Error generating VOSviewer URL: {e}")
            return None
    
    # Update iframe or fallback visualization
    @app.callback(
        [Output("vosviewer-iframe", "src"),
         Output("vosviewer-iframe", "style"),
         Output("vosviewer-network-graph", "figure"),
         Output("vosviewer-network-graph", "style")],
        [Input("vosviewer-embed-url", "data"),
         Input("vosviewer-data", "data")]
    )
    def update_visualization(embed_url, data):
        # Default styles
        iframe_style = {"width": "100%", "height": "600px", "border": "none", "display": "none"}
        graph_style = {"height": "600px", "display": "none"}
        
        # Empty figure
        empty_fig = go.Figure()
        
        if embed_url:
            # Use VOSviewer Online iframe
            iframe_style["display"] = "block"
            return embed_url, iframe_style, empty_fig, graph_style
        
        elif data:
            # Use Plotly fallback visualization
            fig = create_fallback_network_visualization(data)
            graph_style["display"] = "block"
            return "", iframe_style, fig, graph_style
        
        # Nothing to display
        return "", iframe_style, empty_fig, graph_style
    
    # Download VOSviewer network file
    @app.callback(
        Output("vosviewer-download", "data"),
        Input("vosviewer-download-button", "n_clicks"),
        State("vosviewer-data", "data"),
        State("vosviewer-network-type", "value"),
        prevent_initial_call=True
    )
    def download_network_file(n_clicks, data, network_type):
        if not data:
            return None
        
        # Create JSON data for download
        json_data = json.dumps(data, indent=2)
        
        # Record download interaction
        enhancex.record_interaction(
            interaction_type="vosviewer_network_download",
            data={
                "network_type": network_type,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        )
        
        # Return file download
        return dict(
            content=json_data,
            filename=f"vosviewer_network_{network_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            type="application/json"
        )

def create_network_data(network_type: str, min_node_size: int = 5, 
                       max_node_size: int = 30, node_distance: int = 5) -> Dict:
    """
    Create network data for VOSviewer based on the selected network type.
    
    Args:
        network_type: Type of network to create (species_relationship, biodiversity_hotspots, conservation_status)
        min_node_size: Minimum node size
        max_node_size: Maximum node size
        node_distance: Distance between nodes
        
    Returns:
        Dictionary with network data in VOSviewer format
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    if network_type == "species_relationship":
        # Load species data
        try:
            species_file = data_dir / "conservation_status_20250614.csv"
            species_df = pd.read_csv(species_file)
            
            # Create network of species relationships
            nodes = []
            links = []
            
            # Create nodes for each species
            for idx, row in species_df.iterrows():
                # Calculate node weight based on conservation status
                weight = 1.0
                if row['conservation_status'] == 'Critically Endangered':
                    weight = 1.0
                elif row['conservation_status'] == 'Endangered':
                    weight = 0.8
                elif row['conservation_status'] == 'Vulnerable':
                    weight = 0.6
                elif row['conservation_status'] == 'Near Threatened':
                    weight = 0.4
                else:  # Least Concern or other
                    weight = 0.2
                
                # Create node
                nodes.append({
                    "id": idx,
                    "label": row['species'],
                    "description": f"{row['common_name']} - {row['conservation_status']}",
                    "weight": weight,
                    "cluster": get_cluster_for_status(row['conservation_status'])
                })
            
            # Create links between species with similar threats or conservation status
            for i in range(len(species_df)):
                for j in range(i+1, len(species_df)):
                    if species_df.iloc[i]['conservation_status'] == species_df.iloc[j]['conservation_status']:
                        # Species with the same conservation status
                        links.append({
                            "source": i,
                            "target": j,
                            "weight": 1.0
                        })
                    
                    if species_df.iloc[i]['main_threats'] == species_df.iloc[j]['main_threats']:
                        # Species with the same threats
                        links.append({
                            "source": i,
                            "target": j,
                            "weight": 0.8
                        })
            
            # Format for VOSviewer
            return {
                "network": {
                    "nodes": nodes,
                    "links": links
                },
                "options": {
                    "scale": {
                        "min": min_node_size,
                        "max": max_node_size
                    },
                    "distance": node_distance
                }
            }
            
        except Exception as e:
            print(f"Error creating species relationship network: {e}")
            # Return empty network
            return {"network": {"nodes": [], "links": []}, "options": {}}
    
    elif network_type == "biodiversity_hotspots":
        # Load biodiversity hotspots data
        try:
            hotspots_file = data_dir / "biodiversity_hotspots_20250614.csv"
            hotspots_df = pd.read_csv(hotspots_file)
            
            # Create network of biodiversity hotspots
            nodes = []
            links = []
            
            # Create nodes for each hotspot
            for idx, row in hotspots_df.iterrows():
                # Node weight based on endemic species
                weight = min(1.0, row['endemic_species'] / 40000.0)  # Normalize by max endemic species
                
                # Create node
                nodes.append({
                    "id": idx,
                    "label": row['hotspot_name'],
                    "description": f"Country: {row['country']}, Endemic species: {row['endemic_species']}",
                    "weight": weight,
                    "cluster": get_cluster_for_threat(row['threat_level'])
                })
            
            # Create links between geographically close hotspots
            # Using a simple distance calculation between latitude/longitude
            for i in range(len(hotspots_df)):
                for j in range(i+1, len(hotspots_df)):
                    distance = calculate_distance(
                        hotspots_df.iloc[i]['latitude'], hotspots_df.iloc[i]['longitude'],
                        hotspots_df.iloc[j]['latitude'], hotspots_df.iloc[j]['longitude']
                    )
                    
                    # Link hotspots that are close to each other (distance threshold)
                    if distance < 50:  # Arbitrary threshold
                        links.append({
                            "source": i,
                            "target": j,
                            "weight": 1.0 - (distance / 50.0)  # Weight inversely proportional to distance
                        })
                    
                    # Also link hotspots with similar threat levels
                    if hotspots_df.iloc[i]['threat_level'] == hotspots_df.iloc[j]['threat_level']:
                        links.append({
                            "source": i,
                            "target": j,
                            "weight": 0.8
                        })
            
            # Format for VOSviewer
            return {
                "network": {
                    "nodes": nodes,
                    "links": links
                },
                "options": {
                    "scale": {
                        "min": min_node_size,
                        "max": max_node_size
                    },
                    "distance": node_distance
                }
            }
            
        except Exception as e:
            print(f"Error creating biodiversity hotspots network: {e}")
            # Return empty network
            return {"network": {"nodes": [], "links": []}, "options": {}}
    
    elif network_type == "conservation_status":
        # Load conservation status data
        try:
            status_file = data_dir / "conservation_status_20250614.csv"
            status_df = pd.read_csv(status_file)
            
            # Create network based on conservation status groups
            # Group species by conservation status
            status_groups = status_df.groupby('conservation_status')
            
            nodes = []
            links = []
            
            # Create nodes for each conservation status
            status_ids = {}
            for i, (status, group) in enumerate(status_groups):
                # Node weight based on number of species with this status
                weight = min(1.0, len(group) / 10.0)  # Normalize by max expected group size
                
                # Store status ID for link creation
                status_ids[status] = i
                
                # Create node
                nodes.append({
                    "id": i,
                    "label": status,
                    "description": f"{len(group)} species with {status} status",
                    "weight": weight,
                    "cluster": get_cluster_for_status(status)
                })
            
            # Create links between conservation statuses
            # Link adjacent statuses (e.g., Critically Endangered <-> Endangered <-> Vulnerable)
            status_order = [
                "Critically Endangered", "Endangered", "Vulnerable", 
                "Near Threatened", "Least Concern", "Extinct in Wild"
            ]
            
            for i in range(len(status_order) - 1):
                if status_order[i] in status_ids and status_order[i+1] in status_ids:
                    links.append({
                        "source": status_ids[status_order[i]],
                        "target": status_ids[status_order[i+1]],
                        "weight": 1.0
                    })
            
            # Format for VOSviewer
            return {
                "network": {
                    "nodes": nodes,
                    "links": links
                },
                "options": {
                    "scale": {
                        "min": min_node_size,
                        "max": max_node_size
                    },
                    "distance": node_distance
                }
            }
            
        except Exception as e:
            print(f"Error creating conservation status network: {e}")
            # Return empty network
            return {"network": {"nodes": [], "links": []}, "options": {}}
    
    # Default: return empty network
    return {"network": {"nodes": [], "links": []}, "options": {}}

def create_fallback_network_visualization(data: Dict) -> go.Figure:
    """
    Create a fallback network visualization using Plotly when VOSviewer iframe is not available.
    
    Args:
        data: Network data in VOSviewer format
        
    Returns:
        Plotly figure with network visualization
    """
    # Extract nodes and links
    nodes = data.get("network", {}).get("nodes", [])
    links = data.get("network", {}).get("links", [])
    
    if not nodes:
        # Return empty figure
        return go.Figure()
    
    # Create node positions using a simple force-directed layout
    positions = calculate_node_positions(nodes, links)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in nodes:
        x, y = positions[node["id"]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node['label']}<br>{node.get('description', '')}")
        
        # Node size based on weight
        size = 10 + (node.get("weight", 0.5) * 20)
        node_size.append(size)
        
        # Node color based on cluster
        node_color.append(node.get("cluster", 0))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white')
        )
    )
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for link in links:
        source = link["source"]
        target = link["target"]
        
        source_pos = positions[source]
        target_pos = positions[target]
        
        edge_x.extend([source_pos[0], target_pos[0], None])
        edge_y.extend([source_pos[1], target_pos[1], None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title='Network Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                  )
    
    return fig

def calculate_node_positions(nodes: List[Dict], links: List[Dict]) -> Dict[int, Tuple[float, float]]:
    """
    Calculate node positions for network visualization using a simple force-directed layout.
    
    Args:
        nodes: List of nodes
        links: List of links between nodes
        
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    # Initialize random positions
    positions = {}
    for node in nodes:
        positions[node["id"]] = (np.random.random(), np.random.random())
    
    # Very simple force-directed layout
    # In a real implementation, you would use a proper algorithm
    iterations = 50
    for _ in range(iterations):
        # Calculate forces
        forces = {node["id"]: [0, 0] for node in nodes}
        
        # Repulsive forces between all nodes
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Calculate distance
                pos1 = positions[node1["id"]]
                pos2 = positions[node2["id"]]
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                distance = max(0.01, np.sqrt(dx*dx + dy*dy))
                
                # Repulsive force
                force = 0.1 / (distance * distance)
                fx = force * dx / distance
                fy = force * dy / distance
                
                forces[node1["id"]][0] += fx
                forces[node1["id"]][1] += fy
                forces[node2["id"]][0] -= fx
                forces[node2["id"]][1] -= fy
        
        # Attractive forces along links
        for link in links:
            source = link["source"]
            target = link["target"]
            
            # Calculate distance
            pos1 = positions[source]
            pos2 = positions[target]
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            distance = max(0.01, np.sqrt(dx*dx + dy*dy))
            
            # Attractive force
            force = 0.05 * distance * link.get("weight", 1.0)
            fx = force * dx / distance
            fy = force * dy / distance
            
            forces[source][0] -= fx
            forces[source][1] -= fy
            forces[target][0] += fx
            forces[target][1] += fy
        
        # Apply forces
        for node in nodes:
            node_id = node["id"]
            fx, fy = forces[node_id]
            x, y = positions[node_id]
            
            # Limit force magnitude
            magnitude = np.sqrt(fx*fx + fy*fy)
            if magnitude > 0.1:
                fx = 0.1 * fx / magnitude
                fy = 0.1 * fy / magnitude
            
            # Update position
            positions[node_id] = (x + fx, y + fy)
    
    # Normalize positions to [0,1] range
    min_x = min(p[0] for p in positions.values())
    max_x = max(p[0] for p in positions.values())
    min_y = min(p[1] for p in positions.values())
    max_y = max(p[1] for p in positions.values())
    
    x_range = max_x - min_x or 1.0
    y_range = max_y - min_y or 1.0
    
    for node_id, (x, y) in positions.items():
        positions[node_id] = ((x - min_x) / x_range, (y - min_y) / y_range)
    
    return positions

def get_cluster_for_status(status: str) -> int:
    """Map conservation status to cluster ID"""
    status_map = {
        "Critically Endangered": 0,
        "Endangered": 1,
        "Vulnerable": 2,
        "Near Threatened": 3,
        "Least Concern": 4,
        "Extinct in Wild": 5
    }
    return status_map.get(status, 6)

def get_cluster_for_threat(threat: str) -> int:
    """Map threat level to cluster ID"""
    threat_map = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3
    }
    return threat_map.get(threat, 4)

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate approximate distance between two points in latitude and longitude.
    Uses a simplified formula for demonstration purposes.
    
    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2
        
    Returns:
        Approximate distance in degrees (not in km/miles)
    """
    # This is a simplified distance calculation
    # In a real implementation, you would use the haversine formula
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
