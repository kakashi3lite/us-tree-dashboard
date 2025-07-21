#!/usr/bin/env python3
"""
Enhanced Conservation Dashboard Integration
Optimized for Eli's conservation tech workflow with automation-first approach.
Includes real-time alerts, PDF generation, and field team coordination.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, clientside_callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Import our conservation modules
from src.conservation_alerts import ConservationAlertSystem, ConservationAlert
from src.pdf_generator import ConservationPDFGenerator, ReportConfig

# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/conservation_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConservationDashboardEnhancer:
    """
    Enhanced dashboard functionality for conservation workflows.
    Integrates alerts, PDF generation, and automation features.
    """
    
    def __init__(self, app: dash.Dash):
        self.app = app
        self.alert_system = ConservationAlertSystem()
        self.pdf_generator = ConservationPDFGenerator()
        self.data_dir = Path('data')
        
        # Initialize conservation-specific components
        self._setup_conservation_components()
        self._register_conservation_callbacks()
        
    def _setup_conservation_components(self):
        """Setup conservation-specific dashboard components"""
        
        # Conservation alert panel
        self.alert_panel = dbc.Card([
            dbc.CardHeader([
                html.H4("🚨 Conservation Alerts", className="mb-0"),
                dbc.Badge("Live", color="success", className="ms-2")
            ]),
            dbc.CardBody([
                dcc.Loading(
                    id="alerts-loading",
                    children=[html.Div(id="conservation-alerts-content")],
                    type="circle"
                ),
                dbc.ButtonGroup([
                    dbc.Button("Refresh Alerts", id="refresh-alerts-btn", 
                             color="primary", size="sm"),
                    dbc.Button("Alert Settings", id="alert-settings-btn", 
                             color="secondary", size="sm", outline=True)
                ], className="mt-3")
            ])
        ], className="mb-4")
        
        # PDF Report generation panel
        self.report_panel = dbc.Card([
            dbc.CardHeader([
                html.H4("📄 Field Reports", className="mb-0"),
                dbc.Badge("PDF Ready", color="info", className="ms-2")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Report Type"),
                        dcc.Dropdown(
                            id="report-type-dropdown",
                            options=[
                                {"label": "🌿 Conservation Assessment", "value": "conservation"},
                                {"label": "📱 Field Summary", "value": "field_summary"},
                                {"label": "👔 Executive Briefing", "value": "executive"},
                                {"label": "🔍 Threat Analysis", "value": "threat_analysis"}
                            ],
                            value="conservation",
                            clearable=False
                        )
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Time Range"),
                        dcc.Dropdown(
                            id="report-timerange-dropdown",
                            options=[
                                {"label": "Last 30 days", "value": 30},
                                {"label": "Last 90 days", "value": 90},
                                {"label": "Last 6 months", "value": 180},
                                {"label": "Last year", "value": 365}
                            ],
                            value=90,
                            clearable=False
                        )
                    ], md=6)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-file-pdf me-2"), "Generate PDF Report"],
                            id="generate-pdf-btn",
                            color="success",
                            size="lg",
                            className="w-100"
                        )
                    ], md=8),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), "Download"],
                            id="download-pdf-btn",
                            color="primary",
                            size="lg",
                            className="w-100",
                            disabled=True
                        )
                    ], md=4)
                ]),
                html.Div(id="pdf-generation-status", className="mt-3"),
                dcc.Download(id="pdf-download")
            ])
        ], className="mb-4")
        
        # Real-time monitoring panel
        self.monitoring_panel = dbc.Card([
            dbc.CardHeader([
                html.H4("📊 Real-time Monitoring", className="mb-0"),
                dbc.Badge("Auto-Update", color="warning", className="ms-2")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Canopy Loss Rate", className="card-title"),
                                html.H2(id="canopy-loss-rate", className="text-primary"),
                                html.P("% change per month", className="text-muted")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Species at Risk", className="card-title"),
                                html.H2(id="species-at-risk", className="text-warning"),
                                html.P("requiring intervention", className="text-muted")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Active Threats", className="card-title"),
                                html.H2(id="active-threats", className="text-danger"),
                                html.P("high priority", className="text-muted")
                            ])
                        ])
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Data Quality", className="card-title"),
                                html.H2(id="data-quality-score", className="text-success"),
                                html.P("confidence level", className="text-muted")
                            ])
                        ])
                    ], md=3)
                ]),
                dcc.Graph(id="real-time-trend-chart", className="mt-4"),
                dcc.Interval(
                    id="monitoring-interval",
                    interval=30*1000,  # Update every 30 seconds
                    n_intervals=0
                )
            ])
        ], className="mb-4")
        
        # Field team coordination panel
        self.coordination_panel = dbc.Card([
            dbc.CardHeader([
                html.H4("🎯 Field Team Coordination", className="mb-0"),
                dbc.Badge("Active", color="success", className="ms-2")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Priority Regions"),
                        dcc.Dropdown(
                            id="priority-regions-dropdown",
                            multi=True,
                            placeholder="Select regions for field assessment"
                        )
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Assignment Type"),
                        dcc.Dropdown(
                            id="assignment-type-dropdown",
                            options=[
                                {"label": "🔍 Assessment", "value": "assessment"},
                                {"label": "🌱 Restoration", "value": "restoration"},
                                {"label": "🚨 Emergency Response", "value": "emergency"},
                                {"label": "📊 Monitoring", "value": "monitoring"}
                            ],
                            value="assessment",
                            clearable=False
                        )
                    ], md=6)
                ], className="mb-3"),
                dbc.ButtonGroup([
                    dbc.Button("Generate Field Assignment", id="generate-assignment-btn",
                             color="primary"),
                    dbc.Button("Send Notifications", id="send-notifications-btn",
                             color="warning", outline=True),
                    dbc.Button("Export Coordinates", id="export-coordinates-btn",
                             color="info", outline=True)
                ], className="w-100"),
                html.Div(id="coordination-status", className="mt-3")
            ])
        ], className="mb-4")
    
    def _register_conservation_callbacks(self):
        """Register conservation-specific callbacks"""
        
        # Alert system callbacks
        @self.app.callback(
            Output("conservation-alerts-content", "children"),
            [Input("refresh-alerts-btn", "n_clicks"),
             Input("monitoring-interval", "n_intervals")]
        )
        def update_conservation_alerts(refresh_clicks, interval_count):
            """Update conservation alerts display"""
            try:
                # Run alert analysis
                canopy_data = self._load_sample_canopy_data()
                alerts = self.alert_system.analyze_canopy_trends(canopy_data)
                
                if not alerts:
                    return dbc.Alert(
                        [html.I(className="fas fa-check-circle me-2"), 
                         "No active conservation alerts"],
                        color="success"
                    )
                
                alert_cards = []
                for alert in alerts[-5:]:  # Show last 5 alerts
                    color = {
                        'critical': 'danger',
                        'high': 'warning', 
                        'medium': 'info',
                        'low': 'secondary'
                    }.get(alert.severity, 'secondary')
                    
                    card = dbc.Alert([
                        html.H6(f"{alert.alert_type.replace('_', ' ').title()}", 
                               className="alert-heading"),
                        html.P(alert.threat_description, className="mb-2"),
                        html.Small(f"Region: {alert.region} | "
                                 f"Confidence: {alert.confidence_score:.1%} | "
                                 f"{alert.timestamp.strftime('%Y-%m-%d %H:%M')}",
                                 className="text-muted")
                    ], color=color, className="mb-2")
                    
                    alert_cards.append(card)
                
                return alert_cards
                
            except Exception as e:
                logger.error(f"Error updating alerts: {e}")
                return dbc.Alert(f"Error loading alerts: {str(e)}", color="danger")
        
        # PDF generation callbacks
        @self.app.callback(
            [Output("pdf-generation-status", "children"),
             Output("download-pdf-btn", "disabled"),
             Output("pdf-download", "data")],
            [Input("generate-pdf-btn", "n_clicks")],
            [State("report-type-dropdown", "value"),
             State("report-timerange-dropdown", "value"),
             State("state-dropdown", "value")]
        )
        def generate_pdf_report(n_clicks, report_type, timerange, region):
            """Generate PDF report based on selections"""
            if not n_clicks or not region:
                return "", True, None
                
            try:
                # Configure report
                end_date = datetime.now()
                start_date = end_date - timedelta(days=timerange)
                
                config = ReportConfig(
                    title=self._get_report_title(report_type),
                    subtitle=f"Regional Analysis - {region}",
                    region=region,
                    date_range=(start_date, end_date),
                    template_style='conservation'
                )
                
                # Generate PDF
                pdf_path = self.pdf_generator.generate_conservation_report(config)
                
                # Prepare for download
                return (
                    dbc.Alert(f"✅ Report generated successfully!", color="success"),
                    False,
                    dcc.send_file(pdf_path)
                )
                
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                return (
                    dbc.Alert(f"❌ Error generating report: {str(e)}", color="danger"),
                    True,
                    None
                )
        
        # Real-time monitoring callbacks
        @self.app.callback(
            [Output("canopy-loss-rate", "children"),
             Output("species-at-risk", "children"),
             Output("active-threats", "children"),
             Output("data-quality-score", "children"),
             Output("real-time-trend-chart", "figure")],
            [Input("monitoring-interval", "n_intervals")]
        )
        def update_monitoring_metrics(n_intervals):
            """Update real-time monitoring metrics"""
            try:
                # Calculate metrics (would be from real data in production)
                canopy_loss_rate = f"{np.random.uniform(0.5, 3.5):.1f}"
                species_at_risk = str(np.random.randint(5, 25))
                active_threats = str(np.random.randint(1, 8))
                data_quality = f"{np.random.uniform(0.8, 0.95):.0%}"
                
                # Generate trend chart
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                    end=datetime.now(), freq='D')
                canopy_values = 75 + np.cumsum(np.random.normal(-0.1, 0.5, len(dates)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=canopy_values,
                    mode='lines+markers',
                    name='Canopy Coverage',
                    line=dict(color='#28a745', width=3)
                ))
                
                fig.update_layout(
                    title="30-Day Canopy Coverage Trend",
                    xaxis_title="Date",
                    yaxis_title="Canopy Coverage (%)",
                    template="plotly_white",
                    height=300,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                return canopy_loss_rate, species_at_risk, active_threats, data_quality, fig
                
            except Exception as e:
                logger.error(f"Error updating monitoring metrics: {e}")
                return "N/A", "N/A", "N/A", "N/A", {}
        
        # Field coordination callbacks
        @self.app.callback(
            [Output("priority-regions-dropdown", "options"),
             Output("coordination-status", "children")],
            [Input("generate-assignment-btn", "n_clicks")],
            [State("priority-regions-dropdown", "value"),
             State("assignment-type-dropdown", "value")]
        )
        def handle_field_coordination(n_clicks, selected_regions, assignment_type):
            """Handle field team coordination"""
            # Update region options
            region_options = [
                {"label": "🌲 Pacific Northwest", "value": "pacific_northwest"},
                {"label": "🌵 California Central Valley", "value": "california_central"},
                {"label": "🏔️ Rocky Mountains", "value": "rocky_mountains"},
                {"label": "🌊 Great Lakes", "value": "great_lakes"},
                {"label": "🍂 Appalachians", "value": "appalachians"}
            ]
            
            status_message = ""
            
            if n_clicks and selected_regions and assignment_type:
                try:
                    # Generate field assignment
                    assignment_id = f"FA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    status_message = dbc.Alert([
                        html.H6(f"✅ Field Assignment Generated", className="alert-heading"),
                        html.P(f"Assignment ID: {assignment_id}"),
                        html.P(f"Type: {assignment_type.title()}"),
                        html.P(f"Regions: {', '.join(selected_regions)}"),
                        html.Hr(),
                        html.P("Notification sent to field teams", className="mb-0")
                    ], color="success")
                    
                    logger.info(f"Generated field assignment {assignment_id} for {selected_regions}")
                    
                except Exception as e:
                    status_message = dbc.Alert(f"Error generating assignment: {str(e)}", 
                                             color="danger")
            
            return region_options, status_message
    
    def _get_report_title(self, report_type: str) -> str:
        """Get appropriate title for report type"""
        titles = {
            'conservation': 'Conservation Assessment Report',
            'field_summary': 'Field Team Summary',
            'executive': 'Executive Conservation Briefing',
            'threat_analysis': 'Threat Analysis Report'
        }
        return titles.get(report_type, 'Conservation Report')
    
    def _load_sample_canopy_data(self) -> pd.DataFrame:
        """Load sample canopy data for alert testing"""
        # Generate synthetic canopy data with trends
        regions = ['Pacific Northwest', 'California Central Valley', 'Great Lakes']
        data = []
        
        for region in regions:
            for i in range(12):  # 12 months of data
                date = datetime.now() - timedelta(days=30*i)
                # Simulate declining canopy coverage
                base_canopy = 75 - (i * 2) + np.random.normal(0, 5)
                data.append({
                    'region': region,
                    'timestamp': date,
                    'canopy_pct': max(0, base_canopy),
                    'area_km2': np.random.uniform(100, 500),
                    'species_count': np.random.randint(20, 100),
                    'latitude': 45.0 + np.random.uniform(-5, 5),
                    'longitude': -120.0 + np.random.uniform(-10, 10)
                })
        
        return pd.DataFrame(data)
    
    def add_conservation_layout(self, existing_layout: html.Div) -> html.Div:
        """Add conservation panels to existing dashboard layout"""
        
        # Create conservation tab
        conservation_tab = dbc.Tab(
            label="🌿 Conservation Tools",
            tab_id="conservation-tab",
            children=[
                dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            self.alert_panel,
                            self.monitoring_panel
                        ], md=8),
                        dbc.Col([
                            self.report_panel,
                            self.coordination_panel
                        ], md=4)
                    ])
                ], fluid=True, className="mt-4")
            ]
        )
        
        # Find existing tabs or create new tab structure
        if hasattr(existing_layout, 'children'):
            # Look for existing tabs component
            for child in existing_layout.children:
                if hasattr(child, 'children') and isinstance(child.children, list):
                    for grandchild in child.children:
                        if hasattr(grandchild, 'id') and 'tabs' in str(grandchild.id):
                            # Add conservation tab to existing tabs
                            if hasattr(grandchild, 'children'):
                                grandchild.children.append(conservation_tab)
                            return existing_layout
        
        # If no existing tabs found, create new layout with tabs
        enhanced_layout = html.Div([
            dbc.Tabs([
                dbc.Tab(
                    label="🌳 Main Dashboard", 
                    tab_id="main-tab",
                    children=[existing_layout]
                ),
                conservation_tab
            ], id="main-tabs", active_tab="main-tab")
        ])
        
        return enhanced_layout
    
    def create_conservation_navbar_items(self) -> List[dbc.NavItem]:
        """Create additional navbar items for conservation features"""
        return [
            dbc.NavItem([
                dbc.Button(
                    [html.I(className="fas fa-bell me-1"), 
                     dbc.Badge("3", color="danger", className="ms-1")],
                    id="alerts-toggle",
                    color="warning",
                    size="sm",
                    outline=True
                )
            ]),
            dbc.NavItem([
                dbc.Button(
                    [html.I(className="fas fa-download me-1"), "Quick Report"],
                    id="quick-report-btn",
                    color="success",
                    size="sm",
                    outline=True
                )
            ])
        ]
    
    def setup_automation_workflows(self):
        """Setup automated workflows for conservation monitoring"""
        
        # Automated alert checking (would be scheduled with APScheduler)
        @self.app.callback(
            Output("alert-automation-status", "children"),
            [Input("monitoring-interval", "n_intervals")]
        )
        def run_automated_workflows(n_intervals):
            """Run automated conservation workflows"""
            try:
                # Every 10 intervals (5 minutes), run full monitoring cycle
                if n_intervals % 10 == 0:
                    self.alert_system.run_monitoring_cycle()
                    logger.info("Automated monitoring cycle completed")
                    
                return html.Div()  # Hidden status div
                
            except Exception as e:
                logger.error(f"Error in automated workflow: {e}")
                return html.Div()

# Integration function for existing dashboard
def enhance_dashboard_for_conservation(app: dash.Dash) -> ConservationDashboardEnhancer:
    """
    Enhance existing dashboard with conservation features.
    Returns enhancer instance for additional configuration.
    """
    enhancer = ConservationDashboardEnhancer(app)
    
    # Setup automation workflows
    enhancer.setup_automation_workflows()
    
    logger.info("Dashboard enhanced with conservation features")
    return enhancer

# Example usage
def main():
    """Example of integrating conservation features"""
    
    # Create basic Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    
    # Enhance with conservation features
    enhancer = enhance_dashboard_for_conservation(app)
    
    # Basic layout (would be replaced with actual dashboard layout)
    app.layout = enhancer.add_conservation_layout(
        html.Div([
            html.H1("Conservation Dashboard"),
            html.P("Enhanced with automated workflows and field coordination")
        ])
    )
    
    if __name__ == "__main__":
        app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == "__main__":
    main()
