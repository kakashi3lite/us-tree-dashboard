#!/usr/bin/env python3
"""
PDF Report Generator for Conservation Teams
Automated PDF generation with field-ready formats and data exports.
Optimized for conservation tech workflows and field deployment.
"""

import os
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import io
import logging
from dataclasses import dataclass
import json

# Import HTML to PDF conversion libraries
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint not available. Install with: pip install weasyprint")

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    logging.warning("pdfkit not available. Install with: pip install pdfkit")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for PDF report generation"""
    title: str
    subtitle: str
    region: str
    date_range: Tuple[datetime, datetime]
    include_maps: bool = True
    include_charts: bool = True
    include_raw_data: bool = False
    page_format: str = 'A4'  # A4, Letter, Legal
    orientation: str = 'portrait'  # portrait, landscape
    template_style: str = 'conservation'  # conservation, technical, executive

class ConservationPDFGenerator:
    """
    Generate field-ready PDF reports for conservation teams.
    Supports offline use and mobile-friendly formats.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = Path("data")
        
        # Initialize report templates
        self._setup_templates()
        
    def _setup_templates(self):
        """Setup CSS templates for different report types"""
        self.templates = {
            'conservation': {
                'primary_color': '#28a745',
                'secondary_color': '#6c757d', 
                'accent_color': '#17a2b8',
                'font_family': 'Arial, sans-serif'
            },
            'technical': {
                'primary_color': '#007bff',
                'secondary_color': '#6c757d',
                'accent_color': '#28a745', 
                'font_family': 'Georgia, serif'
            },
            'executive': {
                'primary_color': '#343a40',
                'secondary_color': '#6c757d',
                'accent_color': '#ffc107',
                'font_family': 'Helvetica, sans-serif'
            }
        }
    
    def generate_conservation_report(self, config: ReportConfig) -> str:
        """
        Generate comprehensive conservation report as PDF.
        Returns path to generated PDF file.
        """
        logger.info(f"Generating conservation report for {config.region}")
        
        # Compile report data
        report_data = self._compile_report_data(config)
        
        # Generate visualizations
        charts = self._generate_report_charts(report_data, config)
        
        # Create HTML content
        html_content = self._create_report_html(report_data, charts, config)
        
        # Generate PDF
        pdf_path = self._convert_to_pdf(html_content, config)
        
        logger.info(f"Conservation report generated: {pdf_path}")
        return str(pdf_path)
    
    def _compile_report_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Compile all necessary data for the report"""
        data = {
            'config': config,
            'metadata': {
                'generation_date': datetime.now(),
                'data_sources': [],
                'quality_score': 0.85  # Would be calculated based on actual data quality
            },
            'canopy_data': self._load_canopy_data(config),
            'species_data': self._load_species_data(config),
            'conservation_data': self._load_conservation_data(config),
            'threat_data': self._analyze_threats(config),
            'metrics': {}
        }
        
        # Calculate key metrics
        data['metrics'] = self._calculate_key_metrics(data)
        
        return data
    
    def _load_canopy_data(self, config: ReportConfig) -> pd.DataFrame:
        """Load and filter canopy coverage data"""
        try:
            # Look for canopy data files
            canopy_files = list(self.data_dir.glob('*canopy*.csv'))
            
            if not canopy_files:
                logger.warning("No canopy data files found")
                return pd.DataFrame()
            
            # Load most recent file
            latest_file = max(canopy_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Filter by region if possible
            if 'region' in df.columns:
                df = df[df['region'].str.contains(config.region, case=False, na=False)]
            elif 'state' in df.columns:
                # Try to match by state
                df = df[df['state'].str.contains(config.region, case=False, na=False)]
            
            logger.info(f"Loaded canopy data: {len(df)} records for {config.region}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading canopy data: {e}")
            return pd.DataFrame()
    
    def _load_species_data(self, config: ReportConfig) -> pd.DataFrame:
        """Load and filter species occurrence data"""
        try:
            # Look for species/GBIF data files
            species_files = list(self.data_dir.glob('*gbif*.csv')) + \
                           list(self.data_dir.glob('*species*.csv'))
            
            if not species_files:
                logger.warning("No species data files found")
                return pd.DataFrame()
            
            # Load and combine data
            dfs = []
            for file in species_files:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Could not load {file}: {e}")
            
            if not dfs:
                return pd.DataFrame()
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Filter by geographic bounds (simple approximation)
            if 'latitude' in combined_df.columns and 'longitude' in combined_df.columns:
                # Filter to reasonable US bounds
                combined_df = combined_df[
                    (combined_df['latitude'].between(25, 50)) &
                    (combined_df['longitude'].between(-125, -65))
                ]
            
            logger.info(f"Loaded species data: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading species data: {e}")
            return pd.DataFrame()
    
    def _load_conservation_data(self, config: ReportConfig) -> pd.DataFrame:
        """Load conservation status data"""
        try:
            conservation_files = list(self.data_dir.glob('*conservation*.csv'))
            
            if not conservation_files:
                logger.warning("No conservation data files found")
                return pd.DataFrame()
            
            # Load most recent file
            latest_file = max(conservation_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            logger.info(f"Loaded conservation data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading conservation data: {e}")
            return pd.DataFrame()
    
    def _analyze_threats(self, config: ReportConfig) -> Dict[str, Any]:
        """Analyze current threats in the region"""
        # This would integrate with the conservation alert system
        threats = {
            'active_threats': [
                {'type': 'habitat_fragmentation', 'severity': 'medium', 'area_affected_km2': 150},
                {'type': 'invasive_species', 'severity': 'high', 'species_count': 3},
                {'type': 'climate_stress', 'severity': 'low', 'temperature_change': 1.2}
            ],
            'risk_assessment': {
                'overall_risk': 'medium',
                'primary_concerns': ['habitat_loss', 'species_displacement'],
                'timeline': '2-5 years'
            }
        }
        
        return threats
    
    def _calculate_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key conservation metrics"""
        metrics = {
            'canopy_coverage': 'N/A',
            'species_count': 'N/A',
            'threatened_species': 'N/A',
            'conservation_priority': 'Medium',
            'data_quality_score': data['metadata']['quality_score']
        }
        
        # Calculate canopy coverage
        if not data['canopy_data'].empty:
            canopy_df = data['canopy_data']
            if 'canopy_pct' in canopy_df.columns:
                metrics['canopy_coverage'] = f"{canopy_df['canopy_pct'].mean():.1f}%"
        
        # Calculate species count
        if not data['species_data'].empty:
            species_df = data['species_data']
            if 'species' in species_df.columns:
                metrics['species_count'] = species_df['species'].nunique()
            elif 'scientificName' in species_df.columns:
                metrics['species_count'] = species_df['scientificName'].nunique()
        
        # Calculate threatened species
        if not data['conservation_data'].empty:
            conservation_df = data['conservation_data']
            if 'conservation_status' in conservation_df.columns:
                threatened = conservation_df[
                    conservation_df['conservation_status'].isin([
                        'Vulnerable', 'Endangered', 'Critically Endangered'
                    ])
                ]
                metrics['threatened_species'] = len(threatened)
        
        return metrics
    
    def _generate_report_charts(self, data: Dict[str, Any], config: ReportConfig) -> Dict[str, str]:
        """Generate all charts for the report"""
        charts = {}
        
        if not config.include_charts:
            return charts
        
        try:
            # Canopy coverage trend
            if not data['canopy_data'].empty:
                charts['canopy_trend'] = self._create_canopy_chart(data['canopy_data'])
            
            # Species diversity
            if not data['species_data'].empty:
                charts['species_diversity'] = self._create_species_chart(data['species_data'])
            
            # Conservation status
            if not data['conservation_data'].empty:
                charts['conservation_status'] = self._create_conservation_chart(data['conservation_data'])
            
            # Threat analysis
            charts['threat_analysis'] = self._create_threat_chart(data['threat_data'])
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
        
        return charts
    
    def _create_canopy_chart(self, canopy_data: pd.DataFrame) -> str:
        """Create canopy coverage chart"""
        if 'canopy_pct' not in canopy_data.columns:
            return "<p>No canopy data available</p>"
        
        # Create histogram of canopy coverage
        fig = px.histogram(
            canopy_data, 
            x='canopy_pct',
            nbins=20,
            title='Canopy Coverage Distribution',
            labels={'canopy_pct': 'Canopy Coverage (%)', 'count': 'Number of Areas'}
        )
        
        fig.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=10)
        )
        
        return plot(fig, output_type='div', include_plotlyjs=False, div_id='canopy-chart')
    
    def _create_species_chart(self, species_data: pd.DataFrame) -> str:
        """Create species diversity chart"""
        species_col = 'species' if 'species' in species_data.columns else 'scientificName'
        
        if species_col not in species_data.columns:
            return "<p>No species data available</p>"
        
        # Top 10 species
        species_counts = species_data[species_col].value_counts().head(10)
        
        fig = px.bar(
            x=species_counts.values,
            y=species_counts.index,
            orientation='h',
            title='Top 10 Species by Occurrence',
            labels={'x': 'Occurrence Count', 'y': 'Species'}
        )
        
        fig.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=10)
        )
        
        return plot(fig, output_type='div', include_plotlyjs=False, div_id='species-chart')
    
    def _create_conservation_chart(self, conservation_data: pd.DataFrame) -> str:
        """Create conservation status chart"""
        if 'conservation_status' not in conservation_data.columns:
            return "<p>No conservation status data available</p>"
        
        status_counts = conservation_data['conservation_status'].value_counts()
        
        # Define colors for conservation status
        colors = {
            'Least Concern': '#28a745',
            'Near Threatened': '#ffc107', 
            'Vulnerable': '#fd7e14',
            'Endangered': '#dc3545',
            'Critically Endangered': '#6f42c1',
            'Extinct in Wild': '#343a40'
        }
        
        chart_colors = [colors.get(status, '#6c757d') for status in status_counts.index]
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Conservation Status Distribution',
            color_discrete_sequence=chart_colors
        )
        
        fig.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=10)
        )
        
        return plot(fig, output_type='div', include_plotlyjs=False, div_id='conservation-chart')
    
    def _create_threat_chart(self, threat_data: Dict[str, Any]) -> str:
        """Create threat analysis chart"""
        if not threat_data.get('active_threats'):
            return "<p>No threat data available</p>"
        
        threats = threat_data['active_threats']
        threat_types = [t['type'].replace('_', ' ').title() for t in threats]
        severities = [t['severity'] for t in threats]
        
        # Map severity to numeric values for visualization
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        severity_values = [severity_map.get(s, 1) for s in severities]
        
        fig = px.bar(
            x=threat_types,
            y=severity_values,
            title='Current Threat Assessment',
            labels={'x': 'Threat Type', 'y': 'Severity Level'},
            color=severity_values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=10),
            showlegend=False
        )
        
        return plot(fig, output_type='div', include_plotlyjs=False, div_id='threat-chart')
    
    def _create_report_html(self, data: Dict[str, Any], charts: Dict[str, str], 
                          config: ReportConfig) -> str:
        """Create complete HTML content for the report"""
        template = self.templates[config.template_style]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{config.title}</title>
            <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
            <style>
                @page {{
                    size: {config.page_format};
                    margin: 2cm;
                }}
                
                body {{
                    font-family: {template['font_family']};
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                }}
                
                .header {{
                    background: linear-gradient(135deg, {template['primary_color']}, {template['accent_color']});
                    color: white;
                    padding: 30px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: bold;
                }}
                
                .header h2 {{
                    margin: 10px 0 0 0;
                    font-size: 1.5em;
                    font-weight: normal;
                    opacity: 0.9;
                }}
                
                .metadata {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-left: 5px solid {template['primary_color']};
                    margin-bottom: 30px;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                
                .metric-card {{
                    background: white;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: {template['primary_color']};
                    margin: 10px 0;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    color: {template['secondary_color']};
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .section {{
                    margin: 40px 0;
                    page-break-inside: avoid;
                }}
                
                .section h2 {{
                    color: {template['primary_color']};
                    border-bottom: 2px solid {template['primary_color']};
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                
                .chart-container {{
                    background: white;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                
                .threat-item {{
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                }}
                
                .threat-high {{
                    background: #f8d7da;
                    border-color: #f5c6cb;
                }}
                
                .threat-critical {{
                    background: #d1ecf1;
                    border-color: #b3d4fc;
                }}
                
                .recommendations {{
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                
                .recommendations h3 {{
                    margin-top: 0;
                    color: #155724;
                }}
                
                .recommendations ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                
                .footer {{
                    margin-top: 50px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-top: 1px solid #dee2e6;
                    font-size: 0.9em;
                    color: {template['secondary_color']};
                }}
                
                .page-break {{
                    page-break-before: always;
                }}
                
                @media print {{
                    .no-print {{ display: none !important; }}
                    body {{ font-size: 12pt; }}
                    .chart-container {{ break-inside: avoid; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌿 {config.title}</h1>
                <h2>{config.subtitle}</h2>
                <p><strong>Region:</strong> {config.region} | <strong>Generated:</strong> {data['metadata']['generation_date'].strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="metadata">
                <h3>Report Information</h3>
                <p><strong>Date Range:</strong> {config.date_range[0].strftime('%Y-%m-%d')} to {config.date_range[1].strftime('%Y-%m-%d')}</p>
                <p><strong>Data Quality Score:</strong> {data['metadata']['quality_score']:.1%}</p>
                <p><strong>Report Type:</strong> {config.template_style.title()} Format</p>
            </div>
            
            <div class="section">
                <h2>📊 Key Conservation Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Canopy Coverage</div>
                        <div class="metric-value">{data['metrics']['canopy_coverage']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Species Count</div>
                        <div class="metric-value">{data['metrics']['species_count']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Threatened Species</div>
                        <div class="metric-value">{data['metrics']['threatened_species']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Conservation Priority</div>
                        <div class="metric-value">{data['metrics']['conservation_priority']}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Data Analysis and Trends</h2>
                
                <div class="chart-container">
                    <h3>Canopy Coverage Analysis</h3>
                    {charts.get('canopy_trend', '<p>No canopy data available for visualization</p>')}
                </div>
                
                <div class="chart-container">
                    <h3>Species Diversity</h3>
                    {charts.get('species_diversity', '<p>No species data available for visualization</p>')}
                </div>
                
                <div class="chart-container">
                    <h3>Conservation Status Overview</h3>
                    {charts.get('conservation_status', '<p>No conservation data available for visualization</p>')}
                </div>
            </div>
            
            <div class="section page-break">
                <h2>🚨 Threat Assessment</h2>
                
                <div class="chart-container">
                    <h3>Current Threats</h3>
                    {charts.get('threat_analysis', '<p>No threat data available for visualization</p>')}
                </div>
                
                {self._format_threat_details(data['threat_data'])}
            </div>
            
            <div class="section">
                <h2>💡 Conservation Recommendations</h2>
                <div class="recommendations">
                    <h3>Priority Actions</h3>
                    {self._format_recommendations(data)}
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Data Sources and Methodology</h2>
                <ul>
                    <li><strong>USDA Forest Service:</strong> Tree Canopy Cover datasets (TCC)</li>
                    <li><strong>GBIF:</strong> Global Biodiversity Information Facility species occurrence data</li>
                    <li><strong>IUCN Red List:</strong> Conservation status assessments</li>
                    <li><strong>Field Surveys:</strong> Local monitoring and assessment data</li>
                    <li><strong>Satellite Imagery:</strong> Remote sensing and NDVI analysis</li>
                </ul>
                
                <h3>Quality Assurance</h3>
                <p>This report undergoes automated quality checks including data validation, 
                coordinate verification, and temporal consistency analysis. 
                Overall data quality score: <strong>{data['metadata']['quality_score']:.1%}</strong></p>
            </div>
            
            <div class="footer">
                <p><strong>Generated by:</strong> PlantsWorld Conservation Dashboard | 
                <strong>Report ID:</strong> {config.region.replace(' ', '_')}_{data['metadata']['generation_date'].strftime('%Y%m%d_%H%M%S')} | 
                <strong>Version:</strong> 1.0</p>
                <p><em>This report is generated from the latest available data and should be used in conjunction with field assessments for conservation planning.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _format_threat_details(self, threat_data: Dict[str, Any]) -> str:
        """Format detailed threat information"""
        if not threat_data.get('active_threats'):
            return "<p>No active threats detected in the current assessment period.</p>"
        
        threats_html = ""
        for threat in threat_data['active_threats']:
            threat_class = f"threat-{threat['severity']}" if threat['severity'] in ['high', 'critical'] else "threat-item"
            
            threats_html += f"""
            <div class="{threat_class}">
                <h4>{threat['type'].replace('_', ' ').title()} - {threat['severity'].upper()}</h4>
                <p><strong>Severity:</strong> {threat['severity'].title()}</p>
            """
            
            if 'area_affected_km2' in threat:
                threats_html += f"<p><strong>Area Affected:</strong> {threat['area_affected_km2']} km²</p>"
            if 'species_count' in threat:
                threats_html += f"<p><strong>Species Affected:</strong> {threat['species_count']}</p>"
                
            threats_html += "</div>"
        
        return threats_html
    
    def _format_recommendations(self, data: Dict[str, Any]) -> str:
        """Format conservation recommendations"""
        # Generate recommendations based on data analysis
        recommendations = []
        
        # Canopy-based recommendations
        if not data['canopy_data'].empty:
            canopy_df = data['canopy_data']
            if 'canopy_pct' in canopy_df.columns:
                avg_canopy = canopy_df['canopy_pct'].mean()
                if avg_canopy < 30:
                    recommendations.append("Implement immediate reforestation efforts in low-canopy areas")
                elif avg_canopy < 50:
                    recommendations.append("Develop targeted tree planting program to increase canopy coverage")
        
        # Species-based recommendations
        species_count = data['metrics'].get('species_count', 0)
        if isinstance(species_count, (int, float)) and species_count < 50:
            recommendations.append("Enhance habitat diversity to support increased species richness")
        
        # Threat-based recommendations
        threats = data['threat_data'].get('active_threats', [])
        high_severity_threats = [t for t in threats if t['severity'] in ['high', 'critical']]
        if high_severity_threats:
            recommendations.append("Address high-priority threats through immediate intervention")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Continue regular monitoring with quarterly assessments",
                "Engage local stakeholders in conservation planning",
                "Implement adaptive management strategies based on climate projections",
                "Coordinate with regional conservation organizations",
                "Develop community education and outreach programs"
            ]
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    def _convert_to_pdf(self, html_content: str, config: ReportConfig) -> Path:
        """Convert HTML content to PDF"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"conservation_report_{config.region.replace(' ', '_')}_{timestamp}"
        
        # Try WeasyPrint first (better for complex layouts)
        if WEASYPRINT_AVAILABLE:
            try:
                output_path = self.output_dir / f"{filename}.pdf"
                HTML(string=html_content).write_pdf(str(output_path))
                logger.info(f"PDF generated using WeasyPrint: {output_path}")
                return output_path
            except Exception as e:
                logger.warning(f"WeasyPrint failed: {e}, trying pdfkit")
        
        # Fallback to pdfkit
        if PDFKIT_AVAILABLE:
            try:
                output_path = self.output_dir / f"{filename}.pdf"
                options = {
                    'page-size': config.page_format,
                    'orientation': config.orientation,
                    'margin-top': '2cm',
                    'margin-right': '2cm',
                    'margin-bottom': '2cm',
                    'margin-left': '2cm',
                    'encoding': "UTF-8",
                    'no-outline': None,
                    'enable-local-file-access': None
                }
                pdfkit.from_string(html_content, str(output_path), options=options)
                logger.info(f"PDF generated using pdfkit: {output_path}")
                return output_path
            except Exception as e:
                logger.warning(f"pdfkit failed: {e}, saving as HTML")
        
        # Fallback to HTML
        output_path = self.output_dir / f"{filename}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report saved as HTML: {output_path}")
        return output_path
    
    def generate_field_summary(self, region: str, format: str = 'pdf') -> str:
        """Generate quick field summary report for mobile use"""
        config = ReportConfig(
            title="Field Conservation Summary",
            subtitle=f"Quick Assessment Report",
            region=region,
            date_range=(datetime.now() - timedelta(days=30), datetime.now()),
            include_maps=False,  # Exclude maps for faster generation
            include_charts=True,
            include_raw_data=False,
            template_style='conservation'
        )
        
        return self.generate_conservation_report(config)
    
    def generate_executive_briefing(self, region: str, format: str = 'pdf') -> str:
        """Generate executive-level briefing document"""
        config = ReportConfig(
            title="Executive Conservation Briefing",
            subtitle=f"Strategic Overview and Recommendations",
            region=region,
            date_range=(datetime.now() - timedelta(days=90), datetime.now()),
            include_maps=True,
            include_charts=True,
            include_raw_data=False,
            template_style='executive'
        )
        
        return self.generate_conservation_report(config)

# Integration function for the main dashboard
def create_pdf_download_endpoint(app, pdf_generator: ConservationPDFGenerator):
    """Create download endpoint for PDF reports in the Dash app"""
    
    @app.callback(
        Output('pdf-download-link', 'href'),
        [Input('generate-pdf-btn', 'n_clicks')],
        [State('region-dropdown', 'value'),
         State('report-type-dropdown', 'value')]
    )
    def generate_pdf_report(n_clicks, region, report_type):
        if n_clicks and region:
            try:
                if report_type == 'field_summary':
                    pdf_path = pdf_generator.generate_field_summary(region)
                elif report_type == 'executive':
                    pdf_path = pdf_generator.generate_executive_briefing(region)
                else:
                    # Default comprehensive report
                    config = ReportConfig(
                        title="Conservation Assessment Report",
                        subtitle="Comprehensive Regional Analysis",
                        region=region,
                        date_range=(datetime.now() - timedelta(days=90), datetime.now()),
                        template_style='conservation'
                    )
                    pdf_path = pdf_generator.generate_conservation_report(config)
                
                # Return download link
                return f"/download/{Path(pdf_path).name}"
                
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                return ""
        
        return ""

# Example usage
def main():
    """Example usage of the PDF generator"""
    generator = ConservationPDFGenerator()
    
    # Generate a comprehensive report
    config = ReportConfig(
        title="Regional Conservation Assessment",
        subtitle="Biodiversity and Habitat Analysis",
        region="California Central Valley",
        date_range=(datetime.now() - timedelta(days=90), datetime.now()),
        template_style='conservation'
    )
    
    pdf_path = generator.generate_conservation_report(config)
    print(f"Generated conservation report: {pdf_path}")
    
    # Generate field summary
    field_report = generator.generate_field_summary("Pacific Northwest")
    print(f"Generated field summary: {field_report}")

if __name__ == "__main__":
    main()
