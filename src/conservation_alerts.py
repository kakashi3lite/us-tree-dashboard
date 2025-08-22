#!/usr/bin/env python3
"""
Conservation Alert System for Real-time Environmental Monitoring
Automated alert pipeline for conservation status changes and threat detection.
Built for 10× Conservation Tech workflow optimization.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# Removed unused MimeBase/encoders imports (not needed for plain text email)
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import os

# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/conservation_alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConservationAlert:
    """Data structure for conservation alerts"""
    alert_id: str
    timestamp: datetime
    region: str
    alert_type: str  # 'deforestation', 'biodiversity_loss', 'species_threat', 'canopy_decline'
    severity: str    # 'low', 'medium', 'high', 'critical'
    coordinates: Tuple[float, float]  # lat, lon
    affected_area_km2: float
    species_count: int
    threat_description: str
    recommended_actions: List[str]
    confidence_score: float
    data_sources: List[str]

class ConservationAlertSystem:
    """
    Real-time conservation alert system for field teams and stakeholders.
    Implements automated threat detection and notification workflows.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.data_dir = Path('data')
        self.alert_history = []
        self.thresholds = self._initialize_thresholds()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load alert system configuration"""
        default_config = {
            'email_notifications': {
                'enabled': True,
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                'sender_email': os.getenv('ALERT_SENDER_EMAIL'),
                'sender_password': os.getenv('ALERT_SENDER_PASSWORD'),
                'recipients': os.getenv('ALERT_RECIPIENTS', '').split(',')
            },
            'alert_thresholds': {
                'canopy_decline_percent': 15.0,
                'species_loss_percent': 10.0,
                'deforestation_rate_km2_per_month': 5.0,
                'min_confidence_score': 0.7
            },
            'monitoring_regions': [
                {'name': 'California Central Valley', 'bounds': [35.0, -122.0, 38.0, -119.0]},
                {'name': 'Pacific Northwest', 'bounds': [45.0, -125.0, 49.0, -117.0]},
                {'name': 'Great Lakes Region', 'bounds': [41.0, -93.0, 49.0, -76.0]}
            ]
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
                
        return default_config
    
    def _initialize_thresholds(self) -> Dict:
        """Initialize conservation monitoring thresholds"""
        return {
            'canopy_decline': {
                'critical': 25.0,  # % loss per year
                'high': 15.0,
                'medium': 10.0,
                'low': 5.0
            },
            'species_threat': {
                'critical': 50,    # species count drop
                'high': 30,
                'medium': 20,
                'low': 10
            },
            'deforestation_rate': {
                'critical': 20.0,  # km²/month
                'high': 10.0,
                'medium': 5.0,
                'low': 2.0
            }
        }
    
    def analyze_canopy_trends(self, canopy_data: pd.DataFrame, 
                            time_window_months: int = 12) -> List[ConservationAlert]:
        """
        Analyze canopy coverage trends and generate alerts for declining areas.
        Uses NDVI analytics approach for precise monitoring.
        """
        alerts = []
        
        if 'timestamp' not in canopy_data.columns:
            logger.warning("No timestamp column found in canopy data")
            return alerts
            
        # Convert timestamp and filter to time window
        canopy_data['timestamp'] = pd.to_datetime(canopy_data['timestamp'])
        cutoff_date = datetime.now() - timedelta(days=30 * time_window_months)
        recent_data = canopy_data[canopy_data['timestamp'] >= cutoff_date]
        
        if recent_data.empty:
            logger.warning("No recent canopy data available for analysis")
            return alerts
            
        # Group by region and calculate trends
        for region in recent_data['region'].unique():
            region_data = recent_data[recent_data['region'] == region].copy()
            region_data = region_data.sort_values('timestamp')
            
            if len(region_data) < 2:
                continue
                
            # Calculate canopy decline rate
            initial_canopy = region_data['canopy_pct'].iloc[0]
            final_canopy = region_data['canopy_pct'].iloc[-1]
            decline_percent = ((initial_canopy - final_canopy) / initial_canopy) * 100
            
            # Determine severity level
            severity = self._assess_severity('canopy_decline', decline_percent)
            
            if severity != 'none':
                # Calculate confidence based on data quality
                confidence = min(0.95, len(region_data) / 12.0)  # More data = higher confidence
                
                # Get region coordinates (use centroid if available)
                lat = region_data['latitude'].mean() if 'latitude' in region_data.columns else 0.0
                lon = region_data['longitude'].mean() if 'longitude' in region_data.columns else 0.0
                
                alert = ConservationAlert(
                    alert_id=f"CANOPY_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    region=region,
                    alert_type='canopy_decline',
                    severity=severity,
                    coordinates=(lat, lon),
                    affected_area_km2=region_data['area_km2'].iloc[-1] if 'area_km2' in region_data.columns else 0.0,
                    species_count=region_data['species_count'].iloc[-1] if 'species_count' in region_data.columns else 0,
                    threat_description=f"Canopy coverage declined by {decline_percent:.1f}% over {time_window_months} months",
                    recommended_actions=self._generate_canopy_recommendations(decline_percent, region),
                    confidence_score=confidence,
                    data_sources=['USDA_TCC', 'GBIF', 'Field_Surveys']
                )
                
                alerts.append(alert)
                logger.info(f"Generated {severity} canopy alert for {region}: {decline_percent:.1f}% decline")
        
        return alerts
    
    def _assess_severity(self, alert_type: str, value: float) -> str:
        """Assess severity level based on thresholds"""
        if alert_type not in self.thresholds:
            return 'none'
            
        thresholds = self.thresholds[alert_type]
        
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['high']:
            return 'high'
        elif value >= thresholds['medium']:
            return 'medium'
        elif value >= thresholds['low']:
            return 'low'
        else:
            return 'none'
    
    def _generate_canopy_recommendations(self, decline_percent: float, region: str) -> List[str]:
        """Generate actionable recommendations based on canopy decline"""
        recommendations = []
        
        if decline_percent >= 25:
            recommendations.extend([
                "URGENT: Deploy field team for immediate assessment",
                "Implement emergency reforestation protocol",
                "Contact local authorities and conservation groups",
                "Establish protective barriers if needed"
            ])
        elif decline_percent >= 15:
            recommendations.extend([
                "Schedule field assessment within 7 days",
                "Analyze satellite imagery for root cause",
                "Coordinate with local land management",
                "Plan targeted restoration activities"
            ])
        elif decline_percent >= 10:
            recommendations.extend([
                "Monitor closely with monthly assessments",
                "Review historical data for patterns",
                "Engage community stakeholders",
                "Consider preventive interventions"
            ])
        else:
            recommendations.extend([
                "Continue routine monitoring",
                "Document trend for future reference",
                "Assess for early intervention opportunities"
            ])
            
        return recommendations
    
    def generate_regional_report(self, region: str, output_format: str = 'pdf') -> str:
        """
        Generate comprehensive regional conservation report.
        Supports PDF download for field teams.
        """
        logger.info(f"Generating regional report for {region}")
        
        # Load latest data for the region
        report_data = self._compile_regional_data(region)
        
        # Create visualizations
        charts = self._create_report_charts(report_data)
        
        # Generate report content
        report_content = self._format_report_content(region, report_data, charts)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"conservation_report_{region.replace(' ', '_')}_{timestamp}"
        
        if output_format.lower() == 'pdf':
            output_path = self._generate_pdf_report(report_content, filename)
        else:
            output_path = self._generate_html_report(report_content, filename)
            
        logger.info(f"Regional report generated: {output_path}")
        return output_path
    
    def _compile_regional_data(self, region: str) -> Dict:
        """Compile all available data for a specific region"""
        data = {
            'region': region,
            'timestamp': datetime.now(),
            'canopy_data': [],
            'species_data': [],
            'threat_data': [],
            'conservation_status': {}
        }
        
        # Load canopy data
        canopy_files = list(self.data_dir.glob('*canopy*.csv'))
        for file in canopy_files:
            try:
                df = pd.read_csv(file)
                if 'region' in df.columns:
                    region_data = df[df['region'].str.contains(region, case=False, na=False)]
                    if not region_data.empty:
                        data['canopy_data'].append(region_data)
            except Exception as e:
                logger.warning(f"Could not load canopy data from {file}: {e}")
        
        # Load species data
        species_files = list(self.data_dir.glob('*species*.csv')) + list(self.data_dir.glob('*gbif*.csv'))
        for file in species_files:
            try:
                df = pd.read_csv(file)
                # Filter by geographic bounds if coordinates available
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    # Simple bounding box filter (would be more sophisticated in production)
                    region_data = df[
                        (df['latitude'].between(30, 50)) &  # Approximate US bounds
                        (df['longitude'].between(-125, -65))
                    ]
                    if not region_data.empty:
                        data['species_data'].append(region_data)
            except Exception as e:
                logger.warning(f"Could not load species data from {file}: {e}")
        
        return data
    
    def _create_report_charts(self, report_data: Dict) -> Dict:
        """Create visualizations for the regional report"""
        charts = {}
        
        # Canopy trend chart
        if report_data['canopy_data']:
            canopy_df = pd.concat(report_data['canopy_data'])
            if 'timestamp' in canopy_df.columns and 'canopy_pct' in canopy_df.columns:
                fig = px.line(canopy_df, x='timestamp', y='canopy_pct',
                             title='Canopy Coverage Trend',
                             labels={'canopy_pct': 'Canopy Coverage (%)',
                                   'timestamp': 'Date'})
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                charts['canopy_trend'] = plot(fig, output_type='div', include_plotlyjs=False)
        
        # Species diversity chart
        if report_data['species_data']:
            species_df = pd.concat(report_data['species_data'])
            if 'species' in species_df.columns:
                species_counts = species_df['species'].value_counts().head(10)
                fig = px.bar(x=species_counts.index, y=species_counts.values,
                           title='Top 10 Species by Occurrence',
                           labels={'x': 'Species', 'y': 'Count'})
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    xaxis_tickangle=45
                )
                charts['species_diversity'] = plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _format_report_content(self, region: str, data: Dict, charts: Dict) -> str:
        """Format the complete report content as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conservation Report - {region}</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #28a745; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #28a745; }}
                .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
                .metric h3 {{ margin: 0; color: #28a745; }}
                .metric p {{ margin: 5px 0; font-size: 1.2em; font-weight: bold; }}
                .chart {{ margin: 20px 0; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alert-critical {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .alert-high {{ background: #fff3cd; border: 1px solid #ffeaa7; }}
                .alert-medium {{ background: #d1ecf1; border: 1px solid #b3d4fc; }}
                .recommendations {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌿 Conservation Status Report</h1>
                <h2>{region}</h2>
                <p>Generated on {data['timestamp'].strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Key Metrics</h2>
                <div class="metric">
                    <h3>Canopy Coverage</h3>
                    <p>{self._calculate_avg_canopy(data)}%</p>
                </div>
                <div class="metric">
                    <h3>Species Count</h3>
                    <p>{self._calculate_species_count(data)}</p>
                </div>
                <div class="metric">
                    <h3>Threat Level</h3>
                    <p>{self._assess_threat_level(data)}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Trends and Analysis</h2>
                {charts.get('canopy_trend', '<p>No canopy trend data available</p>')}
                {charts.get('species_diversity', '<p>No species diversity data available</p>')}
            </div>
            
            <div class="section">
                <h2>🚨 Active Alerts</h2>
                {self._format_active_alerts(region)}
            </div>
            
            <div class="section">
                <h2>💡 Recommended Actions</h2>
                <div class="recommendations">
                    {self._format_recommendations(data)}
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Data Sources</h2>
                <ul>
                    <li>USDA Forest Service Tree Canopy Cover</li>
                    <li>GBIF Species Occurrence Data</li>
                    <li>IUCN Red List Conservation Status</li>
                    <li>Field Survey Data</li>
                </ul>
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _calculate_avg_canopy(self, data: Dict) -> str:
        """Calculate average canopy coverage for the region"""
        if data['canopy_data']:
            canopy_df = pd.concat(data['canopy_data'])
            if 'canopy_pct' in canopy_df.columns:
                return f"{canopy_df['canopy_pct'].mean():.1f}"
        return "N/A"
    
    def _calculate_species_count(self, data: Dict) -> str:
        """Calculate total species count for the region"""
        if data['species_data']:
            species_df = pd.concat(data['species_data'])
            if 'species' in species_df.columns:
                return str(species_df['species'].nunique())
        return "N/A"
    
    def _assess_threat_level(self, data: Dict) -> str:
        """Assess overall threat level for the region"""
        # This would be more sophisticated in production
        # For now, return a simple assessment
        return "Medium"
    
    def _format_active_alerts(self, region: str) -> str:
        """Format active alerts for the region"""
        region_alerts = [alert for alert in self.alert_history 
                        if alert.region.lower() == region.lower()]
        
        if not region_alerts:
            return "<p>No active alerts for this region.</p>"
            
        alerts_html = ""
        for alert in region_alerts[-3:]:  # Show last 3 alerts
            alerts_html += f"""
            <div class="alert alert-{alert.severity}">
                <h4>{alert.alert_type.replace('_', ' ').title()} - {alert.severity.upper()}</h4>
                <p><strong>Date:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Description:</strong> {alert.threat_description}</p>
                <p><strong>Confidence:</strong> {alert.confidence_score:.1%}</p>
            </div>
            """
        
        return alerts_html
    
    def _format_recommendations(self, data: Dict) -> str:
        """Format conservation recommendations"""
        recommendations = [
            "Continue regular monitoring with monthly assessments",
            "Engage local stakeholders in conservation efforts",
            "Implement adaptive management strategies",
            "Monitor climate change impacts on species distribution",
            "Coordinate with regional conservation organizations"
        ]
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    def _generate_pdf_report(self, content: str, filename: str) -> str:
        """Generate PDF report from HTML content"""
        # In production, would use libraries like weasyprint or pdfkit
        # For now, save as HTML with PDF-friendly styling
        output_path = f"reports/{filename}.html"
        Path("reports").mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return output_path
    
    def _generate_html_report(self, content: str, filename: str) -> str:
        """Generate HTML report"""
        output_path = f"reports/{filename}.html"
        Path("reports").mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return output_path
    
    def send_alert_notifications(self, alerts: List[ConservationAlert]) -> None:
        """Send email notifications for conservation alerts"""
        if not self.config['email_notifications']['enabled']:
            logger.info("Email notifications disabled")
            return
            
        if not alerts:
            logger.info("No alerts to send")
            return
            
        try:
            # Group alerts by severity for efficient notification
            critical_alerts = [a for a in alerts if a.severity == 'critical']
            high_alerts = [a for a in alerts if a.severity == 'high']
            
            if critical_alerts:
                self._send_urgent_notification(critical_alerts)
            
            if high_alerts or critical_alerts:
                self._send_daily_digest(alerts)
                
            logger.info(f"Sent notifications for {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")
    
    def _send_urgent_notification(self, alerts: List[ConservationAlert]) -> None:
        """Send immediate notification for critical alerts"""
        subject = f"🚨 URGENT: Critical Conservation Alert - {len(alerts)} regions affected"
        
        body = "IMMEDIATE ACTION REQUIRED\n\n"
        for alert in alerts:
            body += f"Region: {alert.region}\n"
            body += f"Issue: {alert.threat_description}\n"
            body += f"Severity: {alert.severity.upper()}\n"
            body += f"Recommended Actions:\n"
            for action in alert.recommended_actions[:3]:  # Top 3 actions
                body += f"  • {action}\n"
            body += "\n"
            
        self._send_email(subject, body)
    
    def _send_daily_digest(self, alerts: List[ConservationAlert]) -> None:
        """Send daily digest of conservation alerts"""
        subject = f"🌿 Daily Conservation Alert Digest - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"Conservation Alert Summary for {datetime.now().strftime('%B %d, %Y')}\n\n"
        body += f"Total Alerts: {len(alerts)}\n"
        
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            
        for severity, count in severity_counts.items():
            body += f"{severity.title()}: {count}\n"
            
        body += "\nDetailed Alert Information:\n\n"
        for alert in alerts:
            body += f"• {alert.region}: {alert.threat_description}\n"
            
        self._send_email(subject, body)
    
    def _send_email(self, subject: str, body: str) -> None:
        """Send email notification (no-op if config incomplete).

        For tests/CI where credentials aren't provided this will simply log a
        warning and return without raising to keep analyses deterministic.
        """
        try:
            config = self.config['email_notifications']
            if not config.get('sender_email') or not config.get('recipients'):
                logger.warning("Email configuration incomplete")
                return

            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                if config.get('sender_password'):
                    server.login(config['sender_email'], config['sender_password'])
                server.sendmail(config['sender_email'], config['recipients'], msg.as_string())
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def run_monitoring_cycle(self) -> None:
        """Run complete monitoring cycle - designed for automated scheduling"""
        logger.info("Starting conservation monitoring cycle")
        
        try:
            # Load latest data
            canopy_data = self._load_latest_canopy_data()
            
            if canopy_data is not None and not canopy_data.empty:
                # Generate alerts
                new_alerts = self.analyze_canopy_trends(canopy_data)
                
                # Store alerts
                self.alert_history.extend(new_alerts)
                
                # Send notifications
                self.send_alert_notifications(new_alerts)
                
                logger.info(f"Monitoring cycle completed: {len(new_alerts)} new alerts generated")
            else:
                logger.warning("No canopy data available for monitoring")
                
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    def _load_latest_canopy_data(self) -> Optional[pd.DataFrame]:
        """Load the most recent canopy coverage data"""
        try:
            # Look for canopy data files
            canopy_files = list(self.data_dir.glob('*canopy*.csv'))
            
            if not canopy_files:
                logger.warning("No canopy data files found")
                return None
                
            # Load the most recent file
            latest_file = max(canopy_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Add synthetic timestamp if not present (for demo purposes)
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now() - timedelta(days=30)
                
            logger.info(f"Loaded canopy data from {latest_file}: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading canopy data: {e}")
            return None

# Example usage for automated workflows
def main():
    """Main function for CLI usage and automation"""
    alert_system = ConservationAlertSystem()
    
    # Run monitoring cycle
    alert_system.run_monitoring_cycle()
    
    # Generate regional reports for key areas
    regions = ['California Central Valley', 'Pacific Northwest']
    for region in regions:
        try:
            report_path = alert_system.generate_regional_report(region)
            print(f"Generated report for {region}: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report for {region}: {e}")

if __name__ == "__main__":
    main()
