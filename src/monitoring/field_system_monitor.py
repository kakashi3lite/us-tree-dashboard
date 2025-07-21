#!/usr/bin/env python3
"""
Real-time system health monitoring for conservation field operations.
Optimized for Eli's workflow - automated alerts, performance tracking, and field team coordination.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os
import socket
import requests
import subprocess
import sys

# Configure logging for field operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """System health metric data structure"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    status: str  # 'healthy', 'warning', 'critical'
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemAlert:
    """System alert data structure"""
    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    assignee: Optional[str] = None

class FieldSystemMonitor:
    """
    Real-time system health monitoring for field operations.
    Designed for conservation workflows with automated alerts and recovery.
    """
    
    def __init__(self, config_path: str = "config/monitoring.json"):
        self.config = self._load_config(config_path)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: List[SystemAlert] = []
        self.notification_handlers: List[Callable] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Initialize database for persistent storage
        self.db_path = Path("data/system_health.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # Setup default notification handlers
        self._setup_notifications()
        
        logger.info("🌲 Field System Monitor initialized for conservation operations")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring_interval": 30,  # seconds
            "thresholds": {
                "cpu_usage": {"warning": 70, "critical": 90},
                "memory_usage": {"warning": 75, "critical": 90},
                "disk_usage": {"warning": 80, "critical": 95},
                "response_time": {"warning": 5.0, "critical": 10.0},
                "error_rate": {"warning": 0.05, "critical": 0.10}
            },
            "endpoints": [
                {"name": "dashboard", "url": "http://localhost:8050", "timeout": 10},
                {"name": "api_health", "url": "http://localhost:8050/api/v1/health", "timeout": 5}
            ],
            "notifications": {
                "email": {
                    "enabled": True,
                    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                    "username": os.getenv("EMAIL_USERNAME", ""),
                    "password": os.getenv("EMAIL_PASSWORD", ""),
                    "recipients": ["field-team@conservation.org", "alerts@conservation.org"]
                },
                "webhook": {
                    "enabled": False,
                    "url": os.getenv("WEBHOOK_URL", ""),
                    "headers": {"Content-Type": "application/json"}
                }
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            else:
                # Create default config file
                Path(config_path).parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default config at {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    threshold_warning REAL,
                    threshold_critical REAL,
                    metadata TEXT
                )
            """)
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    component TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TEXT,
                    assignee TEXT
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON health_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON system_alerts(timestamp)")
            
            conn.commit()
    
    def _setup_notifications(self):
        """Setup notification handlers for alerts"""
        if self.config["notifications"]["email"]["enabled"]:
            self.notification_handlers.append(self._send_email_notification)
        
        if self.config["notifications"]["webhook"]["enabled"]:
            self.notification_handlers.append(self._send_webhook_notification)
    
    def start_monitoring(self):
        """Start real-time system monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 Real-time monitoring started for field operations")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("⏹️ System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop - runs continuously"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                metrics.extend(app_metrics)
                
                # Store metrics and check thresholds
                for metric in metrics:
                    self._store_metric(metric)
                    self._check_thresholds(metric)
                
                # Check for system anomalies
                self._detect_anomalies()
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.config["monitoring_interval"])
    
    def _collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level metrics (CPU, memory, disk)"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU Usage
            if sys.platform.startswith('win'):
                # Windows-specific CPU monitoring
                import wmi
                c = wmi.WMI()
                cpu_usage = sum(float(cpu.LoadPercentage) for cpu in c.Win32_Processor()) / len(list(c.Win32_Processor()))
            else:
                # Unix-like systems
                cpu_usage = float(subprocess.check_output("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1", shell=True))
            
            metrics.append(HealthMetric(
                metric_name="cpu_usage",
                value=cpu_usage,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status(cpu_usage, "cpu_usage"),
                threshold_warning=self.config["thresholds"]["cpu_usage"]["warning"],
                threshold_critical=self.config["thresholds"]["cpu_usage"]["critical"]
            ))
            
        except Exception as e:
            logger.warning(f"Failed to collect CPU metrics: {e}")
        
        try:
            # Memory Usage
            if sys.platform.startswith('win'):
                import wmi
                c = wmi.WMI()
                total_memory = 0
                available_memory = 0
                for memory in c.Win32_OperatingSystem():
                    total_memory = int(memory.TotalVisibleMemorySize)
                    available_memory = int(memory.AvailablePhysicalMemory)
                
                memory_usage = ((total_memory - available_memory) / total_memory) * 100
            else:
                # Unix-like systems
                memory_info = subprocess.check_output("free | grep Mem", shell=True).decode().split()
                total = int(memory_info[1])
                used = int(memory_info[2])
                memory_usage = (used / total) * 100
            
            metrics.append(HealthMetric(
                metric_name="memory_usage",
                value=memory_usage,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status(memory_usage, "memory_usage"),
                threshold_warning=self.config["thresholds"]["memory_usage"]["warning"],
                threshold_critical=self.config["thresholds"]["memory_usage"]["critical"]
            ))
            
        except Exception as e:
            logger.warning(f"Failed to collect memory metrics: {e}")
        
        try:
            # Disk Usage
            if sys.platform.startswith('win'):
                import shutil
                disk_usage = shutil.disk_usage('C:')
                disk_percent = (disk_usage.used / disk_usage.total) * 100
            else:
                disk_info = subprocess.check_output("df / | tail -1", shell=True).decode().split()
                disk_percent = int(disk_info[4].replace('%', ''))
            
            metrics.append(HealthMetric(
                metric_name="disk_usage",
                value=disk_percent,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status(disk_percent, "disk_usage"),
                threshold_warning=self.config["thresholds"]["disk_usage"]["warning"],
                threshold_critical=self.config["thresholds"]["disk_usage"]["critical"]
            ))
            
        except Exception as e:
            logger.warning(f"Failed to collect disk metrics: {e}")
        
        return metrics
    
    def _collect_application_metrics(self) -> List[HealthMetric]:
        """Collect application-specific metrics (response times, error rates)"""
        metrics = []
        timestamp = datetime.now()
        
        # Test endpoint availability and response times
        for endpoint in self.config["endpoints"]:
            try:
                start_time = time.time()
                response = requests.get(
                    endpoint["url"], 
                    timeout=endpoint.get("timeout", 10)
                )
                response_time = time.time() - start_time
                
                # Response time metric
                metrics.append(HealthMetric(
                    metric_name=f"{endpoint['name']}_response_time",
                    value=response_time,
                    unit="seconds",
                    timestamp=timestamp,
                    status=self._get_status(response_time, "response_time"),
                    threshold_warning=self.config["thresholds"]["response_time"]["warning"],
                    threshold_critical=self.config["thresholds"]["response_time"]["critical"],
                    metadata={"endpoint": endpoint["url"], "status_code": response.status_code}
                ))
                
                # Availability metric
                is_available = 1.0 if response.status_code == 200 else 0.0
                metrics.append(HealthMetric(
                    metric_name=f"{endpoint['name']}_availability",
                    value=is_available,
                    unit="boolean",
                    timestamp=timestamp,
                    status="healthy" if is_available else "critical",
                    metadata={"endpoint": endpoint["url"], "status_code": response.status_code}
                ))
                
            except requests.exceptions.RequestException as e:
                # Endpoint unavailable
                metrics.append(HealthMetric(
                    metric_name=f"{endpoint['name']}_availability",
                    value=0.0,
                    unit="boolean",
                    timestamp=timestamp,
                    status="critical",
                    metadata={"endpoint": endpoint["url"], "error": str(e)}
                ))
                
                logger.warning(f"Endpoint {endpoint['name']} unavailable: {e}")
        
        return metrics
    
    def _get_status(self, value: float, metric_type: str) -> str:
        """Determine status based on thresholds"""
        thresholds = self.config["thresholds"].get(metric_type, {})
        
        critical_threshold = thresholds.get("critical")
        warning_threshold = thresholds.get("warning")
        
        if critical_threshold and value >= critical_threshold:
            return "critical"
        elif warning_threshold and value >= warning_threshold:
            return "warning"
        else:
            return "healthy"
    
    def _store_metric(self, metric: HealthMetric):
        """Store metric in database and memory"""
        # Store in memory for quick access
        self.metrics_history[metric.metric_name].append(metric)
        
        # Store in database for persistence
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO health_metrics 
                    (metric_name, value, unit, timestamp, status, threshold_warning, threshold_critical, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    metric.timestamp.isoformat(),
                    metric.status,
                    metric.threshold_warning,
                    metric.threshold_critical,
                    json.dumps(metric.metadata) if metric.metadata else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def _check_thresholds(self, metric: HealthMetric):
        """Check if metric exceeds thresholds and generate alerts"""
        if metric.status in ['warning', 'critical']:
            alert_id = f"{metric.metric_name}_{metric.status}_{int(time.time())}"
            
            # Check if similar alert already exists
            existing_alert = any(
                alert.component == metric.metric_name and 
                alert.severity == metric.status and 
                not alert.resolved
                for alert in self.active_alerts
            )
            
            if not existing_alert:
                alert = SystemAlert(
                    alert_id=alert_id,
                    severity=metric.status,
                    title=f"{metric.metric_name.replace('_', ' ').title()} {metric.status.title()}",
                    message=f"{metric.metric_name} is {metric.value:.2f} {metric.unit} "
                           f"(threshold: {metric.threshold_warning if metric.status == 'warning' else metric.threshold_critical})",
                    component=metric.metric_name,
                    timestamp=metric.timestamp
                )
                
                self.active_alerts.append(alert)
                self._store_alert(alert)
                self._send_notifications(alert)
                
                logger.warning(f"🚨 Alert generated: {alert.title}")
    
    def _detect_anomalies(self):
        """Detect system anomalies using statistical analysis"""
        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # Need enough data for analysis
                continue
            
            try:
                # Calculate recent statistics
                recent_values = [m.value for m in list(history)[-10:]]
                historical_values = [m.value for m in list(history)[:-10]] if len(history) > 10 else []
                
                if len(historical_values) < 5:
                    continue
                
                # Simple anomaly detection: significant deviation from historical mean
                import statistics
                historical_mean = statistics.mean(historical_values)
                historical_stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                recent_mean = statistics.mean(recent_values)
                
                # Check for anomaly (more than 2 standard deviations from historical mean)
                if historical_stdev > 0 and abs(recent_mean - historical_mean) > 2 * historical_stdev:
                    alert_id = f"anomaly_{metric_name}_{int(time.time())}"
                    
                    alert = SystemAlert(
                        alert_id=alert_id,
                        severity="warning",
                        title=f"Anomaly Detected: {metric_name.replace('_', ' ').title()}",
                        message=f"Recent {metric_name} values significantly different from historical pattern. "
                               f"Recent: {recent_mean:.2f}, Historical: {historical_mean:.2f} ±{historical_stdev:.2f}",
                        component=metric_name,
                        timestamp=datetime.now()
                    )
                    
                    # Check if similar anomaly alert already exists
                    existing_anomaly = any(
                        "Anomaly" in alert.title and 
                        alert.component == metric_name and 
                        not alert.resolved and
                        (datetime.now() - alert.timestamp).seconds < 300  # Within 5 minutes
                        for alert in self.active_alerts
                    )
                    
                    if not existing_anomaly:
                        self.active_alerts.append(alert)
                        self._store_alert(alert)
                        self._send_notifications(alert)
                        
                        logger.warning(f"📊 Anomaly detected: {alert.title}")
                        
            except Exception as e:
                logger.error(f"Error in anomaly detection for {metric_name}: {e}")
    
    def _store_alert(self, alert: SystemAlert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO system_alerts 
                    (alert_id, severity, title, message, component, timestamp, resolved, resolution_time, assignee)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.severity,
                    alert.title,
                    alert.message,
                    alert.component,
                    alert.timestamp.isoformat(),
                    alert.resolved,
                    alert.resolution_time.isoformat() if alert.resolution_time else None,
                    alert.assignee
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _send_notifications(self, alert: SystemAlert):
        """Send notifications for alerts"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
    
    def _send_email_notification(self, alert: SystemAlert):
        """Send email notification for alert"""
        config = self.config["notifications"]["email"]
        
        if not config.get("username") or not config.get("password"):
            logger.warning("Email credentials not configured")
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = config["username"]
            msg['Subject'] = f"🚨 Conservation System Alert: {alert.title}"
            
            # Create HTML email body
            body = f"""
            <html>
            <body>
                <h2>🌲 Conservation System Alert</h2>
                <p><strong>Severity:</strong> {alert.severity.upper()}</p>
                <p><strong>Component:</strong> {alert.component}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                
                <h3>Recommended Actions:</h3>
                <ul>
                    <li>Check dashboard connectivity</li>
                    <li>Verify field team access</li>
                    <li>Review system resources</li>
                    <li>Contact technical support if critical</li>
                </ul>
                
                <p><em>This is an automated alert from the Conservation Field Operations System.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(body, 'html'))
            
            # Send to all recipients
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            
            for recipient in config["recipients"]:
                msg['To'] = recipient
                server.send_message(msg)
                del msg['To']
            
            server.quit()
            logger.info(f"📧 Email alert sent to {len(config['recipients'])} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def _send_webhook_notification(self, alert: SystemAlert):
        """Send webhook notification for alert"""
        config = self.config["notifications"]["webhook"]
        
        try:
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp.isoformat(),
                "system": "conservation_dashboard"
            }
            
            response = requests.post(
                config["url"],
                json=payload,
                headers=config.get("headers", {}),
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"📡 Webhook alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and resolved alerts"""
        cutoff_date = datetime.now() - timedelta(days=7)  # Keep 7 days of data
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean old metrics
                cursor.execute(
                    "DELETE FROM health_metrics WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                
                # Clean old resolved alerts
                cursor.execute(
                    "DELETE FROM system_alerts WHERE resolved = TRUE AND resolution_time < ?",
                    (cutoff_date.isoformat(),)
                )
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status summary"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
            "critical_alerts": len([a for a in self.active_alerts if a.severity == "critical" and not a.resolved]),
            "warning_alerts": len([a for a in self.active_alerts if a.severity == "warning" and not a.resolved]),
            "metrics": {}
        }
        
        # Get latest metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                latest = history[-1]
                status["metrics"][metric_name] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "status": latest.status,
                    "timestamp": latest.timestamp.isoformat()
                }
                
                # Update overall status
                if latest.status == "critical":
                    status["overall_status"] = "critical"
                elif latest.status == "warning" and status["overall_status"] == "healthy":
                    status["overall_status"] = "warning"
        
        return status
    
    def resolve_alert(self, alert_id: str, assignee: Optional[str] = None):
        """Resolve an alert manually"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                alert.assignee = assignee
                
                self._store_alert(alert)
                logger.info(f"✅ Alert resolved: {alert.title}")
                return True
        
        return False
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        report_path = Path("reports/system_health_report.html")
        report_path.parent.mkdir(exist_ok=True)
        
        current_status = self.get_current_status()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conservation System Health Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #28a745; color: white; padding: 20px; text-align: center; }}
                .status-healthy {{ color: #28a745; }}
                .status-warning {{ color: #ffc107; }}
                .status-critical {{ color: #dc3545; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .alert {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
                .alert-warning {{ border-left-color: #ffc107; background: #fff3cd; }}
                .alert-critical {{ border-left-color: #dc3545; background: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌲 Conservation System Health Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h2 class="status-{current_status['overall_status']}">
                    Overall Status: {current_status['overall_status'].upper()}
                </h2>
            </div>
            
            <h2>Alert Summary</h2>
            <p>Active Alerts: {current_status['active_alerts']}</p>
            <p>Critical: {current_status['critical_alerts']}</p>
            <p>Warning: {current_status['warning_alerts']}</p>
            
            <h2>Current Metrics</h2>
        """
        
        for metric_name, metric_data in current_status["metrics"].items():
            html_content += f"""
            <div class="metric">
                <strong>{metric_name.replace('_', ' ').title()}</strong>
                <span class="status-{metric_data['status']}">
                    {metric_data['value']:.2f} {metric_data['unit']} ({metric_data['status']})
                </span>
            </div>
            """
        
        # Add active alerts
        active_alerts = [a for a in self.active_alerts if not a.resolved]
        if active_alerts:
            html_content += "<h2>Active Alerts</h2>"
            for alert in active_alerts:
                html_content += f"""
                <div class="alert alert-{alert.severity}">
                    <strong>{alert.title}</strong><br>
                    {alert.message}<br>
                    <em>Component: {alert.component} | Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</em>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📊 Health report generated: {report_path}")
        return str(report_path)

def main():
    """Main function for running system monitoring"""
    monitor = FieldSystemMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        logger.info("🌲 Conservation Field System Monitor is running...")
        logger.info("Press Ctrl+C to stop monitoring")
        
        # Keep the main thread alive
        while True:
            time.sleep(10)
            
            # Print status update every minute
            status = monitor.get_current_status()
            logger.info(f"Status: {status['overall_status']} | "
                       f"Active Alerts: {status['active_alerts']} | "
                       f"Metrics: {len(status['metrics'])}")
    
    except KeyboardInterrupt:
        logger.info("Shutting down system monitor...")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
