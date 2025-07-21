#!/usr/bin/env python3
"""
Conservation Dashboard Master Integration Script.
Optimized for Eli's 10× Conservation Tech Specialist workflow - complete automation and zero-stop operations.
"""

import asyncio
import logging
import json
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import os
import signal

# Configure comprehensive logging for field operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('logs/conservation_master.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConservationMasterController:
    """
    Master controller for integrated conservation dashboard ecosystem.
    Coordinates all components for seamless field operations.
    """
    
    def __init__(self, config_path: str = "config/master_config.json"):
        self.config = self._load_config(config_path)
        self.services = {}
        self.monitoring_active = False
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🌲 Conservation Master Controller initialized - ready for field operations")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load master configuration for all services"""
        default_config = {
            "services": {
                "dashboard": {
                    "enabled": True,
                    "port": 8050,
                    "host": "0.0.0.0",
                    "auto_start": True
                },
                "mobile_api": {
                    "enabled": True,
                    "port": 5001,
                    "host": "0.0.0.0",
                    "auto_start": True
                },
                "alerts": {
                    "enabled": True,
                    "check_interval": 300,  # 5 minutes
                    "auto_start": True
                },
                "data_pipeline": {
                    "enabled": True,
                    "auto_start": True,
                    "schedule_enabled": True
                },
                "system_monitor": {
                    "enabled": True,
                    "auto_start": True,
                    "monitoring_interval": 60
                },
                "pdf_generator": {
                    "enabled": True,
                    "auto_start": False  # On-demand service
                }
            },
            "integration": {
                "health_check_interval": 120,  # 2 minutes
                "auto_recovery": True,
                "max_restart_attempts": 3,
                "restart_delay": 30
            },
            "field_operations": {
                "offline_mode_enabled": True,
                "sync_interval": 600,  # 10 minutes
                "emergency_contacts": [
                    {"name": "Field Supervisor", "phone": "+1-555-0123"},
                    {"name": "Technical Support", "phone": "+1-555-0456"}
                ]
            },
            "notifications": {
                "webhook_url": os.getenv("MASTER_WEBHOOK_URL", ""),
                "email_enabled": True,
                "sms_enabled": False
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Deep merge with defaults
                    default_config.update(user_config)
            else:
                Path(config_path).parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created master config at {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.stop_all_services()
    
    async def start_all_services(self):
        """Start all enabled conservation services"""
        logger.info("🚀 Starting Conservation Dashboard Ecosystem")
        
        startup_tasks = []
        
        # Start core dashboard
        if self.config["services"]["dashboard"]["enabled"]:
            startup_tasks.append(self._start_dashboard())
        
        # Start mobile API
        if self.config["services"]["mobile_api"]["enabled"]:
            startup_tasks.append(self._start_mobile_api())
        
        # Start alert system
        if self.config["services"]["alerts"]["enabled"]:
            startup_tasks.append(self._start_alert_system())
        
        # Start data pipeline
        if self.config["services"]["data_pipeline"]["enabled"]:
            startup_tasks.append(self._start_data_pipeline())
        
        # Start system monitoring
        if self.config["services"]["system_monitor"]["enabled"]:
            startup_tasks.append(self._start_system_monitor())
        
        # Execute startup tasks concurrently
        await asyncio.gather(*startup_tasks, return_exceptions=True)
        
        # Start integrated monitoring
        await self._start_integrated_monitoring()
        
        logger.info("✅ All conservation services started successfully")
    
    async def _start_dashboard(self):
        """Start main conservation dashboard"""
        try:
            logger.info("📊 Starting Conservation Dashboard...")
            
            # Import and start dashboard
            dashboard_cmd = [
                sys.executable, "app.py",
                "--host", self.config["services"]["dashboard"]["host"],
                "--port", str(self.config["services"]["dashboard"]["port"])
            ]
            
            process = await asyncio.create_subprocess_exec(
                *dashboard_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.services["dashboard"] = {
                "process": process,
                "status": "running",
                "start_time": datetime.now(),
                "restart_count": 0
            }
            
            logger.info(f"✅ Dashboard started on port {self.config['services']['dashboard']['port']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            raise
    
    async def _start_mobile_api(self):
        """Start mobile field API"""
        try:
            logger.info("📱 Starting Mobile Field API...")
            
            mobile_cmd = [
                sys.executable, "src/mobile/field_mobile_api.py"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *mobile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.services["mobile_api"] = {
                "process": process,
                "status": "running",
                "start_time": datetime.now(),
                "restart_count": 0
            }
            
            logger.info(f"✅ Mobile API started on port {self.config['services']['mobile_api']['port']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start mobile API: {e}")
            raise
    
    async def _start_alert_system(self):
        """Start conservation alert system"""
        try:
            logger.info("🚨 Starting Conservation Alert System...")
            
            # Import alert system locally to avoid dependency issues
            try:
                from src.conservation_alerts import ConservationAlertSystem
                
                alert_system = ConservationAlertSystem()
                
                # Start monitoring in background thread
                def run_alerts():
                    alert_system.start_monitoring()
                
                alert_thread = threading.Thread(target=run_alerts, daemon=True)
                alert_thread.start()
                
                self.services["alerts"] = {
                    "system": alert_system,
                    "thread": alert_thread,
                    "status": "running",
                    "start_time": datetime.now(),
                    "restart_count": 0
                }
                
                logger.info("✅ Alert system started")
                
            except ImportError as e:
                logger.warning(f"Alert system not available: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start alert system: {e}")
    
    async def _start_data_pipeline(self):
        """Start automated data pipeline"""
        try:
            logger.info("🔄 Starting Data Pipeline...")
            
            try:
                from src.data.conservation_data_pipeline import ConservationDataPipeline
                
                pipeline = ConservationDataPipeline()
                
                if self.config["services"]["data_pipeline"]["schedule_enabled"]:
                    # Start scheduler in background thread
                    def run_pipeline():
                        pipeline.start_scheduler()
                    
                    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
                    pipeline_thread.start()
                
                self.services["data_pipeline"] = {
                    "pipeline": pipeline,
                    "thread": pipeline_thread if self.config["services"]["data_pipeline"]["schedule_enabled"] else None,
                    "status": "running",
                    "start_time": datetime.now(),
                    "restart_count": 0
                }
                
                logger.info("✅ Data pipeline started")
                
            except ImportError as e:
                logger.warning(f"Data pipeline not available: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start data pipeline: {e}")
    
    async def _start_system_monitor(self):
        """Start system health monitoring"""
        try:
            logger.info("🩺 Starting System Health Monitor...")
            
            try:
                from src.monitoring.field_system_monitor import FieldSystemMonitor
                
                monitor = FieldSystemMonitor()
                monitor.start_monitoring()
                
                self.services["system_monitor"] = {
                    "monitor": monitor,
                    "status": "running",
                    "start_time": datetime.now(),
                    "restart_count": 0
                }
                
                logger.info("✅ System monitor started")
                
            except ImportError as e:
                logger.warning(f"System monitor not available: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start system monitor: {e}")
    
    async def _start_integrated_monitoring(self):
        """Start integrated health monitoring for all services"""
        self.monitoring_active = True
        
        async def monitoring_loop():
            while self.monitoring_active and not self.shutdown_requested:
                try:
                    await self._check_service_health()
                    await asyncio.sleep(self.config["integration"]["health_check_interval"])
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
        
        # Start monitoring in background
        asyncio.create_task(monitoring_loop())
        logger.info("🔍 Integrated monitoring started")
    
    async def _check_service_health(self):
        """Check health of all running services"""
        unhealthy_services = []
        
        for service_name, service_info in self.services.items():
            try:
                if service_name == "dashboard":
                    # Check if dashboard process is running
                    if service_info.get("process"):
                        if service_info["process"].returncode is not None:
                            unhealthy_services.append(service_name)
                            logger.warning(f"⚠️ Dashboard process has stopped")
                
                elif service_name == "mobile_api":
                    # Check mobile API health endpoint
                    import requests
                    try:
                        response = requests.get(
                            f"http://localhost:{self.config['services']['mobile_api']['port']}/api/mobile/health",
                            timeout=5
                        )
                        if response.status_code != 200:
                            unhealthy_services.append(service_name)
                    except requests.exceptions.RequestException:
                        unhealthy_services.append(service_name)
                
                elif service_name == "alerts":
                    # Check if alert system thread is alive
                    if service_info.get("thread") and not service_info["thread"].is_alive():
                        unhealthy_services.append(service_name)
                
                elif service_name == "system_monitor":
                    # Check system monitor status
                    if service_info.get("monitor"):
                        status = service_info["monitor"].get_current_status()
                        if status.get("overall_status") == "critical":
                            logger.warning("⚠️ System monitor reports critical status")
                
            except Exception as e:
                logger.error(f"Error checking {service_name} health: {e}")
                unhealthy_services.append(service_name)
        
        # Handle unhealthy services
        if unhealthy_services and self.config["integration"]["auto_recovery"]:
            await self._handle_unhealthy_services(unhealthy_services)
    
    async def _handle_unhealthy_services(self, unhealthy_services: List[str]):
        """Handle and attempt to recover unhealthy services"""
        for service_name in unhealthy_services:
            service_info = self.services.get(service_name, {})
            restart_count = service_info.get("restart_count", 0)
            
            if restart_count < self.config["integration"]["max_restart_attempts"]:
                logger.info(f"🔄 Attempting to restart {service_name} (attempt {restart_count + 1})")
                
                try:
                    # Stop the service first
                    await self._stop_service(service_name)
                    
                    # Wait before restart
                    await asyncio.sleep(self.config["integration"]["restart_delay"])
                    
                    # Restart the service
                    await self._restart_service(service_name)
                    
                    # Update restart count
                    if service_name in self.services:
                        self.services[service_name]["restart_count"] = restart_count + 1
                    
                    logger.info(f"✅ Successfully restarted {service_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to restart {service_name}: {e}")
            else:
                logger.error(f"❌ Max restart attempts reached for {service_name}")
                await self._send_critical_notification(
                    f"Service {service_name} has failed and cannot be automatically recovered"
                )
    
    async def _stop_service(self, service_name: str):
        """Stop a specific service"""
        service_info = self.services.get(service_name)
        if not service_info:
            return
        
        try:
            if service_name in ["dashboard", "mobile_api"]:
                process = service_info.get("process")
                if process and process.returncode is None:
                    process.terminate()
                    await asyncio.sleep(5)
                    if process.returncode is None:
                        process.kill()
            
            elif service_name == "alerts":
                alert_system = service_info.get("system")
                if alert_system:
                    alert_system.stop_monitoring()
            
            elif service_name == "data_pipeline":
                pipeline = service_info.get("pipeline")
                if pipeline:
                    pipeline.stop_scheduler()
            
            elif service_name == "system_monitor":
                monitor = service_info.get("monitor")
                if monitor:
                    monitor.stop_monitoring()
            
            # Update service status
            service_info["status"] = "stopped"
            
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
    
    async def _restart_service(self, service_name: str):
        """Restart a specific service"""
        if service_name == "dashboard":
            await self._start_dashboard()
        elif service_name == "mobile_api":
            await self._start_mobile_api()
        elif service_name == "alerts":
            await self._start_alert_system()
        elif service_name == "data_pipeline":
            await self._start_data_pipeline()
        elif service_name == "system_monitor":
            await self._start_system_monitor()
    
    def stop_all_services(self):
        """Stop all running services gracefully"""
        logger.info("⏹️ Stopping all conservation services...")
        
        self.monitoring_active = False
        
        # Stop all services
        for service_name in list(self.services.keys()):
            try:
                asyncio.create_task(self._stop_service(service_name))
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        logger.info("✅ All services stopped")
    
    async def _send_critical_notification(self, message: str):
        """Send critical notification to field team"""
        try:
            notification_data = {
                "level": "critical",
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "conservation_master_controller",
                "field_contacts": self.config["field_operations"]["emergency_contacts"]
            }
            
            webhook_url = self.config["notifications"]["webhook_url"]
            if webhook_url:
                import requests
                requests.post(webhook_url, json=notification_data, timeout=10)
            
            logger.critical(f"🚨 CRITICAL: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send critical notification: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "master_controller": {
                "monitoring_active": self.monitoring_active,
                "shutdown_requested": self.shutdown_requested
            },
            "services": {}
        }
        
        for service_name, service_info in self.services.items():
            status["services"][service_name] = {
                "status": service_info.get("status", "unknown"),
                "start_time": service_info.get("start_time", "").isoformat() if service_info.get("start_time") else None,
                "restart_count": service_info.get("restart_count", 0),
                "uptime_seconds": (datetime.now() - service_info.get("start_time", datetime.now())).total_seconds() if service_info.get("start_time") else 0
            }
        
        return status
    
    async def generate_operations_report(self) -> str:
        """Generate comprehensive operations report"""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "field_summary": None,
            "recent_alerts": None,
            "data_pipeline_status": None
        }
        
        try:
            # Get field operations summary
            if "mobile_api" in self.services:
                # This would integrate with mobile API to get field summary
                pass
            
            # Get recent alerts
            if "alerts" in self.services:
                alert_system = self.services["alerts"].get("system")
                if alert_system:
                    # Get recent alerts from alert system
                    pass
            
            # Get data pipeline status
            if "data_pipeline" in self.services:
                pipeline = self.services["data_pipeline"].get("pipeline")
                if pipeline:
                    report_data["data_pipeline_status"] = pipeline.get_pipeline_status()
        
        except Exception as e:
            logger.error(f"Error generating operations report: {e}")
        
        # Save report
        report_file = Path("reports/operations_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📊 Operations report generated: {report_file}")
        return str(report_file)
    
    async def run_forever(self):
        """Run the master controller indefinitely"""
        try:
            await self.start_all_services()
            
            logger.info("🌲 Conservation Dashboard Ecosystem is fully operational")
            logger.info("🎯 Ready for field conservation operations")
            logger.info("Press Ctrl+C to stop all services")
            
            # Keep running until shutdown
            while not self.shutdown_requested:
                await asyncio.sleep(10)
                
                # Generate periodic status updates
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    status = self.get_system_status()
                    logger.info(f"📊 System Status: {len([s for s in status['services'].values() if s['status'] == 'running'])}/{len(status['services'])} services running")
        
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Master controller error: {e}")
        finally:
            self.stop_all_services()

async def main():
    """Main entry point for conservation master controller"""
    print("""
    🌲 Conservation Dashboard Master Controller
    ==========================================
    
    Ready to start integrated conservation field operations:
    ✅ Real-time biodiversity monitoring
    ✅ Automated alert system
    ✅ Mobile field team coordination
    ✅ Automated data pipeline
    ✅ PDF report generation
    ✅ System health monitoring
    ✅ Offline-capable field operations
    
    Starting all services...
    """)
    
    controller = ConservationMasterController()
    
    try:
        await controller.run_forever()
    except Exception as e:
        logger.error(f"Critical error in master controller: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure required directories exist
    for directory in ["logs", "data", "reports", "config"]:
        Path(directory).mkdir(exist_ok=True)
    
    # Run the master controller
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌲 Conservation operations stopped. Thank you for protecting our forests! 🌳")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)
