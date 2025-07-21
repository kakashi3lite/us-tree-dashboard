#!/usr/bin/env python3
"""
Automated data pipeline for conservation datasets.
Optimized for Eli's workflow - automated GBIF updates, forest data synchronization, and field team data integration.
"""

import asyncio
import logging
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import sqlite3
import subprocess
import requests
import zipfile
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import schedule

# Configure logging for data pipeline operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    source_type: str  # 'api', 'ftp', 'http', 'local'
    url: str
    update_frequency: str  # 'daily', 'weekly', 'monthly'
    format: str  # 'csv', 'json', 'geojson', 'shp'
    destination_path: str
    authentication: Optional[Dict[str, str]] = None
    preprocessing_steps: Optional[List[str]] = None
    validation_rules: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PipelineRun:
    """Data pipeline execution record"""
    run_id: str
    source_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # 'running', 'completed', 'failed'
    records_processed: int = 0
    errors: List[str] = None
    file_size_mb: float = 0.0
    checksum: Optional[str] = None

class ConservationDataPipeline:
    """
    Automated data pipeline for conservation datasets.
    Handles GBIF, USDA, IUCN, and other conservation data sources.
    """
    
    def __init__(self, config_path: str = "config/data_pipeline.json"):
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize tracking database
        self.db_path = Path("data/pipeline_tracking.db")
        self._init_database()
        
        # Data sources configuration
        self.data_sources = self._setup_data_sources()
        
        # Initialize scheduler
        self.scheduler_running = False
        
        logger.info("🌲 Conservation Data Pipeline initialized for automated field operations")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            "concurrent_downloads": 3,
            "retry_attempts": 3,
            "retry_delay": 30,
            "data_retention_days": 90,
            "validation_enabled": True,
            "notifications": {
                "enabled": True,
                "webhook_url": os.getenv("PIPELINE_WEBHOOK_URL", ""),
                "email_recipients": ["data-team@conservation.org"]
            },
            "gbif": {
                "api_base": "https://api.gbif.org/v1",
                "download_base": "https://download.gbif.org/occurrence",
                "username": os.getenv("GBIF_USERNAME", ""),
                "password": os.getenv("GBIF_PASSWORD", ""),
                "email": os.getenv("GBIF_EMAIL", "")
            },
            "usda": {
                "forest_service_api": "https://apps.fs.usda.gov/fsgisx01/rest/services",
                "api_key": os.getenv("USDA_API_KEY", "")
            },
            "iucn": {
                "red_list_api": "https://apiv3.iucnredlist.org/api/v3",
                "api_key": os.getenv("IUCN_API_KEY", "")
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            else:
                Path(config_path).parent.mkdir(exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default pipeline config at {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def _init_database(self):
        """Initialize SQLite database for pipeline tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Pipeline runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    source_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    records_processed INTEGER DEFAULT 0,
                    errors TEXT,
                    file_size_mb REAL DEFAULT 0.0,
                    checksum TEXT
                )
            """)
            
            # Data checksums table for change detection
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_checksums (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    UNIQUE(source_name, file_path)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON pipeline_runs(start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checksums_source ON data_checksums(source_name)")
            
            conn.commit()
    
    def _setup_data_sources(self) -> List[DataSource]:
        """Setup data source configurations"""
        sources = [
            # GBIF Plant Occurrence Data
            DataSource(
                name="gbif_plants",
                source_type="api",
                url="https://api.gbif.org/v1/occurrence/download/request",
                update_frequency="weekly",
                format="csv",
                destination_path="data/gbif_plants_{date}.csv",
                authentication={
                    "username": self.config["gbif"]["username"],
                    "password": self.config["gbif"]["password"]
                },
                preprocessing_steps=["validate_coordinates", "clean_scientific_names", "filter_usa"],
                validation_rules=["check_required_fields", "validate_coordinates", "check_duplicates"],
                metadata={"taxon_key": 6, "country": "US", "basis_of_record": "HUMAN_OBSERVATION,OBSERVATION"}
            ),
            
            # USDA Forest Service Tree Canopy Data
            DataSource(
                name="usda_tree_canopy",
                source_type="api",
                url="https://apps.fs.usda.gov/fsgisx01/rest/services/RDW_Wildfire/US_TCC2016/MapServer",
                update_frequency="monthly",
                format="geojson",
                destination_path="data/tree_canopy_{date}.geojson",
                preprocessing_steps=["reproject_coordinates", "simplify_geometry"],
                validation_rules=["validate_geometry", "check_coverage_values"]
            ),
            
            # IUCN Red List Conservation Status
            DataSource(
                name="iucn_conservation_status",
                source_type="api",
                url="https://apiv3.iucnredlist.org/api/v3/species/page",
                update_frequency="monthly",
                format="json",
                destination_path="data/conservation_status_{date}.json",
                authentication={"token": self.config["iucn"]["api_key"]},
                preprocessing_steps=["extract_us_species", "normalize_status"],
                validation_rules=["check_status_codes", "validate_species_names"]
            ),
            
            # Biodiversity Hotspots (Critical Ecosystem Partnership Fund)
            DataSource(
                name="biodiversity_hotspots",
                source_type="http",
                url="https://www.cepf.net/sites/default/files/biodiversity-hotspots-shapefile.zip",
                update_frequency="yearly",
                format="shp",
                destination_path="data/biodiversity_hotspots_{date}.shp",
                preprocessing_steps=["extract_zip", "filter_us_regions"],
                validation_rules=["validate_shapefile", "check_geometries"]
            ),
            
            # Plant Families Database
            DataSource(
                name="plant_families",
                source_type="http",
                url="https://www.mobot.org/MOBOT/research/APweb/families.txt",
                update_frequency="monthly",
                format="txt",
                destination_path="data/plant_families_{date}.txt",
                preprocessing_steps=["parse_family_data", "create_hierarchy"],
                validation_rules=["check_family_names", "validate_hierarchy"]
            )
        ]
        
        return sources
    
    def start_scheduler(self):
        """Start automated scheduling for data updates"""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        # Schedule jobs based on update frequency
        for source in self.data_sources:
            if source.update_frequency == "daily":
                schedule.every().day.at("02:00").do(self.update_source, source.name)
            elif source.update_frequency == "weekly":
                schedule.every().monday.at("01:00").do(self.update_source, source.name)
            elif source.update_frequency == "monthly":
                schedule.every().month.do(self.update_source, source.name)
        
        # Health check every hour
        schedule.every().hour.do(self._health_check)
        
        # Cleanup old data weekly
        schedule.every().monday.at("00:00").do(self._cleanup_old_data)
        
        self.scheduler_running = True
        logger.info("📅 Data pipeline scheduler started")
        
        # Run scheduler in background
        import threading
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop automated scheduling"""
        self.scheduler_running = False
        schedule.clear()
        logger.info("⏹️ Data pipeline scheduler stopped")
    
    async def update_all_sources(self) -> Dict[str, bool]:
        """Update all data sources concurrently"""
        logger.info("🚀 Starting update for all data sources")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(self.config["concurrent_downloads"])
        
        async def update_with_semaphore(source_name):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self.update_source, source_name
                )
        
        # Run updates concurrently
        tasks = [update_with_semaphore(source.name) for source in self.data_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        update_status = {}
        for i, result in enumerate(results):
            source_name = self.data_sources[i].name
            if isinstance(result, Exception):
                logger.error(f"Failed to update {source_name}: {result}")
                update_status[source_name] = False
            else:
                update_status[source_name] = result
        
        logger.info(f"✅ Completed updates. Success: {sum(update_status.values())}/{len(update_status)}")
        return update_status
    
    def update_source(self, source_name: str) -> bool:
        """Update a specific data source"""
        source = next((s for s in self.data_sources if s.name == source_name), None)
        if not source:
            logger.error(f"Unknown data source: {source_name}")
            return False
        
        run_id = f"{source_name}_{int(time.time())}"
        run = PipelineRun(
            run_id=run_id,
            source_name=source_name,
            start_time=datetime.now()
        )
        
        logger.info(f"📥 Starting update for {source_name}")
        
        try:
            # Check if update is needed
            if not self._needs_update(source):
                logger.info(f"📋 {source_name} is up to date, skipping")
                return True
            
            # Download data
            file_path = self._download_data(source, run)
            
            # Validate data
            if self.config["validation_enabled"]:
                validation_passed = self._validate_data(source, file_path, run)
                if not validation_passed:
                    raise Exception("Data validation failed")
            
            # Preprocess data
            if source.preprocessing_steps:
                file_path = self._preprocess_data(source, file_path, run)
            
            # Update checksums
            self._update_checksum(source, file_path)
            
            # Update metadata
            self._update_metadata(source, file_path, run)
            
            # Mark as completed
            run.status = "completed"
            run.end_time = datetime.now()
            
            # Calculate file size
            if os.path.exists(file_path):
                run.file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                run.checksum = self._calculate_checksum(file_path)
            
            self._store_pipeline_run(run)
            
            logger.info(f"✅ Successfully updated {source_name} ({run.records_processed:,} records, {run.file_size_mb:.1f} MB)")
            
            # Send notification
            self._send_notification(f"✅ Data update completed: {source_name}", 
                                  f"Successfully processed {run.records_processed:,} records")
            
            return True
            
        except Exception as e:
            run.status = "failed"
            run.end_time = datetime.now()
            if run.errors is None:
                run.errors = []
            run.errors.append(str(e))
            
            self._store_pipeline_run(run)
            
            logger.error(f"❌ Failed to update {source_name}: {e}")
            
            # Send error notification
            self._send_notification(f"❌ Data update failed: {source_name}", 
                                  f"Error: {str(e)}")
            
            return False
    
    def _needs_update(self, source: DataSource) -> bool:
        """Check if data source needs updating based on frequency and last update"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(start_time) FROM pipeline_runs 
                    WHERE source_name = ? AND status = 'completed'
                """, (source.name,))
                
                result = cursor.fetchone()
                if not result[0]:
                    return True  # No previous successful run
                
                last_update = datetime.fromisoformat(result[0])
                now = datetime.now()
                
                # Check based on update frequency
                if source.update_frequency == "daily":
                    return (now - last_update).days >= 1
                elif source.update_frequency == "weekly":
                    return (now - last_update).days >= 7
                elif source.update_frequency == "monthly":
                    return (now - last_update).days >= 30
                else:
                    return True
                    
        except Exception as e:
            logger.warning(f"Error checking update necessity for {source.name}: {e}")
            return True  # Default to update if check fails
    
    def _download_data(self, source: DataSource, run: PipelineRun) -> str:
        """Download data from source"""
        date_str = datetime.now().strftime("%Y%m%d")
        file_path = source.destination_path.format(date=date_str)
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if source.source_type == "api":
            return self._download_from_api(source, file_path, run)
        elif source.source_type == "http":
            return self._download_from_http(source, file_path, run)
        elif source.source_type == "ftp":
            return self._download_from_ftp(source, file_path, run)
        else:
            raise Exception(f"Unsupported source type: {source.source_type}")
    
    def _download_from_api(self, source: DataSource, file_path: str, run: PipelineRun) -> str:
        """Download data from API"""
        if source.name == "gbif_plants":
            return self._download_gbif_data(source, file_path, run)
        elif source.name == "usda_tree_canopy":
            return self._download_usda_data(source, file_path, run)
        elif source.name == "iucn_conservation_status":
            return self._download_iucn_data(source, file_path, run)
        else:
            # Generic API download
            return self._download_generic_api(source, file_path, run)
    
    def _download_gbif_data(self, source: DataSource, file_path: str, run: PipelineRun) -> str:
        """Download GBIF occurrence data"""
        logger.info("📡 Requesting GBIF plant occurrence data for USA")
        
        # Create download request
        download_request = {
            "creator": source.authentication["username"],
            "notification_address": [self.config["gbif"]["email"]],
            "sendNotification": True,
            "format": "SIMPLE_CSV",
            "predicate": {
                "type": "and",
                "predicates": [
                    {"type": "equals", "key": "COUNTRY", "value": "US"},
                    {"type": "equals", "key": "TAXON_KEY", "value": source.metadata["taxon_key"]},
                    {"type": "in", "key": "BASIS_OF_RECORD", "values": source.metadata["basis_of_record"].split(",")}
                ]
            }
        }
        
        # Submit download request
        response = requests.post(
            f"{self.config['gbif']['api_base']}/occurrence/download/request",
            json=download_request,
            auth=(source.authentication["username"], source.authentication["password"]),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        download_key = response.text.strip('"')
        logger.info(f"📋 GBIF download request submitted: {download_key}")
        
        # Wait for download to complete (with timeout)
        max_wait_time = 3600  # 1 hour
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(
                f"{self.config['gbif']['api_base']}/occurrence/download/{download_key}"
            )
            status_data = status_response.json()
            
            if status_data["status"] == "SUCCEEDED":
                # Download the file
                download_url = f"{self.config['gbif']['download_base']}/{download_key}.zip"
                self._download_file(download_url, file_path + ".zip")
                
                # Extract CSV from zip
                with zipfile.ZipFile(file_path + ".zip", 'r') as zip_file:
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    if csv_files:
                        zip_file.extract(csv_files[0], Path(file_path).parent)
                        extracted_path = Path(file_path).parent / csv_files[0]
                        extracted_path.rename(file_path)
                        
                        # Count records
                        with open(file_path, 'r') as f:
                            run.records_processed = sum(1 for line in f) - 1  # Exclude header
                        
                        logger.info(f"📥 Downloaded {run.records_processed:,} GBIF records")
                        return file_path
                
                raise Exception("No CSV file found in GBIF download")
            
            elif status_data["status"] == "FAILED":
                raise Exception(f"GBIF download failed: {status_data}")
            
            # Wait before checking again
            time.sleep(30)
        
        raise Exception("GBIF download timeout")
    
    def _download_usda_data(self, source: DataSource, file_path: str, run: PipelineRun) -> str:
        """Download USDA Forest Service data"""
        logger.info("📡 Downloading USDA Tree Canopy Coverage data")
        
        # Query USDA ArcGIS REST service
        query_params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true"
        }
        
        response = requests.get(
            f"{source.url}/0/query",
            params=query_params,
            timeout=300
        )
        response.raise_for_status()
        
        with open(file_path, 'w') as f:
            json.dump(response.json(), f, indent=2)
        
        # Count features
        data = response.json()
        run.records_processed = len(data.get("features", []))
        
        logger.info(f"📥 Downloaded {run.records_processed:,} USDA features")
        return file_path
    
    def _download_iucn_data(self, source: DataSource, file_path: str, run: PipelineRun) -> str:
        """Download IUCN Red List data"""
        logger.info("📡 Downloading IUCN Conservation Status data")
        
        all_species = []
        page = 0
        
        while True:
            response = requests.get(
                f"{source.url}/{page}",
                headers={"X-Authentication-Token": source.authentication["token"]},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            species_page = data.get("result", [])
            
            if not species_page:
                break
            
            all_species.extend(species_page)
            page += 1
            
            # Rate limiting
            time.sleep(1)
        
        # Save data
        with open(file_path, 'w') as f:
            json.dump(all_species, f, indent=2)
        
        run.records_processed = len(all_species)
        logger.info(f"📥 Downloaded {run.records_processed:,} IUCN species records")
        return file_path
    
    def _download_from_http(self, source: DataSource, file_path: str, run: PipelineRun) -> str:
        """Download data from HTTP URL"""
        logger.info(f"📡 Downloading from HTTP: {source.url}")
        
        response = requests.get(source.url, timeout=300, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"📥 Downloaded file: {file_path}")
        return file_path
    
    def _download_file(self, url: str, file_path: str):
        """Generic file download with progress"""
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (1024 * 1024) == 0:  # Log every MB
                        logger.info(f"📥 Download progress: {progress:.1f}%")
    
    def _validate_data(self, source: DataSource, file_path: str, run: PipelineRun) -> bool:
        """Validate downloaded data"""
        if not source.validation_rules:
            return True
        
        logger.info(f"🔍 Validating {source.name} data")
        
        validation_errors = []
        
        try:
            if source.format == "csv":
                validation_errors.extend(self._validate_csv(file_path, source.validation_rules))
            elif source.format == "json":
                validation_errors.extend(self._validate_json(file_path, source.validation_rules))
            elif source.format == "geojson":
                validation_errors.extend(self._validate_geojson(file_path, source.validation_rules))
            
            if validation_errors:
                if run.errors is None:
                    run.errors = []
                run.errors.extend(validation_errors)
                logger.error(f"❌ Validation failed for {source.name}: {validation_errors}")
                return False
            
            logger.info(f"✅ Validation passed for {source.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation error for {source.name}: {e}")
            return False
    
    def _validate_csv(self, file_path: str, rules: List[str]) -> List[str]:
        """Validate CSV file"""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                if "check_required_fields" in rules:
                    required_fields = ["scientificName", "decimalLatitude", "decimalLongitude"]
                    missing_fields = [field for field in required_fields if field not in headers]
                    if missing_fields:
                        errors.append(f"Missing required fields: {missing_fields}")
                
                if "validate_coordinates" in rules:
                    coord_errors = 0
                    for i, row in enumerate(reader):
                        try:
                            lat = float(row.get("decimalLatitude", 0))
                            lon = float(row.get("decimalLongitude", 0))
                            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                                coord_errors += 1
                        except (ValueError, TypeError):
                            coord_errors += 1
                        
                        if i > 1000:  # Sample validation
                            break
                    
                    if coord_errors > 0:
                        errors.append(f"Invalid coordinates found: {coord_errors} errors")
        
        except Exception as e:
            errors.append(f"CSV validation error: {e}")
        
        return errors
    
    def _validate_json(self, file_path: str, rules: List[str]) -> List[str]:
        """Validate JSON file"""
        errors = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "check_status_codes" in rules and isinstance(data, list):
                valid_statuses = ["LC", "NT", "VU", "EN", "CR", "EW", "EX", "DD", "NE"]
                for item in data[:100]:  # Sample validation
                    if "category" in item and item["category"] not in valid_statuses:
                        errors.append(f"Invalid status code: {item['category']}")
        
        except Exception as e:
            errors.append(f"JSON validation error: {e}")
        
        return errors
    
    def _validate_geojson(self, file_path: str, rules: List[str]) -> List[str]:
        """Validate GeoJSON file"""
        errors = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "validate_geometry" in rules:
                if "features" not in data:
                    errors.append("Invalid GeoJSON: missing features")
                else:
                    for i, feature in enumerate(data["features"][:100]):
                        if "geometry" not in feature:
                            errors.append(f"Feature {i} missing geometry")
        
        except Exception as e:
            errors.append(f"GeoJSON validation error: {e}")
        
        return errors
    
    def _preprocess_data(self, source: DataSource, file_path: str, run: PipelineRun) -> str:
        """Preprocess downloaded data"""
        logger.info(f"🔄 Preprocessing {source.name} data")
        
        # Implementation would depend on specific preprocessing steps
        # This is a simplified version
        
        return file_path
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _update_checksum(self, source: DataSource, file_path: str):
        """Update file checksum in database"""
        checksum = self._calculate_checksum(file_path)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO data_checksums 
                (source_name, file_path, checksum, last_updated)
                VALUES (?, ?, ?, ?)
            """, (source.name, file_path, checksum, datetime.now().isoformat()))
            conn.commit()
    
    def _update_metadata(self, source: DataSource, file_path: str, run: PipelineRun):
        """Update dataset metadata"""
        metadata = {
            "source_name": source.name,
            "file_path": file_path,
            "last_updated": datetime.now().isoformat(),
            "records": run.records_processed,
            "file_size_mb": run.file_size_mb,
            "checksum": run.checksum
        }
        
        metadata_file = self.data_dir / "metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        # Update metadata for this source
        all_metadata[source.name] = metadata
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
    
    def _store_pipeline_run(self, run: PipelineRun):
        """Store pipeline run record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO pipeline_runs 
                (run_id, source_name, start_time, end_time, status, records_processed, errors, file_size_mb, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.source_name,
                run.start_time.isoformat(),
                run.end_time.isoformat() if run.end_time else None,
                run.status,
                run.records_processed,
                json.dumps(run.errors) if run.errors else None,
                run.file_size_mb,
                run.checksum
            ))
            conn.commit()
    
    def _send_notification(self, title: str, message: str):
        """Send notification about pipeline status"""
        if not self.config["notifications"]["enabled"]:
            return
        
        try:
            # Send webhook notification if configured
            webhook_url = self.config["notifications"]["webhook_url"]
            if webhook_url:
                payload = {
                    "title": title,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "source": "conservation_data_pipeline"
                }
                
                requests.post(webhook_url, json=payload, timeout=10)
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _health_check(self):
        """Perform health check on data pipeline"""
        logger.info("🩺 Performing pipeline health check")
        
        issues = []
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage(self.data_dir)
        free_space_gb = disk_usage.free / (1024**3)
        
        if free_space_gb < 5:  # Less than 5GB free
            issues.append(f"Low disk space: {free_space_gb:.1f}GB remaining")
        
        # Check recent failures
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM pipeline_runs 
                WHERE status = 'failed' AND start_time > datetime('now', '-24 hours')
            """)
            recent_failures = cursor.fetchone()[0]
        
        if recent_failures > 0:
            issues.append(f"{recent_failures} pipeline failures in last 24 hours")
        
        if issues:
            self._send_notification("⚠️ Pipeline Health Issues", "\n".join(issues))
        else:
            logger.info("✅ Pipeline health check passed")
    
    def _cleanup_old_data(self):
        """Clean up old data files and records"""
        logger.info("🧹 Cleaning up old data")
        
        cutoff_date = datetime.now() - timedelta(days=self.config["data_retention_days"])
        
        # Clean up old pipeline runs
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM pipeline_runs 
                WHERE start_time < ? AND status IN ('completed', 'failed')
            """, (cutoff_date.isoformat(),))
            deleted_runs = cursor.rowcount
            conn.commit()
        
        logger.info(f"🗑️ Cleaned up {deleted_runs} old pipeline runs")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get recent runs
            cursor.execute("""
                SELECT source_name, status, start_time, records_processed, file_size_mb
                FROM pipeline_runs 
                WHERE start_time > datetime('now', '-7 days')
                ORDER BY start_time DESC
            """)
            recent_runs = cursor.fetchall()
            
            # Get failure count
            cursor.execute("""
                SELECT COUNT(*) FROM pipeline_runs 
                WHERE status = 'failed' AND start_time > datetime('now', '-24 hours')
            """)
            recent_failures = cursor.fetchone()[0]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "scheduler_running": self.scheduler_running,
            "recent_failures": recent_failures,
            "total_sources": len(self.data_sources),
            "recent_runs": [
                {
                    "source": run[0],
                    "status": run[1],
                    "time": run[2],
                    "records": run[3],
                    "size_mb": run[4]
                }
                for run in recent_runs[:10]
            ]
        }

def main():
    """Main function for running data pipeline"""
    pipeline = ConservationDataPipeline()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "update-all":
            asyncio.run(pipeline.update_all_sources())
        elif command == "start-scheduler":
            pipeline.start_scheduler()
            logger.info("Pipeline scheduler started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                pipeline.stop_scheduler()
        elif command.startswith("update-"):
            source_name = command.replace("update-", "")
            pipeline.update_source(source_name)
        elif command == "status":
            status = pipeline.get_pipeline_status()
            print(json.dumps(status, indent=2))
        else:
            print("Usage: python data_pipeline.py [update-all|start-scheduler|update-<source>|status]")
    else:
        print("Conservation Data Pipeline")
        print("Available commands:")
        print("  update-all        - Update all data sources")
        print("  start-scheduler   - Start automated scheduling")
        print("  update-<source>   - Update specific source")
        print("  status           - Show pipeline status")

if __name__ == "__main__":
    main()
