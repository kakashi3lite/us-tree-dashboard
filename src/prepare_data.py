import os
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd
import zipfile
import rasterio
from rasterio.mask import mask
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from shapely.geometry import Point, Polygon, box
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import redis
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
DRYAD_DATA_URL = "https://datadryad.org/api/v2/datasets/doi:10.5061/dryad.2jm63xsrf/download"
USDA_TCC_BASE_URL = "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover"
TIGER_COUNTIES_URL = "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip"
NOAA_CLIMATE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

# Initialize PostGIS connection
postgis_engine = create_engine(os.getenv('POSTGIS_URL', 'postgresql://user:pass@localhost:5432/trees'))

class DataPreparer:
    """Class to handle data preparation and processing."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        
    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> None:
        """Download a file with progress tracking."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    def prepare_tree_inventory(self) -> None:
        """Prepare and process tree inventory data with PostGIS integration."""
        dryad_zip = self.data_dir / "dryad_data.zip"
        dryad_dir = self.data_dir / "dryad"
        
        if not dryad_zip.exists():
            logger.info("Downloading Dryad dataset...")
            self.download_file(DRYAD_DATA_URL, dryad_zip)
        
        if not dryad_dir.exists():
            logger.info("Extracting Dryad dataset...")
            with zipfile.ZipFile(dryad_zip, 'r') as zip_ref:
                zip_ref.extractall(dryad_dir)
        
        # Process city files in parallel
        city_files = list(dryad_dir.glob("*.csv"))
        with ThreadPoolExecutor() as executor:
            dfs = list(executor.map(self._process_city_file, city_files))
        
        combined_gdf = pd.concat(dfs)
        
        # Store in PostGIS
        with postgis_engine.connect() as conn:
            # Create spatial index
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tree_inventory (
                    id SERIAL PRIMARY KEY,
                    city VARCHAR(100),
                    species VARCHAR(200),
                    diameter FLOAT,
                    height FLOAT,
                    health VARCHAR(50),
                    geom GEOMETRY(Point, 4326)
                );
                CREATE INDEX IF NOT EXISTS tree_inventory_geom_idx 
                ON tree_inventory USING GIST (geom);
            """))
            
            # Insert data
            combined_gdf.to_postgis(
                'tree_inventory',
                postgis_engine,
                if_exists='append',
                index=False
            )

    def _process_city_file(self, file_path: Path) -> gpd.GeoDataFrame:
        """Process individual city file with enhanced validation."""
        df = pd.read_csv(file_path)
        city_name = file_path.stem.replace("_Street_Trees", "")
        
        # Data validation and cleaning
        df = df.dropna(subset=['lat', 'lon', 'species'])
        df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])],
            crs="EPSG:4326"
        )
        
        gdf['city'] = city_name
        return gdf

    def prepare_climate_data(self) -> None:
        """Prepare and integrate climate data with spatial components."""
        climate_dir = self.data_dir / "climate"
        climate_dir.mkdir(exist_ok=True)
        
        # Download and process NOAA climate data
        current_year = datetime.now().year
        for year in range(current_year-5, current_year+1):
            cache_key = f"climate_data_{year}"
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                climate_df = pd.read_json(cached_data)
            else:
                climate_file = climate_dir / f"gsod_{year}.csv"
                if not climate_file.exists():
                    url = f"{NOAA_CLIMATE_URL}/{year}/"
                    self.download_file(url, climate_file)
                
                climate_df = pd.read_csv(climate_file)
                redis_client.setex(
                    cache_key,
                    86400,  # Cache for 24 hours
                    climate_df.to_json()
                )
            
            # Store in PostGIS with spatial index
            with postgis_engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS climate_data (
                        id SERIAL PRIMARY KEY,
                        station_id VARCHAR(20),
                        date DATE,
                        temperature FLOAT,
                        precipitation FLOAT,
                        geom GEOMETRY(Point, 4326)
                    );
                    CREATE INDEX IF NOT EXISTS climate_data_geom_idx 
                    ON climate_data USING GIST (geom);
                    CREATE INDEX IF NOT EXISTS climate_data_date_idx 
                    ON climate_data (date);
                """))
                
                # Convert to GeoDataFrame and store
                climate_gdf = gpd.GeoDataFrame(
                    climate_df,
                    geometry=[Point(xy) for xy in zip(climate_df['longitude'], climate_df['latitude'])],
                    crs="EPSG:4326"
                )
                climate_gdf.to_postgis(
                    'climate_data',
                    postgis_engine,
                    if_exists='append',
                    index=False
                )

    def calculate_environmental_metrics(self) -> None:
        """Calculate environmental metrics using PostGIS and cache results."""
        with postgis_engine.connect() as conn:
            # Calculate tree density per square kilometer
            density_query = text("""
                WITH grid AS (
                    SELECT ST_CreateGrid(
                        ST_EstimatedExtent('tree_inventory', 'geom'),
                        1000, 1000
                    ) AS cell
                )
                SELECT 
                    ST_AsGeoJSON(cell) as geometry,
                    COUNT(t.id) as tree_count,
                    AVG(c.temperature) as avg_temperature
                FROM grid
                LEFT JOIN tree_inventory t ON ST_Contains(cell, t.geom)
                LEFT JOIN climate_data c ON ST_Contains(cell, c.geom)
                GROUP BY cell;
            """)
            
            result = conn.execute(density_query)
            metrics = pd.DataFrame(result.fetchall())
            
            # Cache results
            redis_client.setex(
                'environmental_metrics',
                3600,  # Cache for 1 hour
                metrics.to_json()
            )

def main():
    """Main function to orchestrate data preparation."""
    try:
        preparer = DataPreparer()
        
        logger.info("Starting data preparation...")
        
        # Prepare tree inventory data
        preparer.prepare_tree_inventory()
        
        # Prepare climate data
        preparer.prepare_climate_data()
        
        # Calculate environmental metrics
        preparer.calculate_environmental_metrics()
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
