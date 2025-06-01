import os
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd
import zipfile
import rasterio
from rasterio.mask import mask
import json
from typing import Dict, List, Tuple
import numpy as np

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
DRYAD_DATA_URL = "https://datadryad.org/api/v2/datasets/doi:10.5061/dryad.2jm63xsrf/download"
USDA_TCC_BASE_URL = "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover"
TIGER_COUNTIES_URL = "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip"

def download_file(url: str, output_path: Path) -> None:
    """Download a file from a URL to the specified path"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_and_prepare_dryad_data() -> None:
    """Download and prepare the Dryad 63-city tree inventory dataset"""
    dryad_zip = DATA_DIR / "dryad_data.zip"
    dryad_dir = DATA_DIR / "dryad"
    
    # Download if not exists
    if not dryad_zip.exists():
        print("Downloading Dryad dataset...")
        download_file(DRYAD_DATA_URL, dryad_zip)
    
    # Extract if needed
    if not dryad_dir.exists():
        print("Extracting Dryad dataset...")
        with zipfile.ZipFile(dryad_zip, 'r') as zip_ref:
            zip_ref.extractall(dryad_dir)
    
    # Process and combine city files
    city_files = list(dryad_dir.glob("*.csv"))
    combined_df = pd.DataFrame()
    
    for file in city_files:
        df = pd.read_csv(file)
        city_name = file.stem.replace("_Street_Trees", "")
        df['city'] = city_name
        combined_df = pd.concat([combined_df, df])
    
    # Save combined dataset
    combined_df.to_parquet(DATA_DIR / "combined_trees.parquet")

def download_and_prepare_tcc_data() -> None:
    """Download and prepare the USDA Tree Canopy Cover dataset"""
    tcc_dir = DATA_DIR / "tcc"
    tcc_dir.mkdir(exist_ok=True)
    
    # Download state TCC files (example for a few states)
    states = ['MA', 'NY', 'CA']  # Add more states as needed
    
    for state in states:
        tcc_file = tcc_dir / f"{state}_TCC.tif"
        if not tcc_file.exists():
            url = f"{USDA_TCC_BASE_URL}/{state}_TCC.tif"
            print(f"Downloading TCC data for {state}...")
            download_file(url, tcc_file)

def download_and_prepare_counties() -> None:
    """Download and prepare US counties shapefile"""
    counties_zip = DATA_DIR / "counties.zip"
    counties_dir = DATA_DIR / "counties"
    
    # Download if not exists
    if not counties_zip.exists():
        print("Downloading US counties shapefile...")
        download_file(TIGER_COUNTIES_URL, counties_zip)
    
    # Extract if needed
    if not counties_dir.exists():
        print("Extracting counties shapefile...")
        with zipfile.ZipFile(counties_zip, 'r') as zip_ref:
            zip_ref.extractall(counties_dir)
    
    # Read and simplify counties for faster rendering
    counties = gpd.read_file(counties_dir / "tl_2021_us_county.shp")
    counties_simple = counties.simplify(tolerance=0.01)
    counties_simple.to_file(DATA_DIR / "counties_simplified.gpkg", driver="GPKG")

def calculate_county_canopy() -> None:
    """Calculate average canopy cover per county"""
    counties = gpd.read_file(DATA_DIR / "counties_simplified.gpkg")
    tcc_dir = DATA_DIR / "tcc"
    results = []
    
    for tcc_file in tcc_dir.glob("*.tif"):
        state = tcc_file.stem[:2]
        print(f"Processing {state}...")
        
        with rasterio.open(tcc_file) as src:
            state_counties = counties[counties['STATEFP'] == state]
            
            for idx, county in state_counties.iterrows():
                try:
                    geom = [county.geometry.__geo_interface__]
                    out_image, out_transform = mask(src, geom, crop=True)
                    valid_pixels = out_image[out_image >= 0]
                    if len(valid_pixels) > 0:
                        mean_canopy = float(valid_pixels.mean())
                    else:
                        mean_canopy = 0
                except Exception as e:
                    print(f"Error processing county {county['NAME']}: {e}")
                    mean_canopy = 0
                
                results.append({
                    'GEOID': county['GEOID'],
                    'STATE': state,
                    'COUNTY': county['NAME'],
                    'canopy_pct': mean_canopy
                })
    
    # Save results
    canopy_df = pd.DataFrame(results)
    canopy_df.to_parquet(DATA_DIR / "county_canopy.parquet")

def main():
    """Main function to download and prepare all data"""
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    print("Starting data preparation...")
    
    try:
        # Download and prepare Dryad tree inventory data
        download_and_prepare_dryad_data()
        
        # Download and prepare USDA TCC data
        download_and_prepare_tcc_data()
        
        # Download and prepare county boundaries
        download_and_prepare_counties()
        
        # Calculate county-level canopy cover
        calculate_county_canopy()
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise

if __name__ == "__main__":
    main()
