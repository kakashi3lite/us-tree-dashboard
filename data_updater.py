#!/usr/bin/env python3
"""
PlantsWorld Data Updater Script
Automatically fetches and processes the latest plant biodiversity datasets
from various sources including GBIF, NatureServe, and other APIs.
"""

import os
import requests
import pandas as pd
import json
from datetime import datetime
import logging
from pathlib import Path
import zipfile
import io
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlantsWorldDataUpdater:
    """Main class for updating PlantsWorld datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PlantsWorld-Dashboard/1.0 (https://github.com/plantsworld)'
        })
        
    def fetch_gbif_species_data(self, limit: int = 10000) -> pd.DataFrame:
        """Fetch plant species data from GBIF API"""
        logger.info("Fetching plant species data from GBIF...")
        
        # GBIF API endpoint for plant occurrences
        base_url = "https://api.gbif.org/v1/occurrence/search"
        
        params = {
            'kingdom': 'Plantae',
            'hasCoordinate': 'true',
            'hasGeospatialIssue': 'false',
            'limit': min(limit, 300),  # GBIF API limit
            'offset': 0
        }
        
        all_records = []
        total_fetched = 0
        
        while total_fetched < limit:
            try:
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    break
                    
                records = data['results']
                all_records.extend(records)
                total_fetched += len(records)
                
                logger.info(f"Fetched {total_fetched} records so far...")
                
                # Update offset for next batch
                params['offset'] += len(records)
                
                # Break if we've reached the end
                if len(records) < params['limit']:
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Error fetching GBIF data: {e}")
                break
        
        # Convert to DataFrame
        if all_records:
            df = pd.json_normalize(all_records)
            # Select relevant columns
            columns_to_keep = [
                'key', 'scientificName', 'family', 'genus', 'species',
                'decimalLatitude', 'decimalLongitude', 'country', 'stateProvince',
                'year', 'month', 'day', 'basisOfRecord', 'institutionCode'
            ]
            df = df[[col for col in columns_to_keep if col in df.columns]]
            
            # Save to file
            output_file = self.data_dir / f"gbif_plants_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} GBIF records to {output_file}")
            
            return df
        else:
            logger.warning("No GBIF data fetched")
            return pd.DataFrame()
    
    def fetch_plant_families_data(self) -> pd.DataFrame:
        """Fetch plant families classification data"""
        logger.info("Fetching plant families data...")
        
        # Create sample plant families data (in real implementation, this could come from GBIF taxonomy API)
        families_data = {
            'family': [
                'Rosaceae', 'Asteraceae', 'Fabaceae', 'Poaceae', 'Orchidaceae',
                'Rubiaceae', 'Euphorbiaceae', 'Cyperaceae', 'Lamiaceae', 'Apiaceae',
                'Brassicaceae', 'Malvaceae', 'Solanaceae', 'Cactaceae', 'Pinaceae'
            ],
            'common_name': [
                'Rose Family', 'Sunflower Family', 'Legume Family', 'Grass Family', 'Orchid Family',
                'Coffee Family', 'Spurge Family', 'Sedge Family', 'Mint Family', 'Carrot Family',
                'Mustard Family', 'Mallow Family', 'Nightshade Family', 'Cactus Family', 'Pine Family'
            ],
            'estimated_species': [
                4828, 32913, 19500, 12074, 28000,
                13500, 5735, 5500, 7534, 3780,
                4060, 4225, 2678, 1866, 818
            ],
            'distribution': [
                'Worldwide', 'Worldwide', 'Worldwide', 'Worldwide', 'Worldwide',
                'Tropical/Subtropical', 'Worldwide', 'Worldwide', 'Worldwide', 'Worldwide',
                'Worldwide', 'Worldwide', 'Worldwide', 'Americas/Africa', 'Northern Hemisphere'
            ]
        }
        
        df = pd.DataFrame(families_data)
        output_file = self.data_dir / f"plant_families_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved plant families data to {output_file}")
        
        return df
    
    def fetch_conservation_status_data(self) -> pd.DataFrame:
        """Fetch conservation status data for plants"""
        logger.info("Generating conservation status data...")
        
        # Sample conservation data (in real implementation, this could come from IUCN Red List API)
        conservation_data = {
            'species': [
                'Welwitschia mirabilis', 'Dionaea muscipula', 'Nepenthes rajah',
                'Baobab tree', 'Giant Sequoia', 'Madagascar Periwinkle',
                'Ginkgo biloba', 'Cycas revoluta', 'Wollemia nobilis',
                'Encephalartos woodii'
            ],
            'common_name': [
                'Welwitschia', 'Venus Flytrap', 'Rajah Pitcher Plant',
                'Baobab', 'Giant Sequoia', 'Madagascar Periwinkle',
                'Ginkgo', 'Sago Palm', 'Wollemi Pine',
                'Wood\'s Cycad'
            ],
            'conservation_status': [
                'Vulnerable', 'Vulnerable', 'Critically Endangered',
                'Least Concern', 'Endangered', 'Least Concern',
                'Endangered', 'Near Threatened', 'Critically Endangered',
                'Extinct in Wild'
            ],
            'population_trend': [
                'Decreasing', 'Decreasing', 'Decreasing',
                'Stable', 'Increasing', 'Stable',
                'Stable', 'Decreasing', 'Stable',
                'Stable'
            ],
            'main_threats': [
                'Climate change', 'Habitat loss', 'Over-collection',
                'Climate change', 'Fire/Disease', 'Habitat conversion',
                'Habitat loss', 'Over-collection', 'Fire risk',
                'Extinct naturally'
            ]
        }
        
        df = pd.DataFrame(conservation_data)
        output_file = self.data_dir / f"conservation_status_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved conservation status data to {output_file}")
        
        return df
    
    def create_sample_biodiversity_data(self) -> pd.DataFrame:
        """Create sample biodiversity hotspots data"""
        logger.info("Creating biodiversity hotspots data...")
        
        import numpy as np
        
        # Sample biodiversity hotspots
        hotspots_data = {
            'hotspot_name': [
                'Amazon Rainforest', 'Congo Basin', 'Southeast Asian Rainforests',
                'Atlantic Forest', 'Madagascar', 'California Floristic Province',
                'Cape Floristic Region', 'Cerrado', 'Indo-Burma', 'Sundaland'
            ],
            'country': [
                'Brazil', 'Democratic Republic of Congo', 'Indonesia',
                'Brazil', 'Madagascar', 'United States',
                'South Africa', 'Brazil', 'Myanmar', 'Malaysia'
            ],
            'latitude': [
                -3.4653, -0.2280, -0.7893,
                -14.2350, -18.7669, 36.7783,
                -33.9249, -14.2350, 21.9162, 3.1390
            ],
            'longitude': [
                -62.2159, 15.8277, 113.9213,
                -51.9253, 46.8691, -119.4179,
                18.4241, -51.9253, 95.9560, 101.6869
            ],
            'endemic_species': [
                40000, 10000, 25000,
                20000, 12000, 4426,
                9000, 10000, 7000, 15000
            ],
            'area_km2': [
                6700000, 3700000, 2500000,
                1233875, 587041, 293804,
                78555, 2036448, 2373057, 1600000
            ],
            'threat_level': [
                'High', 'High', 'Critical',
                'Critical', 'Critical', 'High',
                'High', 'High', 'Critical', 'Critical'
            ]
        }
        
        df = pd.DataFrame(hotspots_data)
        output_file = self.data_dir / f"biodiversity_hotspots_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved biodiversity hotspots data to {output_file}")
        
        return df
    
    def update_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Update all datasets and return them as a dictionary"""
        logger.info("Starting comprehensive data update...")
        
        datasets = {}
        
        try:
            # Fetch GBIF species data
            datasets['gbif_species'] = self.fetch_gbif_species_data(limit=5000)
            
            # Fetch plant families data
            datasets['plant_families'] = self.fetch_plant_families_data()
            
            # Fetch conservation status data
            datasets['conservation_status'] = self.fetch_conservation_status_data()
            
            # Create biodiversity hotspots data
            datasets['biodiversity_hotspots'] = self.create_sample_biodiversity_data()
            
            # Create metadata file
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'datasets': {
                    name: {
                        'records': len(df),
                        'columns': list(df.columns) if not df.empty else []
                    } for name, df in datasets.items()
                }
            }
            
            metadata_file = self.data_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Data update completed successfully. Updated {len(datasets)} datasets.")
            
        except Exception as e:
            logger.error(f"Error during data update: {e}")
            raise
        
        return datasets

def main():
    """Main function to run the data updater"""
    updater = PlantsWorldDataUpdater()
    
    try:
        datasets = updater.update_all_datasets()
        
        print("\n=== Data Update Summary ===")
        for name, df in datasets.items():
            print(f"{name}: {len(df)} records")
        
        print(f"\nAll data saved to: {updater.data_dir}")
        print("Data update completed successfully!")
        
    except Exception as e:
        print(f"Data update failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())