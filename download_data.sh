#!/bin/bash

echo "Creating data directories..."
mkdir -p data/raw data/processed

echo "Downloading tree inventory dataset from Dryad..."
curl -L "https://datadryad.org/api/v2/datasets/doi:10.5061/dryad.2jm63xsrf/download" -o data/raw/tree_inventory.zip

echo "Downloading USDA Forest Service Tree Canopy Cover dataset..."
mkdir -p data/raw/tcc
states=("MA" "NY" "CA" "IL" "TX" "FL" "WA")
for state in "${states[@]}"
do
    echo "Downloading $state TCC data..."
    curl -L "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/${state}_TCC.tif" -o "data/raw/tcc/${state}_TCC.tif"
done

echo "Downloading US Census TIGER/Line shapefiles..."
curl -L "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip" -o data/raw/counties.zip

echo "Downloads complete. Run 'python src/prepare_data.py' to process the data."
