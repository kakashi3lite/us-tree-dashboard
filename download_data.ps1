# Download script for Windows users
Write-Host "Creating data directories..."
New-Item -ItemType Directory -Force -Path "data\raw", "data\processed" | Out-Null

Write-Host "Downloading tree inventory dataset from Dryad..."
$url = "https://datadryad.org/api/v2/datasets/doi:10.5061/dryad.2jm63xsrf/download"
Invoke-WebRequest -Uri $url -OutFile "data\raw\tree_inventory.zip"

Write-Host "Downloading USDA Forest Service Tree Canopy Cover dataset..."
New-Item -ItemType Directory -Force -Path "data\raw\tcc" | Out-Null
$states = @("MA", "NY", "CA", "IL", "TX", "FL", "WA")
foreach ($state in $states) {
    Write-Host "Downloading $state TCC data..."
    $url = "https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/${state}_TCC.tif"
    Invoke-WebRequest -Uri $url -OutFile "data\raw\tcc\${state}_TCC.tif"
}

Write-Host "Downloading US Census TIGER/Line shapefiles..."
$url = "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip"
Invoke-WebRequest -Uri $url -OutFile "data\raw\counties.zip"

Write-Host "Downloads complete. Run 'python src/prepare_data.py' to process the data."
