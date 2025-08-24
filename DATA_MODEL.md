# Data Model

The dashboard uses CSV-based datasets loaded into Pandas/GeoPandas DataFrames. Key datasets are defined in `config.py`:

| Key | Description | Source File Pattern |
|-----|-------------|---------------------|
| `gbif_species` | Global plant species occurrences | `gbif_species_*.csv` |
| `plant_families` | Taxonomic family information | `plant_families_*.csv` |
| `conservation_status` | IUCN conservation status | `conservation_status_*.csv` |
| `biodiversity_hotspots` | Hotspot locations and metrics | `biodiversity_hotspots_*.csv` |

## Lineage & Retention
- Data files live under `data/` and are loaded on startup.
- No persistent database; in-memory DataFrames are rebuilt each run.

## PII Considerations
Datasets contain environmental metrics and species names; no PII is expected.

