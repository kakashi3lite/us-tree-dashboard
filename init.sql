-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS postgis_raster;

-- Create spatial indexes
CREATE INDEX idx_tree_locations_geom ON tree_locations USING GIST (location);
CREATE INDEX idx_regions_boundary ON regions USING GIST (boundary);
CREATE INDEX idx_climate_zones_boundary ON climate_zones USING GIST (boundary);

-- Create indexes for frequently queried columns
CREATE INDEX idx_tree_species ON tree_locations (species);
CREATE INDEX idx_tree_health ON tree_locations (health_condition);
CREATE INDEX idx_measurement_date ON tree_measurements (measurement_date);
CREATE INDEX idx_environmental_calc_date ON environmental_impacts (calculation_date);

-- Set up partitioning for large tables
CREATE TABLE tree_measurements_partitioned (
    LIKE tree_measurements INCLUDING INDEXES
) PARTITION BY RANGE (measurement_date);

-- Create partitions for the last 5 years
CREATE TABLE tree_measurements_y2019 PARTITION OF tree_measurements_partitioned
    FOR VALUES FROM ('2019-01-01') TO ('2020-01-01');
CREATE TABLE tree_measurements_y2020 PARTITION OF tree_measurements_partitioned
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
CREATE TABLE tree_measurements_y2021 PARTITION OF tree_measurements_partitioned
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
CREATE TABLE tree_measurements_y2022 PARTITION OF tree_measurements_partitioned
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
CREATE TABLE tree_measurements_y2023 PARTITION OF tree_measurements_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- Create materialized view for common aggregations
CREATE MATERIALIZED VIEW tree_density_by_region AS
SELECT 
    r.id as region_id,
    r.name as region_name,
    COUNT(t.id) as tree_count,
    ST_Area(ST_Transform(r.boundary, 3857)) / 1000000.0 as area_km2,
    COUNT(t.id) / (ST_Area(ST_Transform(r.boundary, 3857)) / 1000000.0) as trees_per_km2
FROM regions r
LEFT JOIN tree_locations t ON ST_Contains(r.boundary, t.location)
GROUP BY r.id, r.name, r.boundary;

-- Create index on materialized view
CREATE INDEX idx_tree_density_region ON tree_density_by_region (region_id);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_tree_density()
RETURNS trigger AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY tree_density_by_region;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to refresh materialized view
CREATE TRIGGER refresh_tree_density_trigger
AFTER INSERT OR UPDATE OR DELETE ON tree_locations
FOR EACH STATEMENT
EXECUTE FUNCTION refresh_tree_density();