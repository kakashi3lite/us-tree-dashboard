# Benchmarks

## Performance Targets
- Initial dashboard load < 3s with sample data.
- API `/health` endpoint < 100ms.

## Known Bottlenecks
- Loading large GBIF CSV files in `TreeDashboard.load_data`.
- Geospatial operations in `geopandas` when rendering maps.

## How to Run
```bash
pip install -r requirements.txt
pytest tests/performance/performance_tests.py
```
The performance suite uses `locust` and `memory_profiler` to record metrics.

