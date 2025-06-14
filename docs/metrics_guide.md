# Tree Dashboard Metrics Guide 📊

## Performance Metrics

### Response Time
```sql
SELECT
  endpoint,
  AVG(response_time) as avg_rt,
  PERCENTILE_CONT(0.95) as p95_rt
FROM request_logs
GROUP BY endpoint;
```

### User Engagement
- Session duration
- Interactive elements usage
- Feature adoption rates

## Environmental Impact Metrics

### Carbon Sequestration
```python
def calculate_carbon_impact(tree_data):
    return {
        'annual_co2_absorption': tree_data['diameter'] * 0.15,
        'lifetime_impact': tree_data['age'] * annual_co2_absorption
    }
```

### Biodiversity Index
- Species diversity
- Age distribution
- Health indicators

## Data Quality Metrics

### Completeness Score
```python
def completeness_score(dataset):
    required_fields = ['species', 'location', 'diameter', 'health']
    return sum(dataset[field].notna().mean() for field in required_fields) / len(required_fields)
```

### Accuracy Validation
- GPS coordinate precision
- Species identification confidence
- Measurement accuracy

## System Health Metrics

### Infrastructure
```promql
# CPU Usage
rate(node_cpu_seconds_total{mode="user"}[5m])

# Memory Usage
node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes
```

### Database Performance
```sql
SELECT
  relname as table_name,
  n_live_tup as row_count,
  n_dead_tup as dead_tuples
FROM pg_stat_user_tables;
```

## API Usage Metrics

### Endpoint Traffic
```promql
rate(http_requests_total{handler=~"/api/.*"}[5m])
```

### Error Rates
```sql
SELECT
  status_code,
  COUNT(*) as error_count
FROM api_logs
WHERE status_code >= 400
GROUP BY status_code;
```

## Cache Performance

### Hit Rates
```python
def calculate_cache_metrics():
    return {
        'hit_rate': cache_hits / total_requests,
        'miss_rate': cache_misses / total_requests,
        'eviction_rate': evictions / time_period
    }
```

### Memory Usage
```redis
INFO memory
```

## Search Performance

### Query Latency
```elasticsearch
GET /_nodes/stats/indices/search
```

### Index Stats
```elasticsearch
GET /trees/_stats
```

## Real-time Monitoring

### Grafana Dashboards
1. System Overview
   - CPU, Memory, Disk Usage
   - Network I/O
   - Error Rates

2. Application Metrics
   - Request Latency
   - Cache Hit Rates
   - Database Connections

3. Business Metrics
   - Active Users
   - Data Updates
   - Search Queries

## Reporting

### Daily Summary
```python
def generate_daily_report():
    return {
        'system_health': get_system_metrics(),
        'user_engagement': get_engagement_metrics(),
        'data_quality': get_quality_metrics(),
        'environmental_impact': get_impact_metrics()
    }
```

### Weekly Analysis
- Performance trends
- Usage patterns
- Resource utilization
- Error patterns

## Alert Thresholds

### Critical Alerts
```yaml
- name: high_error_rate
  threshold: error_rate > 5%
  duration: 5m

- name: high_latency
  threshold: p95_latency > 500ms
  duration: 10m

- name: low_cache_hit_rate
  threshold: cache_hit_rate < 80%
  duration: 15m
```

### Warning Alerts
```yaml
- name: elevated_cpu_usage
  threshold: cpu_usage > 70%
  duration: 10m

- name: increased_dead_tuples
  threshold: dead_tuples > 10000
  duration: 1h
```

## Metric Collection

### Prometheus Scraping
```yaml
scrape_configs:
  - job_name: 'tree_dashboard'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8050']
```

### Log Aggregation
```python
from elasticsearch import Elasticsearch

def index_logs(logs):
    es = Elasticsearch()
    for log in logs:
        es.index(index="dashboard-logs", body=log)
```

## Visualization Examples

### Time Series
```javascript
const timeSeriesChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: timePoints,
        datasets: [{
            label: 'Response Time',
            data: responseTimes
        }]
    }
});
```

### Geospatial Heatmap
```javascript
const heatmap = L.heatLayer(treeLocations, {
    radius: 25,
    blur: 15,
    maxZoom: 10
});
```

## Custom Metrics

### Define New Metrics
```python
class CustomMetric:
    def __init__(self, name, calculation_func):
        self.name = name
        self.calculate = calculation_func

    def record(self, value):
        prometheus_client.Gauge(f"custom_{self.name}").set(value)
```

### Implementation
```python
health_score = CustomMetric(
    "tree_health_score",
    lambda data: weighted_average([
        data['leaf_density'],
        data['trunk_condition'],
        data['growth_rate']
    ])
)
```