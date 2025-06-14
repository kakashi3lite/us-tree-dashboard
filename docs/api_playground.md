# Tree Dashboard API Playground 🌳

## Interactive API Documentation

This playground provides interactive examples for testing and understanding the Tree Dashboard API endpoints. Each endpoint includes example requests, response schemas, and a live testing interface.

## Authentication

```javascript
const headers = {
    'Authorization': 'Bearer YOUR_API_TOKEN',
    'Content-Type': 'application/json'
};
```

## Endpoints

### 1. Nearby Trees Search

```http
GET /api/v1/trees/nearby
```

#### Parameters

| Name     | Type    | Description                    |
|----------|---------|--------------------------------|
| lat      | number  | Latitude coordinate            |
| lon      | number  | Longitude coordinate           |
| radius   | number  | Search radius in meters        |
| species  | string? | Optional species filter        |

#### Example Request

```javascript
const response = await fetch('/api/v1/trees/nearby?lat=40.7128&lon=-74.0060&radius=500');
const data = await response.json();
```

#### Response Schema

```json
{
    "trees": [{
        "id": "string",
        "species": "string",
        "location": {
            "lat": "number",
            "lon": "number"
        },
        "metrics": {
            "height": "number",
            "diameter": "number",
            "health": "string"
        },
        "distance": "number"
    }],
    "total": "number",
    "page": "number",
    "pageSize": "number"
}
```

### 2. Species Distribution

```http
GET /api/v1/trees/species/distribution
```

#### Parameters

| Name     | Type    | Description                    |
|----------|---------|--------------------------------|
| region   | string? | Geographic region filter        |
| year     | number? | Year of data collection        |

#### Example Request

```javascript
const response = await fetch('/api/v1/trees/species/distribution?region=manhattan&year=2023');
const data = await response.json();
```

#### Response Schema

```json
{
    "distribution": [{
        "species": "string",
        "count": "number",
        "percentage": "number",
        "health_distribution": {
            "good": "number",
            "fair": "number",
            "poor": "number"
        }
    }],
    "total_trees": "number",
    "unique_species": "number"
}
```

### 3. Environmental Impact

```http
GET /api/v1/environmental-impact
```

#### Parameters

| Name     | Type    | Description                    |
|----------|---------|--------------------------------|
| tree_ids | array   | Array of tree IDs              |
| metrics  | array?  | Specific metrics to calculate  |

#### Example Request

```javascript
const response = await fetch('/api/v1/environmental-impact', {
    method: 'POST',
    headers,
    body: JSON.stringify({
        tree_ids: ["NYC-123", "NYC-124"],
        metrics: ["co2_absorption", "oxygen_production"]
    })
});
const data = await response.json();
```

#### Response Schema

```json
{
    "impacts": [{
        "tree_id": "string",
        "metrics": {
            "co2_absorption": {
                "annual": "number",
                "lifetime": "number",
                "unit": "string"
            },
            "oxygen_production": {
                "annual": "number",
                "unit": "string"
            }
        }
    }],
    "aggregate_impact": {
        "total_co2_absorbed": "number",
        "total_oxygen_produced": "number"
    }
}
```

### 4. Real-time Updates (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8050/ws');

ws.onopen = () => {
    // Subscribe to updates
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['tree_updates', 'health_alerts']
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received update:', data);
};
```

#### Subscription Message Schema

```json
{
    "type": "string",
    "channels": ["string"],
    "filters": {
        "region": "string?",
        "species": "string?",
        "health_status": "string?"
    }
}
```

## Testing Tools

### cURL Examples

```bash
# Nearby Trees Search
curl -X GET 'http://localhost:8050/api/v1/trees/nearby?lat=40.7128&lon=-74.0060&radius=500' \
     -H 'Authorization: Bearer YOUR_API_TOKEN'

# Species Distribution
curl -X GET 'http://localhost:8050/api/v1/trees/species/distribution?region=manhattan' \
     -H 'Authorization: Bearer YOUR_API_TOKEN'

# Environmental Impact
curl -X POST 'http://localhost:8050/api/v1/environmental-impact' \
     -H 'Authorization: Bearer YOUR_API_TOKEN' \
     -H 'Content-Type: application/json' \
     -d '{
         "tree_ids": ["NYC-123", "NYC-124"],
         "metrics": ["co2_absorption", "oxygen_production"]
     }'
```

### Python Client Example

```python
import requests

class TreeDashboardClient:
    def __init__(self, base_url, api_token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def get_nearby_trees(self, lat, lon, radius):
        response = requests.get(
            f'{self.base_url}/api/v1/trees/nearby',
            params={'lat': lat, 'lon': lon, 'radius': radius},
            headers=self.headers
        )
        return response.json()
    
    def get_species_distribution(self, region=None, year=None):
        params = {}
        if region:
            params['region'] = region
        if year:
            params['year'] = year
            
        response = requests.get(
            f'{self.base_url}/api/v1/trees/species/distribution',
            params=params,
            headers=self.headers
        )
        return response.json()
    
    def get_environmental_impact(self, tree_ids, metrics=None):
        data = {'tree_ids': tree_ids}
        if metrics:
            data['metrics'] = metrics
            
        response = requests.post(
            f'{self.base_url}/api/v1/environmental-impact',
            json=data,
            headers=self.headers
        )
        return response.json()
```

## Rate Limits

| Plan      | Requests/min | Burst Limit |
|-----------|-------------|-------------|
| Free      | 60          | 100         |
| Pro       | 300         | 500         |
| Enterprise| 1000        | 2000        |

## Error Codes

| Code | Description                    | Solution                         |
|------|--------------------------------|----------------------------------|
| 429  | Rate limit exceeded           | Reduce request frequency         |
| 401  | Unauthorized                  | Check API token                  |
| 400  | Invalid parameters           | Verify request parameters        |
| 404  | Resource not found           | Check resource IDs               |
| 500  | Internal server error        | Contact support                  |

## Webhooks

### Registration

```javascript
const response = await fetch('/api/v1/webhooks', {
    method: 'POST',
    headers,
    body: JSON.stringify({
        url: 'https://your-server.com/webhook',
        events: ['tree.updated', 'health.alert'],
        secret: 'your-webhook-secret'
    })
});
```

### Event Types

- `tree.created`
- `tree.updated`
- `tree.deleted`
- `health.alert`
- `data.imported`
- `analysis.completed`

## SDK Examples

### React Component

```jsx
import { useTreeData } from '@tree-dashboard/react';

function NearbyTrees({ lat, lon }) {
    const { trees, loading, error } = useTreeData({
        type: 'nearby',
        params: { lat, lon, radius: 500 }
    });

    if (loading) return <LoadingSpinner />;
    if (error) return <ErrorMessage error={error} />;

    return (
        <div className="nearby-trees">
            {trees.map(tree => (
                <TreeCard key={tree.id} tree={tree} />
            ))}
        </div>
    );
}
```

### Vue Component

```vue
<template>
    <div class="species-distribution">
        <pie-chart
            v-if="distribution"
            :data="chartData"
            :options="chartOptions"
        />
        <loading-spinner v-else />
    </div>
</template>

<script>
import { useTreeApi } from '@tree-dashboard/vue';

export default {
    setup() {
        const { data: distribution, loading } = useTreeApi('species/distribution');

        const chartData = computed(() => ({
            labels: distribution.value?.map(d => d.species),
            datasets: [{
                data: distribution.value?.map(d => d.count)
            }]
        }));

        return { distribution, loading, chartData };
    }
};
</script>
```

## GraphQL API (Beta)

```graphql
query NearbyTrees($lat: Float!, $lon: Float!, $radius: Float!) {
    nearbyTrees(lat: $lat, lon: $lon, radius: $radius) {
        id
        species
        location {
            lat
            lon
        }
        metrics {
            height
            diameter
            health
        }
        environmentalImpact {
            co2Absorption
            oxygenProduction
        }
    }
}
```