# Data Visualization Guide 📊

## Interactive Visualizations

### Tree Distribution Map

```javascript
const map = L.map('map').setView([40.7128, -74.0060], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// Tree clusters with custom markers
const markers = L.markerClusterGroup({
    iconCreateFunction: (cluster) => {
        return L.divIcon({
            html: `<div class="cluster-icon">${cluster.getChildCount()}</div>`,
            className: 'custom-cluster',
            iconSize: L.point(40, 40)
        });
    }
});

// Add tree markers with tooltips
trees.forEach(tree => {
    const marker = L.marker([tree.lat, tree.lon], {
        icon: L.divIcon({
            html: `🌳`,
            className: `tree-icon ${tree.health}`,
            iconSize: [20, 20]
        })
    });
    
    marker.bindPopup(`
        <div class="tree-popup">
            <h3>${tree.species}</h3>
            <p>Health: ${tree.health}</p>
            <p>Height: ${tree.height}m</p>
            <p>Age: ${tree.age} years</p>
            <button onclick="showTreeDetails('${tree.id}')">Details</button>
        </div>
    `);
    
    markers.addLayer(marker);
});

map.addLayer(markers);
```

### Species Distribution Chart

```javascript
const ctx = document.getElementById('speciesChart').getContext('2d');

const speciesChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: speciesData.map(d => d.name),
        datasets: [{
            label: 'Number of Trees',
            data: speciesData.map(d => d.count),
            backgroundColor: speciesData.map(d => 
                d.health_score > 0.8 ? '#4CAF50' :
                d.health_score > 0.6 ? '#FFC107' : '#F44336'
            )
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { position: 'top' },
            tooltip: {
                callbacks: {
                    afterBody: (tooltipItems) => {
                        const idx = tooltipItems[0].dataIndex;
                        const species = speciesData[idx];
                        return [
                            `Health Score: ${species.health_score.toFixed(2)}`,
                            `Average Age: ${species.avg_age} years`,
                            `CO₂ Absorption: ${species.co2_absorption}kg/year`
                        ];
                    }
                }
            }
        }
    }
});
```

### Environmental Impact Dashboard

```javascript
const createGauge = (elementId, value, max, title) => {
    const gauge = new JustGage({
        id: elementId,
        value: value,
        min: 0,
        max: max,
        title: title,
        label: "metric tons/year",
        pointer: true,
        pointerOptions: {
            toplength: -15,
            bottomlength: 10,
            bottomwidth: 12,
            color: '#8e8e93',
            stroke: '#ffffff',
            stroke_width: 3,
            stroke_linecap: 'round'
        },
        customSectors: [{
            color: "#ff4444",
            lo: 0,
            hi: max/3
        }, {
            color: "#ffbb33",
            lo: max/3,
            hi: 2*max/3
        }, {
            color: "#00C851",
            lo: 2*max/3,
            hi: max
        }]
    });
    return gauge;
};

// Create environmental impact gauges
const co2Gauge = createGauge('co2Gauge', co2Absorption, 1000, 'CO₂ Absorption');
const o2Gauge = createGauge('o2Gauge', o2Production, 800, 'O₂ Production');
```

### Time Series Analysis

```javascript
const timeSeriesChart = new ApexCharts(document.querySelector("#timeSeriesChart"), {
    series: [{
        name: 'Tree Growth',
        data: growthData.map(d => ({
            x: new Date(d.date).getTime(),
            y: d.height
        }))
    }, {
        name: 'Environmental Impact',
        data: growthData.map(d => ({
            x: new Date(d.date).getTime(),
            y: d.impact
        }))
    }],
    chart: {
        type: 'area',
        stacked: false,
        height: 350,
        zoom: {
            type: 'x',
            enabled: true,
            autoScaleYaxis: true
        },
        toolbar: {
            autoSelected: 'zoom'
        }
    },
    dataLabels: {
        enabled: false
    },
    markers: {
        size: 0
    },
    fill: {
        type: 'gradient',
        gradient: {
            shadeIntensity: 1,
            inverseColors: false,
            opacityFrom: 0.5,
            opacityTo: 0,
            stops: [0, 90, 100]
        }
    },
    yaxis: [{
        title: {
            text: 'Tree Height (m)'
        }
    }, {
        opposite: true,
        title: {
            text: 'Environmental Impact Score'
        }
    }],
    tooltip: {
        shared: true
    }
});

timeSeriesChart.render();
```

### Health Distribution Heatmap

```javascript
const heatmapData = trees.map(tree => ([
    tree.lat,
    tree.lon,
    tree.health_score
]));

const heatmap = L.heatLayer(heatmapData, {
    radius: 25,
    blur: 15,
    maxZoom: 10,
    max: 1.0,
    gradient: {
        0.4: '#FF4444',
        0.6: '#FFBB33',
        0.7: '#00C851',
        0.8: '#007E33'
    }
}).addTo(map);

// Add layer controls
const overlayMaps = {
    "Trees": markers,
    "Health Heatmap": heatmap
};

L.control.layers(null, overlayMaps).addTo(map);
```

### Interactive Data Table

```javascript
const dataTable = new DataTable('#treeTable', {
    columns: [
        { data: 'id', title: 'ID' },
        { data: 'species', title: 'Species' },
        { 
            data: 'health',
            title: 'Health',
            render: (data, type, row) => {
                if (type === 'display') {
                    return `<div class="health-indicator ${data.toLowerCase()}">
                        ${data}
                    </div>`;
                }
                return data;
            }
        },
        { 
            data: 'metrics',
            title: 'Environmental Impact',
            render: (data, type, row) => {
                if (type === 'display') {
                    return `<div class="impact-meter">
                        <div class="impact-bar" style="width: ${data.impact * 100}%"></div>
                        <span>${(data.impact * 100).toFixed(1)}%</span>
                    </div>`;
                }
                return data.impact;
            }
        },
        {
            data: null,
            title: 'Actions',
            render: (data, type, row) => `
                <button onclick="showDetails('${row.id}')">Details</button>
                <button onclick="showLocation('${row.id}')">Location</button>
            `
        }
    ],
    data: trees,
    pageLength: 10,
    responsive: true,
    dom: 'Bfrtip',
    buttons: [
        'copy', 'csv', 'excel', 'pdf', 'print'
    ]
});
```

### Custom CSS Styling

```css
.tree-icon {
    font-size: 20px;
    text-align: center;
    line-height: 20px;
}

.tree-icon.healthy { color: #4CAF50; }
.tree-icon.fair { color: #FFC107; }
.tree-icon.poor { color: #F44336; }

.cluster-icon {
    background: #43a047;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 40px;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
}

.tree-popup {
    padding: 10px;
    max-width: 200px;
}

.tree-popup h3 {
    margin: 0 0 10px 0;
    color: #2e7d32;
}

.health-indicator {
    padding: 5px 10px;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.health-indicator.good { background: #4CAF50; }
.health-indicator.fair { background: #FFC107; }
.health-indicator.poor { background: #F44336; }

.impact-meter {
    width: 100px;
    height: 20px;
    background: #f5f5f5;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}

.impact-bar {
    height: 100%;
    background: linear-gradient(90deg, #81c784 0%, #43a047 100%);
    transition: width 0.3s ease;
}

.impact-meter span {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    color: #333;
    font-size: 12px;
    font-weight: bold;
}
```

### Mobile Responsiveness

```javascript
// Adjust visualization based on screen size
const updateVisualization = () => {
    const width = window.innerWidth;
    
    if (width < 768) { // Mobile
        speciesChart.options.legend.position = 'bottom';
        map.setZoom(map.getZoom() - 1);
    } else { // Desktop
        speciesChart.options.legend.position = 'right';
        map.setZoom(map.getZoom() + 1);
    }
    
    speciesChart.update();
};

window.addEventListener('resize', updateVisualization);
```

### Data Export Options

```javascript
const exportData = (format) => {
    const data = trees.map(tree => ({
        id: tree.id,
        species: tree.species,
        health: tree.health,
        height: tree.height,
        diameter: tree.diameter,
        latitude: tree.lat,
        longitude: tree.lon,
        co2_absorption: tree.metrics.co2_absorption,
        oxygen_production: tree.metrics.oxygen_production
    }));
    
    switch(format) {
        case 'csv':
            return Papa.unparse(data);
        case 'json':
            return JSON.stringify(data, null, 2);
        case 'geojson':
            return {
                type: 'FeatureCollection',
                features: data.map(tree => ({
                    type: 'Feature',
                    geometry: {
                        type: 'Point',
                        coordinates: [tree.longitude, tree.latitude]
                    },
                    properties: tree
                }))
            };
    }
};
```