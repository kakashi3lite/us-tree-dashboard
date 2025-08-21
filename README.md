# 🌿 PlantsWorld - Interactive Plant Biodiversity Dashboard

A scalable, interactive dashboard for exploring plant biodiversity data with real-time visualizations, conservation insights, network analysis, and API integration capabilities.

## 🚀 Features

### 🌳 Tree Dashboard

- **Interactive Tree Map**: Visualize tree distribution across different states and cities
- **Summary Statistics**: Get quick insights with summary cards showing key metrics
- **Data Filtering**: Filter data by state and city for focused analysis
- **Responsive Design**: Works seamlessly across desktop and mobile devices
- **Real-time Updates**: Dynamic updates based on user selections

### 🌿 Global Biodiversity Dashboard

- **Biodiversity Statistics**: Track total species, families, hotspots, and threatened species
- **Interactive Controls**: Filter by plant family and conservation status
- **Global Hotspots Map**: Explore biodiversity hotspots worldwide with interactive mapping
- **Conservation Charts**: Visualize plant families and conservation status distributions
- **Species Details**: Detailed information about plant families and characteristics
- **Data Export**: Download biodiversity datasets in CSV format
- **Enhanced Styling**: Beautiful cards with animations and responsive design

### 🌐 Network Visualization [NEW]

- **Interactive Network Maps**: Explore relationships between species, hotspots, and conservation statuses
- **Multiple Visualization Types**: Choose between network, density, and cluster density visualizations
- **Customizable Display**: Adjust node sizes, distances, and color schemes
- **Pattern Discovery**: Identify important connections and clusters in biodiversity data
- **VOSviewer Integration**: Powered by advanced network visualization technology
- **See more details**: [VOSviewer Integration Documentation](README_VOSVIEWER.md)

### Core Dashboard Features

- **Interactive Visualizations**: Dynamic charts and maps powered by Plotly
- **Real-time Data Processing**: Efficient data handling and updates
- **Modern UI/UX**: Clean, intuitive interface design
- **Cross-platform Compatibility**: Works on Windows, macOS, and Linux
- **Conservation Insights**: IUCN conservation status tracking and analysis
- **Biodiversity Hotspots**: Global hotspot mapping and metrics
- **Species Explorer**: Detailed plant family and species information

### Scalability & Integration

- **Configurable Architecture**: Easy customization through `config.py`
- **REST API**: Full API access for data integration
- **Multiple Export Formats**: CSV, JSON, Excel export capabilities
- **Environment-based Configuration**: Development, production, and testing configs
- **Modular Design**: Easy to extend and integrate into other projects

### Data Management

- **Automated Data Updates**: Scheduled data fetching from multiple sources
- **Data Validation**: Built-in data quality checks
- **Multiple Data Sources**: GBIF, conservation databases, research datasets
- **Flexible Data Loading**: Support for various file formats and patterns

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd us-tree-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Update datasets** (optional)
   ```bash
   python data_updater.py
   ```

4. **Run the dashboard**
   ```bash
   python app.py
   ```

5. **Access the dashboard**
   - Dashboard: http://localhost:8050
   - API (if enabled): http://localhost:8050/api/v1

## ⚙️ Configuration

The dashboard is highly configurable through the `config.py` file:

### Basic Configuration
```python
from config import config

# App settings
config.APP_NAME = "PlantsWorld"
config.APP_TITLE = "🌿 PlantsWorld Dashboard"
config.DEBUG = True
config.PORT = 8050
```

### Environment Variables
```bash
# Server configuration
export DEBUG=True
export HOST=0.0.0.0
export PORT=8050

# API configuration
export API_ENABLED=True
export CORS_ENABLED=True

# Data configuration
export AUTO_UPDATE=False
export UPDATE_INTERVAL_HOURS=24
export MAX_RECORDS=10000
```

### Dataset Configuration
Add new datasets by updating `DATASETS_CONFIG` in `config.py`:

```python
DATASETS_CONFIG = {
    'my_dataset': {
        'filename_pattern': 'my_data_*.csv',
        'display_name': 'My Dataset',
        'description': 'Description of my dataset'
    }
}
```

## 🔌 API Usage

When API is enabled, the following endpoints are available:

### Health Check
```bash
GET /api/v1/health
```

### List Datasets
```bash
GET /api/v1/datasets
```

### Get Dataset
```bash
GET /api/v1/datasets/{dataset_key}?page=1&per_page=100
```

### Export Dataset
```bash
GET /api/v1/datasets/{dataset_key}/export?format=csv
```

### Search Data
```bash
GET /api/v1/search?q=search_term&dataset=dataset_key
```

### Statistics
```bash
GET /api/v1/stats
```

## 📊 Data Sources

The dashboard integrates data from multiple sources:

- **GBIF (Global Biodiversity Information Facility)**: Species occurrence data
- **IUCN Red List**: Conservation status information
- **Biodiversity Hotspots**: Conservation priority areas
- **Plant Families Database**: Taxonomic information

## 🛠️ Development

### Project Structure
```
us-tree-dashboard/
├── app.py              # Main dashboard application
├── config.py           # Configuration management
├── api.py              # REST API implementation
├── data_updater.py     # Data fetching and updating
├── requirements.txt    # Python dependencies
├── data/              # Data storage directory
├── logs/              # Application logs
└── tests/             # Test suite
```

### Adding New Features

1. **New Visualizations**: Add to the `create_layout()` method in `app.py`
2. **New Data Sources**: Update `data_updater.py` and `config.py`
3. **New API Endpoints**: Extend the `api.py` module
4. **Custom Styling**: Modify the CSS in the layout components

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t plantsworld .

# Run container
docker run -p 8050:8050 plantsworld
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn --bind 0.0.0.0:8050 app:server

# With environment variables
export FLASK_ENV=production
export DEBUG=False
gunicorn --bind 0.0.0.0:8050 app:server
```

### Environment Configurations

- **Development**: `config.DevelopmentConfig`
- **Production**: `config.ProductionConfig`
- **Testing**: `config.TestingConfig`

## 🔧 Integration with Other Projects

### As a Standalone Service
```python
from app import app, server

# Use the Dash app
if __name__ == '__main__':
    app.run_server()
```

### As an API Service
```python
from api import create_api_app

api_app = create_api_app()
api_app.run()
```

### As a Module
```python
from app import PlantsWorldDashboard
from config import config

# Create dashboard instance
dashboard = PlantsWorldDashboard()
data = dashboard.load_datasets()
```

## 📈 Performance Optimization

- **Data Caching**: Datasets are cached in memory
- **Pagination**: API endpoints support pagination
- **Lazy Loading**: Data loaded on demand
- **Configurable Limits**: Maximum records per dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌱 Environmental Impact

This dashboard is designed to support conservation efforts and environmental research. By providing easy access to biodiversity data, we aim to contribute to the protection of our planet's plant life.

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the configuration options

---

**Made with 💚 for our planet's biodiversity**
