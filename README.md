# 🌳 US Tree Dashboard

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![GDAL](https://img.shields.io/badge/gdal-3.6.2-green.svg)
![Dash](https://img.shields.io/badge/dash-2.9.3-blue.svg)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-brightgreen.svg)

An interactive dashboard for analyzing US tree data with machine learning insights. This project combines geospatial analysis using GDAL with modern web technologies and AI-powered analytics to provide comprehensive visualization and analysis of tree distribution across the United States. Built with a focus on performance and usability, it offers both a web interface and Jupyter notebook environment for deep analysis.

## 🎯 Features

### 🤖 AI-Powered Analysis
- Intelligent pattern recognition in tree distribution
- Species diversity insights with ecological context
- Smart health trend analysis with recommendations
- Environmental impact predictions
- AI-guided urban forestry recommendations
- Natural language query interface for data exploration
- Context-aware visualization suggestions

### 🗺️ Interactive Visualization
- Real-time tree location mapping with interactive filters
- County-level canopy coverage analysis with insights
- Dynamic species distribution charts and diversity metrics
- Tree health status visualization with trend analysis
- Advanced navigation with section anchors
- Responsive design for all devices
- Integrated charitable organization information

### 📊 Data Analysis & ML Capabilities
- City-level tree inventories from 63 major US cities
- State and county-level canopy coverage metrics with historical trends
- Species diversity analysis with Shannon diversity index
- Tree health distribution statistics with predictive indicators
- DBH (Diameter at Breast Height) analytics
- Environmental impact assessments
- Interactive Jupyter notebooks for deep analysis
- Machine Learning Components:
  - Tree health prediction using Random Forest models
  - Growth forecasting with environmental factor consideration
  - Environmental impact calculations
  - Optimal planting location recommendations
  - Model persistence and reusability
  - AutoML capabilities with Optuna
  - Experiment tracking with MLflow
  - GPT-4 powered pattern analysis
  - AI-assisted data interpretation
  - Smart outlier detection and analysis

### 🔍 Advanced Filtering
- State and city selection
- Canopy coverage range filtering
- Species filtering
- Health status filtering

## 🛠️ Technology Stack

- **Frontend**: Dash, Plotly
- **Backend**: Python, Flask (via Dash)
- **Data Processing**: Pandas, GeoPandas
- **AI & ML**: OpenAI GPT-4, scikit-learn, LightGBM, XGBoost
- **Deployment**: Gunicorn, Whitenoise
- **Database**: PostgreSQL, SQLAlchemy
- **Caching**: Redis
- **Task Queue**: Celery
- **Maps**: Mapbox

## 📦 Installation

### Prerequisites

- Docker and Docker Compose
- Git
- 4GB RAM minimum (8GB recommended)
- 20GB free disk space
- OpenAI API key for AI features

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/kakashi3lite/us-tree-dashboard.git
   cd us-tree-dashboard
   ```

2. Set up environment variables:
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # Linux/Mac
   ```
   
3. Configure your OpenAI API key in .env:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Start the main dashboard:
   ```bash
   docker-compose up --build -d
   ```

5. (Optional) Start the Jupyter environment for analysis:
   ```bash
   docker-compose -f docker-compose-jupyter.yml up --build -d
   ```

### Environment Setup

The project uses Docker containers with all dependencies pre-configured. However, if you need to modify environment variables:
1. Edit `.env` to configure:
   - Mapbox API token
   - Database credentials
   - Other service configurations

## 🚀 Running the Application

### Google Colab
For quick exploration and analysis:

1. Open our Colab notebook:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kakashi3lite/us-tree-dashboard/blob/main/notebooks/colab_analysis.ipynb)

2. The notebook will automatically:
   - Clone the repository
   - Install dependencies
   - Set up the GDAL environment
   - Load sample data

3. Run the cells to perform:
   - Geographic analysis
   - ML predictions
   - Environmental impact assessment

### Local Development
Start the dashboard using Docker Compose:
```bash
docker-compose up
```
Open your browser and navigate to: `http://127.0.0.1:8050`

### Testing
Install dependencies and run the unit tests:
```bash
pip install -r requirements.txt
pytest
```

### Production Deployment
1. Set environment variables for production
2. Use Gunicorn to run the application:
   ```bash
   gunicorn app:server --preload --workers 4
   ```

## 📊 Data Sources

- **Tree Inventory**: Dryad (5.66M trees across 63 US cities)
  - Comprehensive street tree data
  - Species information
  - Health status
  - Size measurements

- **Canopy Coverage**: USDA Forest Service Tree Canopy Cover (TCC) dataset
  - High-resolution canopy coverage data
  - County-level statistics
  - Regular updates

- **Geographic Data**: US Census Bureau TIGER/Line shapefiles
  - County boundaries
  - State boundaries
  - Geographic reference data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Data provided by Dryad and USDA Forest Service
- US Census Bureau for geographic data
- Mapbox for mapping capabilities
- All contributors and maintainers
