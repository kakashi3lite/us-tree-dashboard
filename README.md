# 🌳 US Tree Dashboard

An interactive dashboard for analyzing US tree data with machine learning insights. This project combines geospatial analysis using GDAL with modern web technologies and AI-powered analytics to provide comprehensive visualization and analysis of tree distribution across the United States.

## 🎯 Features

### 🤖 AI-Powered Analysis
- Intelligent pattern recognition in tree distribution
- Species diversity insights with ecological context
- Smart health trend analysis with recommendations
- Environmental impact predictions

### 🗺️ Interactive Visualization
- Real-time tree location mapping with interactive filters
- County-level canopy coverage analysis
- Dynamic species distribution charts
- Tree health status visualization

### 📊 Data Analysis Capabilities
- City-level tree inventories
- State and county-level canopy coverage metrics
- Species diversity analysis
- Tree health distribution statistics

## 🛠️ Technology Stack
- Frontend: Dash, Plotly
- Backend: Python, Flask (via Dash)
- Data Processing: Pandas, GeoPandas
- Deployment: Docker, Gunicorn

## 📦 Installation

### Prerequisites
- Docker and Docker Compose
- Git
- 4GB RAM minimum (8GB recommended)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/kakashi3lite/us-tree-dashboard.git
cd us-tree-dashboard
```

2. Set up environment variables:
```bash
# Windows
copy .env.example .env
# Linux/Mac
cp .env.example .env
```

3. Start the dashboard:
```bash
docker-compose up --build -d
```

4. (Optional) Start Jupyter environment:
```bash
docker-compose -f docker-compose-jupyter.yml up --build -d
```

## 🚀 Local Development

1. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to: http://127.0.0.1:8050

## 🧪 Testing
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
pytest
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## 📧 Contact
Swanand Tanavade - swanandtanavade100@gmail.com

Project Link: [https://github.com/kakashi3lite/us-tree-dashboard](https://github.com/kakashi3lite/us-tree-dashboard)