# US Tree Dashboard 🌳

A scalable, interactive dashboard for visualizing and analyzing urban tree data across the United States. Built with Flask, React, PostGIS, and Redis for optimal performance and user experience.

## Features

### Core Functionality
- 🗺️ Interactive map visualization with dynamic clustering
- 📊 Real-time data visualization with D3.js
- 🌱 Species-based filtering and analysis
- 📈 Historical growth trends
- 🌍 Environmental impact calculations

### Technical Highlights
- 🚀 Optimized PostGIS queries with spatial indexing
- 💾 Redis caching for frequent queries
- ⚡ Async API endpoints with rate limiting
- 📱 Responsive design for all devices
- 🔍 Advanced monitoring and error tracking

## Architecture

### Backend
- **Framework**: Flask with async support
- **Database**: PostgreSQL with PostGIS extension
- **Caching**: Redis
- **API**: RESTful with rate limiting

### Frontend
- **Framework**: React with Material-UI
- **Map**: Leaflet.js with custom clustering
- **Charts**: D3.js with responsive layouts
- **State Management**: React Query

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Reverse Proxy**: Nginx
- **Monitoring**: Prometheus & Custom Metrics
- **Logging**: JSON-formatted with rotation

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Git
- Make (optional, for using Makefile commands)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kakashi3lite/us-tree-dashboard.git
cd us-tree-dashboard
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Build and start services:
```bash
docker-compose up --build
```

4. Initialize database:
```bash
docker-compose exec db psql -U postgres -d us_tree_dashboard -f /docker-entrypoint-initdb.d/init.sql
```

The application will be available at:
- Frontend: http://localhost:80
- API: http://localhost:8050/api
- Metrics: http://localhost:8050/metrics

## Development

### Local Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run development server:
```bash
FLASK_ENV=development python app.py
```

### Testing

Run tests with pytest:
```bash
pytest tests/
```

### Code Style

Format code with black:
```bash
black .
```

## API Documentation

### Endpoints

#### GET /api/tree-clusters
Get clustered tree locations within specified bounds.

Parameters:
- `min_lat`: Minimum latitude
- `max_lat`: Maximum latitude
- `min_lon`: Minimum longitude
- `max_lon`: Maximum longitude
- `zoom`: Zoom level (default: 12)

#### GET /api/tree-density
Get tree density statistics by region.

Parameters:
- `region_id`: Optional region ID

#### GET /api/environmental-impact
Calculate environmental impact metrics.

Parameters:
- `region_id`: Optional region ID
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)

#### GET /api/historical-trends
Get historical trends for specified metrics.

Parameters:
- `metric`: One of ['height', 'diameter', 'canopy_width']
- `interval`: One of ['day', 'week', 'month', 'quarter', 'year']
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)

## Monitoring

### Metrics

The application exposes Prometheus metrics at `/metrics` including:
- API request counts and durations
- Query performance metrics
- Cache hit/miss ratios
- Active user counts

### Logging

Logs are written to `logs/us_tree_dashboard.log` in JSON format with the following information:
- Timestamp
- Log level
- Environment
- Message
- Additional context

## Deployment

### Production Setup

1. Configure environment variables in `.env`:
```env
FLASK_ENV=production
ENABLE_PROMETHEUS=true
ENABLE_RATE_LIMITING=true
ENABLE_CACHING=true
```

2. Configure SSL in nginx.conf:
```nginx
ssl_certificate /etc/nginx/ssl/cert.pem;
ssl_certificate_key /etc/nginx/ssl/key.pem;
```

3. Deploy with Docker Compose:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Scaling

To scale the application:

1. Adjust PostgreSQL configuration:
```sql
ALTER SYSTEM SET max_connections = '200';
ALTER SYSTEM SET shared_buffers = '1GB';
```

2. Configure Redis persistence:
```conf
save 900 1
save 300 10
save 60 10000
```

3. Scale application containers:
```bash
docker-compose up -d --scale app=3
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

kakashi3lite (swanandtanavade100@gmail.com)

## Acknowledgments

- OpenStreetMap for map data
- D3.js community for visualization examples
- Flask and React communities for excellent documentation