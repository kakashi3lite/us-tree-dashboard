# US Tree Dashboard

## Overview
A sophisticated data visualization dashboard that provides interactive insights into US tree data using advanced machine learning techniques. The application is built with a focus on scalability, reliability, and maintainable code architecture.

## Features
- Interactive visualization of environmental metrics
- ML model performance monitoring
- Real-time system performance metrics
- Robust error handling and logging
- Containerized deployment support
- Comprehensive test coverage

## Technical Stack
- **Backend Framework**: Python with Dash and Flask
- **Data Processing**: Pandas, GeoPandas
- **Visualization**: Plotly
- **Testing**: PyTest with coverage reporting
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **Logging**: Python's built-in logging with rotation

## Prerequisites
- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- GDAL library (for GeoPandas support)

## Local Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the application:
   ```bash
   python app.py
   ```

## Docker Deployment

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Access the application at `http://localhost:8050`

## Testing

Run the test suite:
```bash
pytest
```

Generate coverage report:
```bash
pytest --cov=src --cov-report=html
```

## Project Structure
```
├── src/
│   ├── config.py           # Application configuration
│   ├── error_handlers.py   # Error handling middleware
│   └── metrics/            # Metric components
├── tests/                  # Test suite
├── app.py                  # Main application
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Container orchestration
└── requirements.txt       # Python dependencies
```

## Error Handling
The application implements comprehensive error handling:
- Custom error responses for different scenarios
- Detailed logging with rotation
- Error tracking and monitoring

## Deployment
The application supports multiple deployment options:

### Docker (Recommended)
```bash
docker-compose up -d
```

### Manual Deployment
1. Set up a Python environment
2. Install dependencies
3. Configure environment variables
4. Run with gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8050 app:server
   ```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT

## Author
kakashi3lite (swanandtanavade100@gmail.com)
