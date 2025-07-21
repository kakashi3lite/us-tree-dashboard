# 🚀 PRODUCTION DEPLOYMENT GUIDE

## Quick Start

### Development Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python simple_test.py

# Start development server
python app.py
```

### Production Deployment

#### Option 1: Docker (Recommended)
```bash
# Setup production environment
python deploy.py

# Configure environment
cp env.production .env
# Edit .env with your production settings

# Deploy with Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Monitor
docker-compose logs -f dashboard
```

#### Option 2: Manual Deployment
```bash
# Install production dependencies
pip install -r requirements.txt
pip install gunicorn

# Set environment variables
export DEBUG=False
export HOST=0.0.0.0
export PORT=8050

# Run with Gunicorn
gunicorn --bind 0.0.0.0:8050 --workers 4 --timeout 120 app:server
```

## Configuration

### Environment Variables
- `DEBUG`: Set to `False` for production
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8050)
- `MAPBOX_TOKEN`: Optional Mapbox API token for enhanced maps
- `SECRET_KEY`: Required for production security

### Data Files
Ensure these files exist in the `data/` directory:
- `gbif_plants_20250614.csv`
- `plant_families_20250614.csv`
- `conservation_status_20250614.csv`
- `biodiversity_hotspots_20250614.csv`

If data files are missing, the dashboard will use sample data automatically.

## Monitoring and Maintenance

### Health Checks
- Dashboard health: `http://your-domain.com/`
- Docker health: `docker-compose ps`

### Logs
- Application logs: `docker-compose logs dashboard`
- Nginx logs: `docker-compose logs nginx`

### Performance Monitoring
- Response times: Check application logs
- Memory usage: `docker stats`
- Error rates: Monitor error logs

## Support and Troubleshooting

### Common Issues
1. **Data not loading**: Check data files in `data/` directory
2. **Maps not working**: Set `MAPBOX_TOKEN` environment variable
3. **Slow performance**: Ensure adequate memory allocation

### Getting Help
- Check logs for error messages
- Run `python simple_test.py` to verify installation
- Review the BUG_FIXES_REPORT.md for known issues

## Security Notes
- Always set `DEBUG=False` in production
- Use strong `SECRET_KEY` values
- Configure SSL/HTTPS in production
- Keep dependencies updated

---

✅ **Status: Production Ready**
🔧 **Last Updated**: 2025-07-20
📋 **Version**: v1.0.0-production
