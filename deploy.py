#!/usr/bin/env python3
"""
Production Deployment Script
Handles production deployment setup and validation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True, capture_output=True)
        print("✅ All requirements satisfied")
        return True
    except subprocess.CalledProcessError:
        print("❌ Missing or conflicting requirements")
        print("Run: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup production environment"""
    print("🔍 Setting up environment...")
    
    # Create .env from template if it doesn't exist
    env_file = Path(".env")
    env_template = Path("env.production")
    
    if not env_file.exists() and env_template.exists():
        shutil.copy(env_template, env_file)
        print("✅ Created .env file from template")
        print("📝 Please edit .env file with your production settings")
    elif env_file.exists():
        print("✅ Environment file exists")
    else:
        print("⚠️ No environment template found")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Logs directory ready")
    
    # Ensure data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()
        print("⚠️ Created empty data directory - please add your data files")
    else:
        print("✅ Data directory exists")
    
    return True

def validate_production_config():
    """Validate production configuration"""
    print("🔍 Validating production configuration...")
    
    # Check for debug mode
    debug = os.getenv('DEBUG', 'True').lower()
    if debug == 'true':
        print("⚠️ DEBUG is enabled - consider disabling for production")
    else:
        print("✅ DEBUG is disabled")
    
    # Check for secret key
    secret_key = os.getenv('SECRET_KEY')
    if not secret_key or secret_key == 'your-secret-key-here':
        print("⚠️ SECRET_KEY not set or using default value")
    else:
        print("✅ SECRET_KEY is configured")
    
    # Check for Mapbox token
    mapbox_token = os.getenv('MAPBOX_TOKEN')
    if not mapbox_token or mapbox_token == 'your_mapbox_token_here':
        print("⚠️ MAPBOX_TOKEN not set - maps may not work properly")
    else:
        print("✅ MAPBOX_TOKEN is configured")
    
    return True

def run_production_tests():
    """Run production tests"""
    print("🔍 Running production tests...")
    try:
        result = subprocess.run([sys.executable, "production_test.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All production tests passed")
            return True
        else:
            print("❌ Some production tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Could not run production tests: {e}")
        return False

def create_docker_assets():
    """Create Docker assets for containerized deployment"""
    print("🔍 Creating Docker assets...")
    
    # Enhanced Dockerfile
    dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8050/ || exit 1

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "4", "--timeout", "120", "app:server"]
'''
    
    with open("Dockerfile.production", "w") as f:
        f.write(dockerfile_content)
    print("✅ Created production Dockerfile")
    
    # Docker compose for production
    compose_content = '''version: '3.8'

services:
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8050:8050"
    environment:
      - DEBUG=False
      - HOST=0.0.0.0
      - PORT=8050
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8050/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - dashboard
    restart: unless-stopped
'''
    
    with open("docker-compose.production.yml", "w") as f:
        f.write(compose_content)
    print("✅ Created production docker-compose")
    
    return True

def create_nginx_config():
    """Create nginx configuration for reverse proxy"""
    nginx_config = '''events {
    worker_connections 1024;
}

http {
    upstream dashboard {
        server dashboard:8050;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        location / {
            proxy_pass http://dashboard;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Static assets
        location /assets/ {
            proxy_pass http://dashboard;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
'''
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    print("✅ Created nginx configuration")

def main():
    """Main deployment function"""
    print("🚀 Production Deployment Setup")
    print("=" * 50)
    
    steps = [
        ("Check Requirements", check_requirements),
        ("Setup Environment", setup_environment),
        ("Validate Configuration", validate_production_config),
        ("Run Tests", run_production_tests),
        ("Create Docker Assets", create_docker_assets),
        ("Create Nginx Config", create_nginx_config),
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 30)
        try:
            if step_func():
                success_count += 1
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Deployment Setup: {success_count}/{len(steps)} steps completed")
    
    if success_count == len(steps):
        print("🟢 DEPLOYMENT READY!")
        print("\n📋 Next steps:")
        print("1. Edit .env with your production settings")
        print("2. Add SSL certificates to ./ssl/ directory")
        print("3. Update nginx.conf with your domain")
        print("4. Run: docker-compose -f docker-compose.production.yml up -d")
        print("5. Monitor logs: docker-compose logs -f")
    else:
        print("🔴 DEPLOYMENT NOT READY - Please fix the issues above")

if __name__ == "__main__":
    main()
