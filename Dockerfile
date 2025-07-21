FROM python:3.9-slim AS builder
WORKDIR /app

# Install system dependencies for geopandas
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim AS runner
WORKDIR /app

# Install runtime dependencies for geopandas
RUN apt-get update && apt-get install -y \
    libgdal28 \
    libproj19 \
    libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY . .

EXPOSE 8050
CMD ["python", "app.py"]
