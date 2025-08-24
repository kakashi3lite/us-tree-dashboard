# Project Brief

## Purpose
The PlantsWorld dashboard visualizes plant biodiversity and tree data for U.S. regions. It offers interactive maps, charts, and an optional REST API for data access.

## Domain
Environmental data analytics focused on trees and plant biodiversity. Dash (Flask-based) web app with optional ML utilities and monitoring.

## Key Flows
1. Load datasets from `data/` via `TreeDashboard` in `app.py`.
2. Render interactive dashboard components (maps, stats, network graphs).
3. Optional `api.py` serves datasets and stats under `/api/v1/*`.
4. Monitoring and performance modules track usage and generate reports.

## Public Interfaces
- Dash UI at `/` (via `app.py`).
- REST API (see `API_CATALOG.md`).

## Environments
| Environment | Start Command | Notes |
|-------------|---------------|-------|
| Local dev   | `python app.py` | Requires `pip install -r requirements.txt` |
| Docker      | `docker-compose up` | Uses `Dockerfile` and `docker-compose.yml` |
| Production  | `gunicorn app:server` | Configurable via environment vars |

