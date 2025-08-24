# Code Map

```
.
├── app.py                 # Dash entry point
├── api.py                 # Flask REST API
├── src/
│   ├── components/        # Reusable UI components (Dash)
│   ├── conservation_alerts.py
│   ├── conservation_dashboard_enhancer.py
│   ├── gdrive_integration.py
│   ├── ml_utils.py        # ML helpers
│   └── ...                # Monitoring, metrics, EnhanceX integration
├── tests/
│   ├── test_cleanup.py
│   ├── test_enhancex.py
│   ├── test_ml_utils.py
│   └── performance/       # Performance tests
└── data/                  # Sample datasets
```

## Responsibilities
- **Dashboard**: `app.py`, components in `src/components`.
- **API**: `api.py` + dataset loading in `config.py`.
- **ML & Monitoring**: `src/ml_utils.py`, `src/monitoring/**`.
- **Data & Utilities**: `src/prepare_data.py`, `src/cleanup.py`.

## Dependencies
- Python 3.11+ with Dash, Flask, Pandas, GeoPandas.
- Optional: OpenAI client, Google APIs, Locust for load tests.

