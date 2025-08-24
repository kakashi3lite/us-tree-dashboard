# API Catalog

All endpoints are prefixed with `/api/v1` when `API_ENABLED` is true.

| Endpoint | Method | Description | Query Params | Auth |
|----------|--------|-------------|--------------|------|
| `/health` | GET | Health check | – | none |
| `/datasets` | GET | List available datasets | – | none |
| `/datasets/<dataset_key>` | GET | Paginated dataset records | `page`, `per_page` | none |
| `/datasets/<dataset_key>/export` | GET | Download dataset | `format` (csv, json, xlsx) | none |
| `/stats` | GET | Summary stats for loaded datasets | – | none |
| `/search` | GET | Search across datasets | `q`, `dataset` | none |

Responses are JSON except for `/export`, which streams files.
No authentication or rate limiting is implemented.

