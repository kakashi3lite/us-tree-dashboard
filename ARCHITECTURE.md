# Architecture

## Components
```mermaid
graph TD
  A[Browser] --> B[Dash App (app.py)]
  B --> C[Flask Server]
  C --> D[Data Files]
  C --> E[ML & Monitoring Modules]
  C --> F[REST API /api.py]
```

## Request Flow
```mermaid
sequenceDiagram
  participant U as User
  participant D as Dash App
  participant S as Server
  participant Data as CSV Files
  U->>D: HTTP Request
  D->>S: Callback
  S->>Data: Load/Update
  Data-->>S: DataFrame
  S-->>D: JSON/Graph Data
  D-->>U: Rendered Visualization
```

## Deployment
- Docker image defined in `Dockerfile` (Python 3.11).
- `docker-compose.yml` for local services.
- GitHub Actions build & test workflows.

