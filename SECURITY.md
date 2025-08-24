# Security

## ASVS Level 2 Checklist

| Control | Status | Notes |
|---------|--------|-------|
| Authentication enforced | N/A | Public dashboard has no login |
| Authorization per role | N/A | No user accounts |
| Session management | N/A | Stateless client requests |
| Input validation | Gap | API search uses simple string matching; no sanitization |
| Output encoding | Pass | Flask/Dash escape by default |
| Cryptographic storage | N/A | No secrets persisted server-side |
| Transport security (HTTPS) | Gap | Rely on deployment platform to enforce TLS |
| HTTP security headers | Gap | No CSP/HSTS configuration in `app.py` |
| Error handling | Pass | API returns JSON errors with codes |
| Logging & monitoring | Pass | Uses `logging` and performance monitors |
| Dependency scanning | Pass | `.github/workflows/security.yml` runs safety and bandit |
| Secrets management | Pass | Gitleaks configured via `security.yml` |
| Rate limiting | Gap | API lacks throttling |
| SSRF protection | N/A | No outbound user-controlled URLs |
| Deserialization safety | Pass | No unsafe deserialization used |

## Secret Handling
- API keys loaded via environment variables; sample `.env.example` provided.
- `security.yml` workflow scans for leaked secrets.

## Dependency Risk Hotspots
- Geospatial packages (`geopandas`, `gdal`) require native libs—ensure patched versions.

