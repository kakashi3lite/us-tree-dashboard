---
applyTo: "Dockerfile* docker-compose*.yml .github/workflows/**"
---
- Pin versions of base images and GitHub Actions.
- Use multi-stage builds to keep images slim.
- Validate `docker-compose` configs with `docker compose config`.
