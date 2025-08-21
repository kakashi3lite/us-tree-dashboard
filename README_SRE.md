# SRE & MLOps Hardening (v2.1.0 Draft)

This document catalogs production-readiness enhancements layering on top of core conservation features (v2.0.0).

## Added Components

- Observability: Prometheus metrics endpoint `/metrics`, health endpoints `/healthz` + `/api/v1/health`, OpenTelemetry traces (OTLP HTTP exporter).
- IaC: Terraform skeleton (`infra/terraform/`) provisioning VPC, subnets, ECS cluster, ALB, security groups, log groups, S3 buckets.
- Runbooks: `RUNBOOKS/` with incident response starting set.
- CLI: `elictl.py` operational helper (plan/apply/status/metrics/scale).
- Cost Model: `COSTING.md` baseline monthly estimate & optimization levers.

## Next Steps (Planned)

| Area | Action | Status |
|------|--------|--------|
| Deployment | Canary & auto-rollback workflow | Planned |
| Security | Add Checkov + tfsec scanning job | Planned |
| Secrets | Integrate AWS Secrets Manager for runtime config | Planned |
| Tracing | Add span annotations around data pipeline & alert engine | Planned |
| Synthetic | Add scheduled probe workflow hitting critical UI paths | Planned |
| SLOs | Codify p95 latency & error budget policies | Planned |

## SLO Draft

- Availability: 99.5% monthly (allowed downtime ~3h 39m)
- Latency: p95 < 800ms (web), p99 < 1.3s
- Error Rate: 5xx < 1% of requests (rolling 5m)

## Alert Policy (High-Level)

- Page: Error budget burn >20% in 1h OR p95 > 1.2s for 10m OR health status=degraded for 3 consecutive checks.
- Ticket: Cost anomaly >30% week-over-week.

## Verification Checklist

| Item | Status |
|------|--------|
| Metrics scrape functional | ✅ |
| Health endpoints return expected JSON | ✅ |
| Terraform plan runs without error (incomplete state backend vars) | ⚠️ Needs backend bucket/table |
| CLI basic commands run | ✅ |
| Runbooks lint clean | In progress |

## Usage

Plan infrastructure:

```bash
python elictl.py deploy plan --env dev
```

Scrape metrics locally:

```bash
curl -s http://localhost:8050/metrics | head
```

Health check:

```bash
curl -s http://localhost:8050/api/v1/health | jq
```

## Notes

Further hardening will extend workflows for security scanning, drift detection, and automated rollback gating on SLO guardrails.
