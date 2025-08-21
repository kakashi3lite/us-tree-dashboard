# Runbook: Service Degradation (Latency / 5xx Increase)

Severity: SEV2 if partial, SEV1 if >50% users impacted or hard down.

## Detection

- Alert sources: Prometheus alert `HighRequestLatency`, `HTTP5xxSpike`.
- SLO: p95 latency < 800ms, Error rate < 1% 5xx over 5m window.

## Immediate Actions (First 5 Minutes)

1. Acknowledge alert in PagerDuty.
2. Check current deployment status (was there a canary / rollout?).

	```bash
	elictl deploy status --env prod
	```

3. Compare last stable build metrics.

	```bash
	elictl metrics compare --window 15m
	```

4. If spike coincides with deployment within last 15m -> initiate automated rollback.

	```bash
	elictl deploy rollback --env prod --strategy canary
	```

## Diagnostics

| Layer | Command | Purpose |
|-------|---------|---------|
| Load Balancer | `aws elbv2 describe-target-health ...` | Target health & codes |
| App Logs | `elictl logs tail --component web --since 5m` | Error patterns |
| Traces | UI: /tracing | Slow spans identification |
| DB (if used) | `aws rds describe-db-instances` | CPU / connections |

Check Prometheus directly:

```bash
curl -s "$PROM_URL/api/v1/query?query=rate(dashboard_requests_total[5m])"
```

## Decision Tree

- Deployment related? (errors started after rollout) -> Rollback.
- Resource saturation? (CPU > 80%, memory > 85%) -> Scale out: `elictl scale web +2`.
- External dependency latency? -> Enable circuit breaker / degrade non-critical features.

## Rollback Criteria

- After rollback, metrics return to baseline within 10m.
- If not, escalate to SEV1 and page platform lead.

## Post-Incident

1. Capture timeline in incident doc.
2. Open root cause ticket with span IDs / log references.
3. Add regression test or guardrail (load test threshold) before next deploy.

## Key Dashboards

- Grafana: `Conservation / Web Latency`, `Conservation / Error Rates`, `Conservation / Canopy Pipeline`.

## Automation Hooks

- Auto-rollback triggers if: p95 latency > 1200ms for 5m OR 5xx rate >2% for 3m during deployment window.
