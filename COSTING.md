# Cost Model (Initial Draft)

Objective: Maintain annual infra spend < $5k while supporting moderate usage (<= 50 concurrent users, daily data ingests, periodic PDF/report jobs).

## Core Components (Prod, us-east-1)

| Component | Qty / Config | Est. Monthly | Notes |
|-----------|--------------|--------------|-------|
| ECS Fargate Service (web) | 2 tasks x 0.5 vCPU / 1GB | $55 | Spot mix could reduce 60% |
| Application Load Balancer | 1 | $18 | LCU within free tier early |
| S3 (data + logs) | 50GB standard | $1.25 | + requests negligible |
| CloudWatch Logs + Metrics | 5GB ingest + retention | $5 | Optimize retention to 30d |
| OTLP Collector (Fargate) | 1 task 0.25 vCPU/0.5GB | $12 | Could merge into sidecar |
| Terraform State (S3 + DDB) | minimal | $1 | Fixed overhead |
| ECR Storage | 2GB | $0.20 | Prune images > 5 versions |
| Bandwidth (egress) | 20GB | $1.80 | Mostly internal traffic |
| Buffer (15%) | - | $14 | Variability |
| TOTAL | - | ~$108 | ~ $1.3k annual |

## Optimization Levers

1. Enable Fargate Spot for non-critical background tasks (health monitor, pipeline).
2. Autoscale tasks on p95 latency > 800ms (cooldown 5m).
3. Downscale to 1 task during low traffic window (00:00–06:00 UTC) – cron event.
4. Move cold logs >30d to S3 Glacier Instant Retrieval.
5. Merge OTLP collector with web task (shared process) once stable.

## Future Considerations

- If usage > 200 concurrent users sustained, consider EKS + Graviton nodes.
- If ingestion pipeline grows, evaluate serverless step functions for spiky loads.

## Tracking

- Implement monthly cost export (AWS CUR) and parse top 5 deltas into Slack summary.
