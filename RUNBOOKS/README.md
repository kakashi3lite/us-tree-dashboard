# Runbooks

Operational runbooks for the Conservation Dashboard platform. Each runbook is designed for rapid triage (<5 min to first action) and clear rollback paths.

## Index

1. service-degradation.md – Elevated error rates / latency
2. canary-failure.md – Canary deployment rollback
3. data-pipeline-stall.md – Ingestion not progressing
4. cost-anomaly.md – Unexpected AWS bill spike

## Conventions

- All CLI references use the helper script `elictl` (to be added) or native AWS CLI.
- Severity levels: SEV1 (user-visible outage), SEV2 (partial degradation), SEV3 (non-urgent anomaly).
- Time targets: MTTA < 5m, MTTD < 2m (via alerts), MTTR < 30m for SEV1.

## Quick Start

Describe incident in Slack channel #oncall-conservation with template:

```text
[INCIDENT] <short title> | Detected: <timestamp UTC> | Symptoms: <summary> | Impact: <users/regions> | Initial Actions: <taken>
```

Escalation path: On-call -> Platform Lead -> Director.

See individual runbooks for specifics.
