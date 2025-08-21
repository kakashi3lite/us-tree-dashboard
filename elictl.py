#!/usr/bin/env python
"""Eli Command Line (elictl) - Operational helper for Conservation Dashboard.

Focus: Simple, dependency-light commands that wrap common SRE / DevOps tasks.

Usage examples:
  python elictl.py deploy plan --env prod
  python elictl.py deploy status --env prod
  python elictl.py metrics compare --window 15m
  python elictl.py scale web +2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import List

def _run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[elictl] ➤ Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True)

def cmd_deploy(args: argparse.Namespace):
    if args.action == "plan":
        os.chdir("infra/terraform")
        _run(["terraform", "init", "-input=false"])
        _run(["terraform", "plan", f"-var=environment={args.env}"])
    elif args.action == "apply":
        os.chdir("infra/terraform")
        _run(["terraform", "apply", "-auto-approve", f"-var=environment={args.env}"])
    elif args.action == "status":
        print("Deployment status (placeholder) – integrate with ECS / GitHub releases.")
    elif args.action == "rollback":
        print("Rollback not yet implemented – would retarget ECS service to previous task def.")

def cmd_metrics(args: argparse.Namespace):
    if args.action == "compare":
        window = args.window
        print(f"Comparing metrics window: {window}")
        # Placeholder diff output
        baseline_latency = 0.42
        current_latency = 0.47
        delta = current_latency - baseline_latency
        print(json.dumps({
            "p95_latency_baseline": baseline_latency,
            "p95_latency_current": current_latency,
            "delta": delta,
            "status": "ok" if delta < 0.15 else "investigate"
        }, indent=2))

def cmd_scale(args: argparse.Namespace):
    target = args.component
    change = args.change
    print(f"Scaling component '{target}' by {change} (placeholder). Update ECS desired count here.")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="elictl", description="Conservation Dashboard Ops CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # deploy
    p_deploy = sub.add_parser("deploy", help="Deployment operations")
    p_deploy.add_argument("action", choices=["plan", "apply", "status", "rollback"]) 
    p_deploy.add_argument("--env", default="dev")
    p_deploy.set_defaults(func=cmd_deploy)

    # metrics
    p_metrics = sub.add_parser("metrics", help="Metrics & SLO utilities")
    p_metrics.add_argument("action", choices=["compare"]) 
    p_metrics.add_argument("--window", default="15m")
    p_metrics.set_defaults(func=cmd_metrics)

    # scale
    p_scale = sub.add_parser("scale", help="Scale service components")
    p_scale.add_argument("component", help="Component name, e.g. web")
    p_scale.add_argument("change", help="Change, e.g. +2 or -1")
    p_scale.set_defaults(func=cmd_scale)

    return p

def main(argv: List[str] | None = None):  # pragma: no cover - thin wrapper
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":  # pragma: no cover
    main()