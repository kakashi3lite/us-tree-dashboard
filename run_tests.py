"""Deterministic test runner ensuring external pytest plugins are disabled.

Sets PYTEST_DISABLE_PLUGIN_AUTOLOAD early (before importing pytest) to prevent
globally installed plugins like `deepeval` / `ollama` from interfering. Relies
on `pytest.ini` for coverage / addopts configuration.
"""
from __future__ import annotations
import os
import sys
import subprocess


def main() -> int:
  # Disable auto plugin loading before pytest import/exec.
  os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
  # Provide default Mapbox token for map rendering code paths.
  os.environ.setdefault("MAPBOX_TOKEN", "test-token")

  # Explicitly load coverage plugin (pytest-cov) since autoload is disabled.
  cmd = [sys.executable, "-m", "pytest", "-p", "pytest_cov"]
  print("Running:", " ".join(cmd))
  result = subprocess.run(cmd)
  return result.returncode


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
