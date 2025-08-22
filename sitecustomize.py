"""Early test environment customization.
Ensures external pytest plugins (deepeval, ollama, etc.) are not auto-loaded
which otherwise cause environment-dependent failures in CI/minimal envs.
"""
import os, sys, types

# Force disable autoload (override any pre-set value) before pytest plugin discovery
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

# Provide lightweight stubs for problematic third-party pytest auto plugins if import attempted
for _mod in ("deepeval", "ollama"):
	if _mod not in sys.modules:
		m = types.ModuleType(_mod)
		# Minimal attributes some plugins may introspect
		m.__dict__.update({
			"__all__": [],
			"__version__": "0.0.0-stub",
		})
		sys.modules[_mod] = m
