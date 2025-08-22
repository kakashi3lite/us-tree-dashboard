import sys, os

# Disable auto-loading external pytest plugins that introduce environment errors
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

