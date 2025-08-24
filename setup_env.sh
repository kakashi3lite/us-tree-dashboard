#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

#----------------------------------------
# 1. Verify Python 3.11 or higher is available
#----------------------------------------

# Extract the major.minor version from python3 --version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

# Compare Python version using Python itself to avoid parsing issues
python3 - <<'PYCHECK'
import sys
major, minor = sys.version_info[:2]
if major < 3 or (major == 3 and minor < 11):
    sys.exit('Python 3.11 or higher is required.')
PYCHECK

#----------------------------------------
# 2. Create (if missing) and activate a virtual environment in ./venv
#----------------------------------------
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source "venv/bin/activate"

#----------------------------------------
# 3. Upgrade pip and install required packages
#----------------------------------------
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

#----------------------------------------
# 4. Run tests and exit on first failure
#----------------------------------------
pytest --maxfail=1 --disable-warnings -q

#----------------------------------------
# 5. Success message
#----------------------------------------
echo "Environment setup complete. Launch the app with: python app.py"
