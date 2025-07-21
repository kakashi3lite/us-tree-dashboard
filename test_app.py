#!/usr/bin/env python3
"""
Simple test script to debug the dashboard startup
"""

import sys
from pathlib import Path

print("Python version:", sys.version)
print("Current working directory:", Path.cwd())

try:
    print("\n=== Testing imports ===")
    import dash
    print("[OK] Dash imported successfully")
    
    import pandas as pd
    print("[OK] Pandas imported successfully")
    
    import numpy as np
    print("[OK] Numpy imported successfully")
    
    import plotly.express as px
    print("[OK] Plotly imported successfully")
    
    from src.charitable_orgs import CHARITABLE_ORGS
    print(f"[OK] Charitable orgs imported successfully ({len(CHARITABLE_ORGS)} orgs)")
    
    print("\n=== Testing app creation ===")
    app = dash.Dash(__name__)
    print("[OK] Dash app created successfully")
    
    print("\n=== Testing data generation ===")
    # Test the TreeDashboard class
    sys.path.append('.')
    from app import TreeDashboard
    dashboard = TreeDashboard()
    print(f"[OK] Dashboard created with {len(dashboard.df_trees)} trees and {len(dashboard.df_canopy)} canopy records")
    
    print("\n=== All tests passed! ===")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()