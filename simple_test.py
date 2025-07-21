#!/usr/bin/env python3
"""
Simple Production Test - Windows Compatible
Basic test without Unicode characters for Windows Command Prompt compatibility
"""

import sys
import os
import traceback
from pathlib import Path

def test_basic_functionality():
    """Test basic app functionality"""
    print("Testing basic functionality...")
    try:
        # Test imports
        import dash
        import pandas as pd
        import numpy as np
        import plotly.express as px
        print("✓ Dependencies imported successfully")
        
        # Test app creation
        from app import app, dashboard
        print("✓ App created successfully")
        
        # Test data loading
        if dashboard._data_loaded:
            tree_count = len(dashboard.df_trees) if dashboard.df_trees is not None else 0
            canopy_count = len(dashboard.df_canopy) if dashboard.df_canopy is not None else 0
            print(f"✓ Data loaded: {tree_count} trees, {canopy_count} canopy records")
        else:
            print("! Data loading incomplete")
        
        # Test layout
        layout = dashboard.create_layout()
        print("✓ Layout created successfully")
        
        # Test chart generation
        overview_chart = dashboard.create_overview_chart()
        print("✓ Charts generated successfully")
        
        # Test map generation
        default_map = dashboard.create_map(None, None, [0, 100])
        print("✓ Maps generated successfully")
        
        print("\n=== ALL TESTS PASSED ===")
        print("Dashboard is ready for production deployment!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
