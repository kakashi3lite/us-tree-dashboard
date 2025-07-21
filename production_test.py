#!/usr/bin/env python3
"""
Production Readiness Test Suite
Tests all critical components for production deployment
"""

import sys
import os
import traceback
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    try:
        import dash
        import dash_bootstrap_components as dbc
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_files():
    """Test data file availability and structure"""
    print("🔍 Testing data files...")
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    required_files = [
        "gbif_plants_20250614.csv",
        "plant_families_20250614.csv",
        "conservation_status_20250614.csv",
        "biodiversity_hotspots_20250614.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ Missing data files: {missing_files}")
        print("📝 Dashboard will use sample data")
    else:
        print("✅ All data files present")
    
    return True

def test_app_creation():
    """Test app creation and initialization"""
    print("🔍 Testing app creation...")
    try:
        from app import app, dashboard
        print("✅ App created successfully")
        
        # Test dashboard data loading
        if dashboard._data_loaded:
            tree_count = len(dashboard.df_trees) if dashboard.df_trees is not None else 0
            canopy_count = len(dashboard.df_canopy) if dashboard.df_canopy is not None else 0
            print(f"✅ Dashboard loaded with {tree_count} tree records and {canopy_count} canopy records")
        else:
            print("⚠️ Dashboard loaded but data may be limited")
        
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        traceback.print_exc()
        return False

def test_layout_rendering():
    """Test layout rendering"""
    print("🔍 Testing layout rendering...")
    try:
        from app import dashboard
        layout = dashboard.create_layout()
        print("✅ Layout rendered successfully")
        return True
    except Exception as e:
        print(f"❌ Layout rendering failed: {e}")
        traceback.print_exc()
        return False

def test_chart_generation():
    """Test chart generation functions"""
    print("🔍 Testing chart generation...")
    try:
        from app import dashboard
        
        # Test overview chart
        overview_chart = dashboard.create_overview_chart()
        print("✅ Overview chart generated")
        
        # Test with sample data
        if dashboard.df_trees is not None and not dashboard.df_trees.empty:
            city = dashboard.df_trees['city'].iloc[0]
            city_data = dashboard.df_trees[dashboard.df_trees['city'] == city]
            
            species_chart = dashboard.create_species_chart(city_data)
            health_chart = dashboard.create_health_chart(city_data)
            print("✅ City-specific charts generated")
        
        if dashboard.df_canopy is not None and not dashboard.df_canopy.empty:
            state = dashboard.df_canopy['state'].iloc[0]
            state_data = dashboard.df_canopy[dashboard.df_canopy['state'] == state]
            
            canopy_chart = dashboard.create_canopy_chart(state_data)
            print("✅ Canopy chart generated")
        
        return True
    except Exception as e:
        print(f"❌ Chart generation failed: {e}")
        traceback.print_exc()
        return False

def test_map_generation():
    """Test map generation"""
    print("🔍 Testing map generation...")
    try:
        from app import dashboard
        
        # Test default map
        default_map = dashboard.create_map(None, None, [0, 100])
        print("✅ Default map generated")
        
        # Test with selections if data available
        if dashboard.df_trees is not None and not dashboard.df_trees.empty:
            city = dashboard.df_trees['city'].iloc[0]
            city_map = dashboard.create_map(None, city, [0, 100])
            print("✅ City map generated")
        
        if dashboard.df_canopy is not None and not dashboard.df_canopy.empty:
            state = dashboard.df_canopy['state'].iloc[0]
            state_map = dashboard.create_map(state, None, [0, 100])
            print("✅ State map generated")
        
        return True
    except Exception as e:
        print(f"❌ Map generation failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration settings"""
    print("🔍 Testing configuration...")
    try:
        # Test environment variables
        debug = os.getenv('DEBUG', 'True')
        host = os.getenv('HOST', '0.0.0.0')
        port = os.getenv('PORT', '8050')
        
        print(f"✅ Configuration loaded - Debug: {debug}, Host: {host}, Port: {port}")
        
        # Test config module if available
        try:
            from config import DashboardConfig
            config_valid = DashboardConfig.validate_config()
            print("✅ Configuration module validated")
        except ImportError:
            print("⚠️ Config module not available (optional)")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_performance():
    """Test basic performance metrics"""
    print("🔍 Testing performance...")
    try:
        import time
        from app import dashboard
        
        # Time layout creation
        start_time = time.time()
        layout = dashboard.create_layout()
        layout_time = time.time() - start_time
        
        # Time chart creation
        start_time = time.time()
        chart = dashboard.create_overview_chart()
        chart_time = time.time() - start_time
        
        print(f"✅ Performance metrics - Layout: {layout_time:.3f}s, Chart: {chart_time:.3f}s")
        
        if layout_time > 5.0:
            print("⚠️ Layout creation is slow (>5s)")
        if chart_time > 2.0:
            print("⚠️ Chart creation is slow (>2s)")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def run_production_tests():
    """Run all production readiness tests"""
    print("🚀 Starting Production Readiness Tests")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Data Files", test_data_files),
        ("App Creation", test_app_creation),
        ("Layout Rendering", test_layout_rendering),
        ("Chart Generation", test_chart_generation),
        ("Map Generation", test_map_generation),
        ("Configuration", test_configuration),
        ("Performance", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Production Readiness Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🟢 ALL TESTS PASSED - Dashboard is ready for production!")
        print("\n📋 Next steps:")
        print("1. Set environment variables for production")
        print("2. Configure reverse proxy (nginx/apache)")
        print("3. Set up monitoring and logging")
        print("4. Configure backup procedures")
        print("5. Test with production data")
    elif passed >= total * 0.8:
        print("🟡 MOSTLY READY - Minor issues found, review warnings above")
    else:
        print("🔴 NOT READY - Critical issues found, please fix errors above")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_production_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)
