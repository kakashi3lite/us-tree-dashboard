"""Tests for `conservation_alerts.ConservationAlertSystem` canopy trend analysis.
Focus: severity assessment edge cases and alert generation happy path.
"""
from datetime import datetime, timedelta
import pandas as pd
from src.conservation_alerts import ConservationAlertSystem


def _build_canopy_df(region: str, start_canopy: float, end_canopy: float):
    """Utility to construct a minimal canopy trend DataFrame spanning two months."""
    now = datetime.now()
    data = [
        {"timestamp": (now - timedelta(days=60)).isoformat(), "region": region, "canopy_pct": start_canopy, "latitude": 40.0, "longitude": -120.0, "area_km2": 10.0, "species_count": 25},
        {"timestamp": now.isoformat(), "region": region, "canopy_pct": end_canopy, "latitude": 40.0, "longitude": -120.0, "area_km2": 10.0, "species_count": 25},
    ]
    return pd.DataFrame(data)


def test_analyze_canopy_trends_generates_alert():
    system = ConservationAlertSystem()
    df = _build_canopy_df("Test Region", 100.0, 80.0)  # 20% decline -> high severity (threshold 15) but <25 critical
    alerts = system.analyze_canopy_trends(df, time_window_months=3)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity in {"high", "critical"}
    assert "declined" in alert.threat_description.lower()


def test_analyze_canopy_trends_no_timestamp():
    system = ConservationAlertSystem()
    df = pd.DataFrame({"region": ["X"], "canopy_pct": [90.0]})
    alerts = system.analyze_canopy_trends(df)
    assert alerts == []  # gracefully handles missing timestamp column


def test_assess_severity_thresholds():
    system = ConservationAlertSystem()
    # direct access to private method acceptable for unit boundary validation
    assert system._assess_severity("canopy_decline", 30) == "critical"
    assert system._assess_severity("canopy_decline", 16) == "high"
    assert system._assess_severity("canopy_decline", 11) == "medium"
    assert system._assess_severity("canopy_decline", 6) == "low"
    assert system._assess_severity("canopy_decline", 1) == "none"


def test_generate_canopy_recommendations():
    system = ConservationAlertSystem()
    recs_low = system._generate_canopy_recommendations(6, "R")
    recs_high = system._generate_canopy_recommendations(20, "R")
    assert any("Monitor" in r or "monitor" in r for r in recs_low)
    assert any("field" in r.lower() for r in recs_high)
