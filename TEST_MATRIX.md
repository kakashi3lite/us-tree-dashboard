# Test Matrix

| Area | Framework | Existing Tests | Gaps |
|------|-----------|----------------|------|
| Core utilities (`src/ml_utils.py`, `src/cleanup.py`) | `pytest` | `tests/test_ml_utils.py`, `tests/test_cleanup.py` | No tests for error handling paths |
| EnhanceX integrations | `pytest` | `tests/test_enhancex.py` | Limited coverage of failure modes |
| Dash UI callbacks | `pytest` + `dash[testing]` | none | Missing smoke tests for `app.py` layout/callbacks |
| API endpoints | `pytest` + `Flask` client | none | Need integration tests for `/api/v1/*` |
| Performance | `pytest`, `locust` | `tests/performance/performance_tests.py` | No automated thresholds or CI hooks |

## Missing High-Value Tests
1. Smoke test for `app.py` to ensure layout loads.
2. Integration test for `/api/v1/health` and dataset routes.
3. Error-path tests for `TreeDashboard.load_data` when files missing.
4. Unit tests for `conservation_dashboard_enhancer.py` callbacks.
5. Performance regression test with fixed data set.

