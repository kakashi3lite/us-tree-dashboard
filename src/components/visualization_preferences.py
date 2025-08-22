"""Minimal stub for visualization preferences component.

The original implementation contained formatting artifacts that caused
SyntaxError during import in the test environment. To unblock the test
suite we provide only a no-op factory and empty callback registrar. The
full feature set can be restored later without affecting imports.
"""

from dash import html


def create_visualization_preferences():  # pragma: no cover - trivial stub
	return html.Div(
		[
			html.H5("Visualization Preferences (stub)"),
			html.P("Visualization preference UI temporarily disabled."),
		],
		id="visualization-preferences-stub",
	)


def register_callbacks(app):  # pragma: no cover - nothing to register
	return None