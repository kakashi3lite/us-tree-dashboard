"""Minimal theme switcher stub for tests.

Original implementation removed due to corruption; this keeps the public
API (create_theme_switcher, register_callbacks) so imports succeed.
"""
from __future__ import annotations
from dash import html


def create_theme_switcher():  # pragma: no cover - trivial
    return html.Div(id="theme-switcher", children=[html.Span("Theme: Light")])


def register_callbacks(app):  # pragma: no cover - no callbacks
    return None