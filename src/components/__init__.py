"""Dashboard components package.

For the current test suite only the VOSviewer network component is
required. Other previously large component modules (theme switcher,
visualization preferences, notification system, project context manager)
are intentionally NOT imported here to avoid failing imports due to
legacy file corruption. They can be reintroduced once repaired or
re-implemented. This keeps `app.py` imports lightweight and stable.
"""

from .vosviewer_network import (
	create_vosviewer_network,
	register_callbacks as register_vosviewer_callbacks,
)

__all__ = [
	"create_vosviewer_network",
	"register_vosviewer_callbacks",
]
