"""
Add the parent ``./`` directory to ``sys.path`` so the raw-traversal
drivers can import the operator primitives from the existing benchmark
scripts (``geosot_overlay_query_v7``, ``geosot_reverse_query``, etc.)
without packaging them as a module.

Usage at the top of every raw_traversal driver:

    from _path_setup import ensure_geosot_on_path
    ensure_geosot_on_path()

After the call, ``import geosot_spatial_reasoning`` etc. resolve to the
canonical implementations in ``./``.
"""

from __future__ import annotations

import os
import sys


def ensure_geosot_on_path() -> str:
    """
    Make the parent ./ directory importable.

    Returns the absolute parent directory that was prepended to
    ``sys.path``. Idempotent; safe to call multiple times.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    parent = os.path.abspath(os.path.join(here, os.pardir))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return parent
