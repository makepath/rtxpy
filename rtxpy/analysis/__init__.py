"""Raster analysis functions using GPU-accelerated ray tracing.

This subpackage provides geospatial analysis functions that leverage
RTX ray tracing for fast computation on GPU.
"""

from .viewshed import viewshed
from .hillshade import hillshade, get_sun_dir

__all__ = ['viewshed', 'hillshade', 'get_sun_dir']
