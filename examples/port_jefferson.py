"""Port Jefferson, NY — 1 m USGS 3DEP LiDAR DEM exploration.

Downloads USGS 1-meter LiDAR-derived DEM tiles for Port Jefferson
harbor and the surrounding village on Long Island's north shore,
then launches the interactive rtxpy viewer with buildings, roads,
and satellite imagery.

First run downloads the DEM tiles from USGS (~30 MB per 10 km tile)
and caches them as a zarr store. Subsequent runs load from cache.

Usage:
    python examples/port_jefferson.py
"""
from rtxpy import quickstart

# Port Jefferson harbor + village — tight bbox to keep 1m DEM manageable
# ~3.5 km E-W x ~3 km N-S
quickstart(
    name='port_jefferson',
    bounds=(-73.10, 40.92, -73.04, 40.97),
    crs='EPSG:32618',
    source='usgs_1m',
    features=['buildings', 'roads', 'water'],
    tiles='satellite',
    tile_zoom=18,
    hydro=True,
    wind=True,
    weather=True,
)
