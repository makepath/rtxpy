"""Los Angeles — GPU-accelerated terrain exploration."""
from rtxpy import quickstart

quickstart(
    name='los_angeles',
    bounds=(-118.52, 33.85, -117.25, 34.23),
    crs='EPSG:32611',
    source='usgs_10m',
    features=['buildings', 'roads', 'water', 'fire'],
)
