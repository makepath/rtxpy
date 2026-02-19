"""Guanajuato — GPU-accelerated terrain exploration."""
from rtxpy import quickstart

quickstart(
    name='guanajuato',
    bounds=(-101.50, 20.70, -100.50, 21.30),
    crs='EPSG:32614',
    features=['buildings', 'roads', 'water', 'fire'],
)
