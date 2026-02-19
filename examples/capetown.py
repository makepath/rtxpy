"""Cape Town — GPU-accelerated terrain exploration."""
from rtxpy import quickstart

quickstart(
    name='capetown',
    bounds=(18.3, -34.2, 18.7, -33.8),
    crs='EPSG:32734',
    features=['buildings', 'roads', 'water', 'fire'],
)
