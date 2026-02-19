"""Rio de Janeiro — GPU-accelerated terrain exploration."""
from rtxpy import quickstart

quickstart(
    name='rio',
    bounds=(-43.42, -23.08, -43.10, -22.84),
    crs='EPSG:32723',
    features=['buildings', 'roads', 'water'],
)
