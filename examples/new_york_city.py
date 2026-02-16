"""New York City — GPU-accelerated terrain exploration."""
import sys
from rtxpy import quickstart

tour = None
if '--tour' in sys.argv:
    idx = sys.argv.index('--tour')
    tour = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'nyc_tour.py'

quickstart(
    name='nyc',
    bounds=(-74.26, 40.49, -73.70, 40.92),
    crs='EPSG:32618',
    features=['buildings', 'roads', 'water', 'fire', 'restaurant_grades', 'gtfs'],
    ao_samples=1,
    tour=tour,
)
