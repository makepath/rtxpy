"""Trinidad and Tobago — GPU-accelerated terrain exploration with hydro flow."""
from rtxpy import quickstart

quickstart(
    name='trinidad',
    bounds=(-61.95, 10.04, -60.44, 11.40),
    crs='EPSG:32620',
    features=['buildings', 'roads', 'water', 'fire', 'places',
              'infrastructure', 'land_use'],
    hydro=True,
    ao_samples=1,
    denoise=True,
)
