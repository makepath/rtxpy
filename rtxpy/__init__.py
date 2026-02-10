from .rtx import (
    RTX,
    has_cupy,
    get_device_count,
    get_device_properties,
    list_devices,
    get_current_device,
)
from .mesh import (
    triangulate_terrain,
    voxelate_terrain,
    write_stl,
    load_glb,
    load_mesh,
    make_transform,
    make_transforms_on_terrain,
)
from .analysis import viewshed, hillshade, render, flyover, view
from .engine import explore

__version__ = "0.0.6"

# Optional convenience — network helpers with lazy dependency checks
try:
    from .remote_data import fetch_dem, fetch_osm, fetch_buildings, fetch_roads, fetch_water, fetch_wind, fetch_firms
except ImportError:
    pass

# Register xarray accessor if xarray is available
try:
    from . import accessor  # noqa: F401
except ImportError:
    pass  # xarray not installed
