from .rtx import (
    RTX,
    has_cupy,
    get_device_count,
    get_device_properties,
    list_devices,
    get_current_device,
    get_capabilities,
)
from .mesh import (
    triangulate_terrain,
    voxelate_terrain,
    add_terrain_skirt,
    build_terrain_skirt,
    write_stl,
    load_glb,
    load_mesh,
    make_transform,
    make_transforms_on_terrain,
)
from .mesh_store import (
    save_meshes_to_zarr,
    load_meshes_from_zarr,
    chunks_for_pixel_window,
)
from .analysis import viewshed, hillshade, render, flyover, view
from .engine import explore

__version__ = "0.1.0"

# Optional convenience — network helpers with lazy dependency checks
try:
    from .remote_data import fetch_dem, fetch_lidar, fetch_osm, fetch_buildings, fetch_roads, fetch_water, fetch_wind, fetch_weather, fetch_firms, fetch_places, fetch_infrastructure, fetch_land_use, fetch_restaurant_grades, fetch_gtfs
except ImportError:
    pass

# Jupyter notebook viewer
try:
    from .notebook import JupyterViewer
except ImportError:
    pass

# One-call launcher
try:
    from .quickstart import quickstart
except ImportError:
    pass

# Register xarray accessor if xarray is available
try:
    from . import accessor  # noqa: F401
except ImportError:
    pass  # xarray not installed
