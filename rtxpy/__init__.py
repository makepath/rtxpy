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

__version__ = "0.0.5"

# Register xarray accessor if xarray is available
try:
    from . import accessor  # noqa: F401
except ImportError:
    pass  # xarray not installed
