# API Reference

## RTXAccessor (xarray accessor)

Registered as `dem.rtx.<method>()` on any `xr.DataArray` when `import rtxpy` is called.

### GPU Transfer

#### `to_cupy()`

Convert DataArray data to a cupy array on the GPU.

Returns a new DataArray with GPU-resident data. If already a cupy array, returns unchanged.

**Returns:** `xr.DataArray`

---

### Terrain Analysis

#### `hillshade(shadows=False, azimuth=225, angle_altitude=25, name='hillshade', rtx=None)`

Compute hillshade illumination with optional ray-traced shadows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shadows` | bool | `False` | Cast shadow rays (shadowed areas at half brightness) |
| `azimuth` | int | `225` | Sun direction in degrees clockwise from north |
| `angle_altitude` | int | `25` | Sun elevation above horizon in degrees |
| `name` | str | `'hillshade'` | Name attribute for output DataArray |
| `rtx` | RTX | `None` | Existing RTX instance to reuse |

**Returns:** `xr.DataArray` — values 0 (dark) to 1 (bright), edge pixels NaN

#### `slope(rtx=None)`

Compute terrain slope in degrees.

**Returns:** `xr.DataArray` — 0 (flat) to 90 (vertical), edge pixels NaN

#### `aspect(rtx=None)`

Compute terrain aspect (downhill compass bearing).

**Returns:** `xr.DataArray` — 0=N, 90=E, 180=S, 270=W; -1=flat; edge pixels NaN

#### `viewshed(x, y, observer_elev=0, target_elev=0, rtx=None)`

Compute line-of-sight visibility from an observer point.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | int/float | required | Observer X coordinate (raster units) |
| `y` | int/float | required | Observer Y coordinate (raster units) |
| `observer_elev` | float | `0` | Height above terrain at observer |
| `target_elev` | float | `0` | Height above terrain at targets |

**Returns:** `xr.DataArray` — viewing angle in degrees for visible cells, -1 for not visible

---

### Terrain Meshing

#### `triangulate(geometry_id='terrain', scale=1.0, pixel_spacing_x=None, pixel_spacing_y=None)`

Create a TIN (triangulated irregular network) mesh from the elevation data and add it to the RTX scene.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry_id` | str | `'terrain'` | Scene geometry ID |
| `scale` | float | `1.0` | Elevation scale factor |
| `pixel_spacing_x` | float | `None` | X pixel spacing in world units (auto from coords if None) |
| `pixel_spacing_y` | float | `None` | Y pixel spacing in world units |

**Returns:** `(vertices, indices)` — numpy float32 and int32 arrays

#### `voxelate(geometry_id='terrain', scale=1.0, base_elevation=None, pixel_spacing_x=None, pixel_spacing_y=None)`

Create a voxelized box-column mesh and add it to the scene. Each cell becomes a rectangular column from `base_elevation` to the cell's elevation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry_id` | str | `'terrain'` | Scene geometry ID |
| `scale` | float | `1.0` | Elevation scale factor |
| `base_elevation` | float | `None` | Column base Z (auto from min elevation if None) |
| `pixel_spacing_x` | float | `None` | X pixel spacing in world units |
| `pixel_spacing_y` | float | `None` | Y pixel spacing in world units |

**Returns:** `(vertices, indices)` — numpy float32 and int32 arrays

---

### Geometry Management

#### `add_geometry(geometry_id, vertices, indices, transform=None)`

Add a geometry to the multi-GAS scene.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry_id` | str | required | Unique geometry ID |
| `vertices` | array | required | Flat float32 array (3 floats per vertex) |
| `indices` | array | required | Flat int32 array (3 ints per triangle) |
| `transform` | list | `None` | 12-float 3x4 affine transform (identity if None) |

**Returns:** `int` — 0 on success

#### `remove_geometry(geometry_id)`

Remove a geometry from the scene.

**Returns:** `int` — 0 on success, -1 if not found

#### `list_geometries()`

**Returns:** `list[str]` — all geometry IDs in the scene

#### `get_geometry_count()`

**Returns:** `int` — number of geometries (0 in single-GAS mode)

#### `has_geometry(geometry_id)`

**Returns:** `bool`

#### `set_geometry_color(geometry_id, color)`

Set a solid color override for a geometry (bypasses elevation colormap).

| Parameter | Type | Description |
|-----------|------|-------------|
| `geometry_id` | str | Geometry to color |
| `color` | tuple | `(r, g, b)` with values [0-1] |

#### `clear()`

Remove all geometries and reset to single-GAS mode.

---

### Feature Placement

#### `place_mesh(mesh_source, positions, geometry_id=None, scale=1.0, rotation_z=0.0, swap_yz=False, center_xy=True, base_at_zero=True, pixel_spacing_x=1.0, pixel_spacing_y=1.0)`

Load a 3D model and place instances on the terrain.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh_source` | str/Path/callable | required | Mesh file path or callable returning `(vertices, indices)` |
| `positions` | list/callable | required | `[(x, y), ...]` pixel coords, or callable `(terrain_2d) -> [(x,y)]` |
| `geometry_id` | str | `None` | Base name (auto from filename if None) |
| `scale` | float | `1.0` | Mesh scale (only for file paths) |
| `rotation_z` | float/`'random'` | `0.0` | Z-axis rotation in radians |
| `swap_yz` | bool | `False` | Swap Y/Z for Y-up models |
| `center_xy` | bool | `True` | Center mesh at XY origin |
| `base_at_zero` | bool | `True` | Place mesh base at Z=0 |

**Returns:** `(vertices, indices, transforms)`

#### `place_geojson(geojson, height=10.0, label_field=None, height_field=None, fill_mesh=None, fill_spacing=None, fill_scale=1.0, models=None, model_field=None, geometry_id='geojson', densify=True, merge=False, extrude=False, mesh_cache=None, color=None, width=None)`

Place GeoJSON features as 3D meshes on terrain.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geojson` | str/Path/dict | required | GeoJSON file, dict, or FeatureCollection |
| `height` | float | `10.0` | Default extrusion height |
| `label_field` | str | `None` | Property field for grouping/naming |
| `height_field` | str | `None` | Property field for per-feature height |
| `fill_mesh` | str/callable | `None` | Mesh to scatter inside polygons |
| `fill_spacing` | float | `None` | Spacing between fill instances (required with fill_mesh) |
| `fill_scale` | float | `1.0` | Scale for fill mesh instances |
| `models` | dict | `None` | `{property_value: mesh_source}` per-feature models |
| `model_field` | str | `None` | Property name to look up in `models` |
| `geometry_id` | str | `'geojson'` | Base ID prefix |
| `densify` | bool/float | `True` | Vertex densification (True=1px, float=custom step) |
| `merge` | bool | `False` | Merge all features into single GAS (faster for many features) |
| `extrude` | bool | `False` | Extrude polygons into solid 3D geometry |
| `mesh_cache` | str/Path | `None` | `.npz` cache file (requires `merge=True`) |
| `color` | tuple | `None` | `(r, g, b)` override for all features |
| `width` | float | `None` | Ribbon half-width (auto from pixel spacing) |

**Returns:** `dict` — `{'features': int, 'geometries': int, 'geometry_ids': list}`

#### `place_buildings(geojson, elev_scale=None, default_height_m=8.0, mesh_cache=None)`

Place building footprints as extruded 3D geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geojson` | dict | required | Building footprint FeatureCollection |
| `elev_scale` | float | `None` | Height scaling factor (auto-computed if None) |
| `default_height_m` | float | `8.0` | Fallback height when feature has no `height` property |
| `mesh_cache` | str/Path | `None` | `.npz` cache file (auto-invalidates on parameter change) |

**Returns:** `dict` — `{'features': int, 'geometries': int, 'geometry_ids': list}`

#### `place_roads(geojson, geometry_id='road', color=None, height=3, mesh_cache=None)`

Place road LineStrings as flat ribbon geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geojson` | dict | required | Road FeatureCollection |
| `geometry_id` | str | `'road'` | Layer name |
| `color` | tuple | `(0.30, 0.30, 0.30)` | RGB color |
| `height` | float | `3` | Ribbon height above terrain |
| `mesh_cache` | str/Path | `None` | `.npz` cache file |

**Returns:** `dict`

#### `place_water(geojson, body_height=0.5, mesh_cache_prefix=None)`

Classify and place water features (major/minor waterways, water bodies).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geojson` | dict | required | Water FeatureCollection |
| `body_height` | float | `0.5` | Extrusion height for water body polygons |
| `mesh_cache_prefix` | str/Path | `None` | Base path for cache files (`_major_mesh.npz`, etc.) |

**Returns:** `dict` — `{'major': info, 'minor': info, 'body': info}`

---

### Mesh I/O

#### `save_meshes(zarr_path)`

Save all baked mesh geometries to a zarr store, spatially partitioned to match the DEM's chunk layout.

Requires features placed with `merge=True` first.

#### `load_meshes(zarr_path, chunks=None)`

Load mesh geometries from a zarr store and add them to the scene.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zarr_path` | str/Path | required | Path to zarr store with `meshes/` group |
| `chunks` | list | `None` | Specific `[(row, col), ...]` to load; `None` loads all |

---

### Rendering

#### `render(camera_position, look_at, ...)`

Render terrain with a perspective camera. See [Rendering section in User Guide](user-guide.md#rendering) for full parameter list.

Key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_position` | tuple | required | `(x, y, z)` in world coords |
| `look_at` | tuple | required | `(x, y, z)` target point |
| `fov` | float | `60.0` | Vertical FOV in degrees |
| `width` / `height` | int | `1920` / `1080` | Output resolution |
| `sun_azimuth` | float | `225` | Sun direction (degrees from north) |
| `sun_altitude` | float | `45` | Sun elevation (degrees) |
| `shadows` | bool | `True` | Cast shadow rays |
| `ambient` | float | `0.15` | Ambient light [0-1] |
| `fog_density` | float | `0.0` | Exponential fog (0 = off) |
| `colormap` | str | `'terrain'` | Matplotlib colormap name |
| `color_range` | tuple | `None` | `(min, max)` for colormap (auto if None) |
| `vertical_exaggeration` | float | `None` | Z scale (auto if None) |
| `ao_samples` | int | `0` | Ambient occlusion rays per pixel |
| `ao_radius` | float | `None` | AO max distance (auto if None) |
| `sun_angle` | float | `0.0` | Soft shadow cone half-angle (degrees) |
| `aperture` | float | `0.0` | Depth-of-field lens radius |
| `focal_distance` | float | `0.0` | DOF focal plane distance (auto if 0) |
| `alpha` | bool | `False` | RGBA output with transparent sky |
| `output_path` | str | `None` | Save to file (PNG, TIFF, etc.) |

**Returns:** `np.ndarray` — `(height, width, 3)` or `(height, width, 4)` float32 [0-1]

#### `flyover(output_path, duration=30.0, fps=10.0, orbit_scale=0.6, altitude_offset=500.0, fov=60.0, fov_range=None, width=1280, height=720, ...)`

Create an orbital flyover GIF animation.

**Returns:** `str` — path to saved GIF

#### `view(x, y, z, output_path, duration=10.0, fps=12.0, look_distance=1000.0, look_down_angle=10.0, fov=70.0, ...)`

Create a 360-degree panoramic GIF from a fixed viewpoint.

**Returns:** `str` — path to saved GIF

---

### Tile Overlays

#### `place_tiles(url='osm', zoom=None)`

Download and drape XYZ map tiles on terrain. Call before `explore()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | `'osm'` | Provider name (`'osm'`, `'satellite'`, `'topo'`) or URL template |
| `zoom` | int | `None` | Tile zoom level (default 13) |

**Returns:** `XYZTileService`

---

### Interactive Viewer

#### `explore(width=800, height=600, render_scale=0.5, start_position=None, look_at=None, mesh_type='tin', subsample=1, wind_data=None, scene_zarr=None, ao_samples=0, denoise=False, ...)`

Launch the interactive terrain viewer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` / `height` | int | `800` / `600` | Window size |
| `render_scale` | float | `0.5` | Render fraction (0.25-1.0) |
| `start_position` | tuple | `None` | Initial camera `(x, y, z)` |
| `look_at` | tuple | `None` | Initial look-at `(x, y, z)` |
| `mesh_type` | str | `'tin'` | `'tin'` or `'voxel'` |
| `subsample` | int | `1` | Initial terrain subsample factor |
| `wind_data` | dict | `None` | Wind data from `fetch_wind()` |
| `scene_zarr` | str/Path | `None` | Zarr store for chunk-based mesh streaming |
| `ao_samples` | int | `0` | Enable AO on launch (0 = off) |
| `denoise` | bool | `False` | Enable OptiX AI Denoiser |
| `title` | str | `None` | Window title |
| `color_stretch` | str | `'linear'` | Initial stretch (`'linear'`, `'sqrt'`, `'cbrt'`, `'log'`) |

---

### Utilities

#### `trace(rays, hits, num_rays, primitive_ids=None, instance_ids=None)`

Trace rays against the current scene.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rays` | array | 8 float32 per ray: `[ox, oy, oz, tmin, dx, dy, dz, tmax]` |
| `hits` | array | 4 float32 per hit: `[t, nx, ny, nz]` (t=-1 = miss) |
| `num_rays` | int | Number of rays |
| `primitive_ids` | array | Optional output: triangle index per ray |
| `instance_ids` | array | Optional output: geometry instance ID per ray |

**Returns:** `int` — 0 on success

#### `memory_usage()`

Print and return memory breakdown of the current scene.

**Returns:** `dict` — byte counts per component

---

## LOD Utilities

Level-of-detail helpers for terrain tiles and instanced geometry.

```python
from rtxpy import compute_lod_level, compute_lod_distances, simplify_mesh, build_lod_chain
```

#### `compute_lod_level(distance, lod_distances)`

Map a distance value to a discrete LOD level.

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance` | float | Distance from camera to object/tile center |
| `lod_distances` | list[float] | Ascending thresholds for LOD transitions |

**Returns:** `int` — LOD level (0 = highest detail)

#### `compute_lod_distances(tile_diagonal, factor=3.0, max_lod=3)`

Generate distance thresholds from tile geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_diagonal` | float | | Tile diagonal in world units |
| `factor` | float | `3.0` | Base multiplier for first threshold |
| `max_lod` | int | `3` | Number of LOD transitions |

**Returns:** `list[float]` — distance thresholds

#### `simplify_mesh(vertices, indices, ratio)`

Simplify a triangle mesh via quadric decimation (requires `trimesh`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `vertices` | ndarray | Flat float32 vertex buffer `(N*3,)` |
| `indices` | ndarray | Flat int32 index buffer `(M*3,)` |
| `ratio` | float | Fraction of triangles to keep (0.0-1.0) |

**Returns:** `(vertices, indices)` — simplified mesh buffers

#### `build_lod_chain(vertices, indices, ratios=(1.0, 0.5, 0.25, 0.1))`

Build a list of progressively simplified meshes.

**Returns:** `list[(vertices, indices)]` — one pair per LOD level

---

## RTX Class

Low-level OptiX wrapper. Use this directly for custom ray tracing without the xarray accessor.

```python
from rtxpy import RTX

rtx = RTX()
```

### `build(gas_id, vertices, indices)`

Build a single-GAS acceleration structure.

### `add_geometry(geometry_id, vertices, indices, transform=None)`

Add geometry for multi-GAS mode.

### `remove_geometry(geometry_id)`

### `clear_scene()`

### `trace(rays, hits, num_rays, primitive_ids=None, instance_ids=None)`

### `list_geometries()`

### `get_geometry_count()`

### `has_geometry(geometry_id)`

### `get_geometry_transform(geometry_id)`

### `update_transform(geometry_id, transform)`

### `set_geometry_visible(geometry_id, visible)`

---

## Standalone Functions

### Data Fetching

#### `fetch_dem(bounds, output_path, source='copernicus', crs=None, cache_dir=None)`

Download, merge, and clip DEM tiles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bounds` | tuple | required | `(west, south, east, north)` WGS84 |
| `output_path` | str/Path | required | `.zarr` or `.tif` output path |
| `source` | str | `'copernicus'` | `'copernicus'`, `'srtm'`, `'usgs_10m'`, `'usgs_1m'` |
| `crs` | str | `None` | Target CRS for reprojection |
| `cache_dir` | str/Path | `None` | Directory for raw tile cache |

**Returns:** `xr.DataArray`

#### `fetch_buildings(bounds, cache_path=None, crs=None, cache_dir=None, source='microsoft')`

Download building footprints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str | `'microsoft'` | `'microsoft'` or `'overture'` |

**Returns:** `dict` — GeoJSON FeatureCollection

#### `fetch_roads(bounds, road_type='all', cache_path=None, crs=None, source='osm')`

Download road data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `road_type` | str | `'all'` | `'major'`, `'minor'`, or `'all'` |
| `source` | str | `'osm'` | `'osm'` or `'overture'` |

**Returns:** `dict` — GeoJSON FeatureCollection

#### `fetch_water(bounds, water_type='all', cache_path=None, crs=None)`

Download water features from OSM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `water_type` | str | `'all'` | `'waterway'`, `'waterbody'`, or `'all'` |

**Returns:** `dict` — GeoJSON FeatureCollection

#### `fetch_osm(bounds, tags=None, crs=None, cache_path=None)`

Download arbitrary OpenStreetMap features.

**Returns:** `dict` — GeoJSON FeatureCollection

#### `fetch_wind(bounds, grid_size=20)`

Fetch current wind data from Open-Meteo.

**Returns:** `dict` — `{'u': ndarray, 'v': ndarray, 'speed': ndarray, 'direction': ndarray, 'lats': ndarray, 'lons': ndarray}`

#### `fetch_firms(bounds, date_span='24h', region=None, cache_path=None, crs=None)`

Download NASA FIRMS LANDSAT 30m fire detection footprints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date_span` | str | `'24h'` | `'24h'`, `'48h'`, `'72h'`, or `'7d'` |

**Returns:** `dict` — GeoJSON FeatureCollection

---

### Mesh Utilities

#### `load_mesh(filepath, scale=1.0, swap_yz=False, center_xy=False, base_at_zero=False)`

Load a mesh file (GLB, OBJ, STL, PLY, etc.) via trimesh.

**Returns:** `(vertices, indices)` — flat float32 and int32 arrays

#### `load_glb(filepath, scale=1.0, swap_yz=False, center_xy=False, base_at_zero=False)`

Load a GLB/glTF mesh file. Same as `load_mesh()`.

#### `make_transform(x=0.0, y=0.0, z=0.0, scale=1.0, rotation_z=0.0)`

Create a 3x4 affine transform matrix.

**Returns:** `list[float]` — 12-element row-major `[r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]`

#### `make_transforms_on_terrain(positions, terrain, scale=1.0, rotation_z=0.0)`

Create transform matrices for placing objects on a terrain surface.

**Returns:** `list[list[float]]` — one 12-element transform per position

#### `triangulate_terrain(verts, triangles, terrain, scale=1.0)`

Convert a 2D elevation array into a triangle mesh. Auto-selects GPU (cupy) or CPU (numba) execution.

#### `voxelate_terrain(verts, triangles, terrain, scale=1.0, base_elevation=0.0)`

Convert a 2D elevation array into a voxelized box-column mesh.

#### `write_stl(filename, verts, triangles)`

Save a triangle mesh to binary STL format.

---

### Zarr Mesh Store

#### `save_meshes_to_zarr(zarr_path, meshes, colors, pixel_spacing, elevation_shape, elevation_chunks)`

Save mesh data spatially partitioned into a zarr store.

#### `load_meshes_from_zarr(zarr_path, chunks=None)`

Load mesh data from a zarr store.

**Returns:** `(meshes, colors, meta)`

#### `chunks_for_pixel_window(zarr_path, x0, y0, x1, y1)`

Find which zarr chunks overlap a pixel-coordinate window.

---

## Render Graph

A configurable DAG of render passes. Declare inputs/outputs per pass, and the graph resolves execution order, manages GPU buffers, and gates passes on hardware capabilities.

```python
from rtxpy import RenderGraph, RenderPass, BufferDesc
```

### `BufferDesc(dtype='float32', channels=3, per_pixel=True)`

Describes a GPU buffer's shape and data type.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | str | `'float32'` | NumPy-compatible dtype |
| `channels` | int | `3` | Channels per element |
| `per_pixel` | bool | `True` | `True` = `(H, W, C)` shape; `False` = `(N, C)` flat |

#### `shape(width, height)`

**Returns:** `tuple[int, ...]` — concrete shape for the given resolution

### `RenderPass` (abstract base class)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Unique pass name |
| `inputs` | dict[str, BufferDesc] | `{}` | Buffers this pass reads |
| `outputs` | dict[str, BufferDesc] | `{}` | Buffers this pass writes |
| `enabled` | bool | `True` | Set `False` to skip |
| `requires` | list[str] | `[]` | Capability keys that must be truthy |

Override `execute(buffers)` with your pass logic. Optionally override `setup(graph)` and `teardown()`.

### `RenderGraph(width=1920, height=1080)`

#### `add_pass(pass_)`

Add a render pass. Raises `ValueError` on duplicate names.

#### `remove_pass(name)`

Remove a pass by name. Raises `KeyError` if not found.

#### `get_pass(name)`

**Returns:** `RenderPass`

#### `set_fallback(buffer, fallback)`

If `buffer`'s producer is disabled, downstream passes read `fallback` instead.

```python
graph.set_fallback("denoised_color", "color")
```

#### `compile(capabilities=None, validate=True)`

Compile the graph: capability-gate passes, topological sort, buffer lifetime analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capabilities` | dict | `None` | From `get_capabilities()` |
| `validate` | bool | `True` | Raise `GraphValidationError` on problems |

**Returns:** `CompiledGraph`

### `CompiledGraph`

#### `execute(external_buffers=None, allocator=None)`

Run all passes in dependency order.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `external_buffers` | dict | `None` | Pre-allocated buffers to inject |
| `allocator` | callable | `None` | `(shape, dtype) -> array`; defaults to `cupy.zeros` |

**Returns:** `dict[str, array]` — all buffers after execution

### `GraphValidationError`

Raised on cycle detection, missing inputs, or other graph errors.

---

### Device Utilities

#### `get_device_count()`

**Returns:** `int` — number of CUDA-capable GPUs

#### `get_device_properties(device_id=0)`

**Returns:** `dict` — GPU properties

#### `list_devices()`

Print all available CUDA devices.

#### `get_current_device()`

**Returns:** `int` — current CUDA device ID
