# rtxpy scene zarr specification

Version 1.0 draft — March 2026

## What this is

A single zarr store that contains everything `explore()` needs to render a
scene: terrain elevation, placed geometry, analysis layers, rendering
parameters, and camera state. One file, no network access required.

The format already exists in pieces across `remote_data.py`, `mesh_store.py`,
`explore_zarr.py`, and `quickstart.py`. This document pulls those pieces
together into a reference spec and adds groups for data that explore()
currently accepts as in-memory arguments but doesn't persist.

## File structure

```
scene.zarr/
  .zattrs                          # root metadata (version, CRS, bounds)
  elevation/                       # primary DEM raster
  elevation_lod2/                  # optional LOD pyramid level (2x downsample)
  elevation_lod4/                  # optional LOD pyramid level (4x downsample)
  elevation_lod8/                  # ...up to elevation_lod64
  elevation_roughness/             # optional per-tile roughness for adaptive LOD
  spatial_ref                      # scalar variable with CRS attributes
  meshes/                          # placed geometry (buildings, roads, water, etc.)
    {geometry_id}/
      {chunk_row}_{chunk_col}/
        vertices, indices          # LOD 0 (full detail)
        vertices_lod1, indices_lod1  # optional pre-simplified LOD levels
        vertices_lod2, indices_lod2
        vertices_lod3, indices_lod3
  overlays/                        # raster overlay layers
    {layer_name}/
  wind/                            # wind velocity grids
  hydro/                           # hydrological flow data
  weather/                         # cloud cover and temperature
  camera/                          # initial camera state (attrs only)
  render/                          # rendering defaults (attrs only)
  tour/                            # camera tour keyframes
  observers/                       # pre-set observer positions
```

Only `elevation` and `spatial_ref` are required. Everything else is optional.
explore() already handles missing data gracefully (no meshes = no placed
geometry, no wind = no wind particles, etc.).

## Root attributes

Stored in `/.zattrs`:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `rtxpy_scene_version` | string | yes | Spec version, e.g. `"1.0"` |
| `created_at` | string | no | ISO 8601 timestamp |
| `name` | string | no | Human-readable scene name |
| `bounds_lonlat` | [W, S, E, N] | no | Geographic bounding box (EPSG:4326) |
| `source` | string | no | How the data was produced (e.g. `"quickstart"`, `"fetch_dem"`) |

## elevation

The primary DEM raster. Uses CF encoding to store float elevation values as
compressed int16.

**Array**: `elevation` — 2D, shape (H, W)

| Property | Value |
|----------|-------|
| dtype (on disk) | int16 |
| scale_factor | float64, typically 0.1 |
| add_offset | float64, typically 0.0 |
| _FillValue | int16, typically -9999 |
| chunks | (min(2048, H), min(2048, W)) |
| compressor | BloscCodec, cname=zstd, clevel=6, shuffle=bitshuffle |

Decoded elevation in meters: `value = raw_int16 * scale_factor + add_offset`.
Fill values decode to NaN and represent ocean or no-data pixels.

**Coordinate arrays**: `x` (W,) and `y` (H,) store the CRS coordinates of
each pixel center. Units depend on the CRS (meters for UTM, degrees for
geographic).

The chunk size determines the spatial partitioning grid that meshes, LOD
tiles, and the chunk manager all align to. Picking chunk sizes that are
powers of 2 (512, 1024, 2048) works well with the LOD pyramid.

## LOD pyramid levels

Optional pre-computed downsampled copies of the elevation array, stored as
sibling arrays at the zarr root.

**Naming**: `elevation_lod{factor}` where factor is a power of 2.

Common set: `elevation_lod2`, `elevation_lod4`, `elevation_lod8`,
`elevation_lod16`, `elevation_lod32`, `elevation_lod64`.

Each level is produced by 2x box-filter downsampling from the previous level
(NaN-aware averaging). A level at factor N has shape
`(ceil(H/N), ceil(W/N))`.

Same CF encoding as the primary array (int16, same scale/offset/fill). Same
compression. Chunks sized to fit the downsampled dimensions.

These are built lazily on first access by `explore_zarr.py`'s
`_build_lod_arrays()` and cached in the store. A producer can pre-build them
with `--build-lods` or skip them and let the viewer build on demand.

### Terrain tile roughness

**Array**: `elevation_roughness` — 2D float32, shape (n_tile_rows, n_tile_cols)

Per-tile roughness scores used by `TerrainLODManager` to adapt LOD thresholds
to terrain complexity. Each value is the standard deviation of elevation
residuals from a bilinear fit through the tile's four corners
(`compute_tile_roughness()` in `lod.py`). Flat tiles score near zero; jagged
ridgelines score high.

At runtime the raw roughness values get log-normalized across all tiles and
mapped to a scale factor in [0.5, 2.0] via `0.5 * 4^t`. Smooth tiles have
their effective camera distance doubled (LOD demoted sooner), rough tiles
have it halved (finer detail kept at greater distance). When all tiles have
equal roughness, every tile gets scale 1.0 (neutral).

**Attributes** on `elevation_roughness`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `tile_size` | int | Tile edge length in pixels used for roughness computation |
| `roughness_floor` | float | Minimum roughness threshold — tiles below this get neutral scale |

Pre-computing roughness avoids a full scan of the elevation grid on viewer
startup. The viewer recomputes if the tile_size doesn't match its own
(e.g. because zarr chunk alignment changed the tile grid).

### Terrain LOD tiling parameters

The `TerrainLODManager` tiles terrain into a grid and assigns per-tile LOD
levels based on camera distance. These parameters control that tiling and
can be stored as attributes on the elevation array or a dedicated group.

**Attributes** on `elevation` (optional):

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `lod_tile_size` | int | chunk_w | Tile edge length in pixels; should match elevation chunk_w for aligned loading |
| `lod_max_level` | int | 3 | Maximum LOD level (0 = full res, 3 = coarsest) |
| `lod_distance_factor` | float | 3.0 | Distance multiplier per LOD threshold step |
| `lod_base_subsample` | int | 1 | Global resolution factor applied before LOD |

When the viewer enables LOD, it reads these to configure the
`TerrainLODManager` instead of using hardcoded defaults. If absent, the
viewer falls back to its current auto-detection (tile_size from chunk grid,
max_level from pyramid depth).

## Mesh LOD

Pre-simplified variants of placed geometry, stored alongside the full-detail
meshes. Without these, the viewer runs `fast_simplification` at runtime
(which requires the optional `fast_simplification` package and burns CPU on
first access for each chunk/LOD combination).

### Layout

```
scene.zarr/
  meshes/
    building/
      0_0/
        vertices              # LOD 0 — full detail (existing)
        indices               # LOD 0 — full detail (existing)
        vertices_lod1         # LOD 1 — 50% triangles
        indices_lod1
        vertices_lod2         # LOD 2 — 25% triangles
        indices_lod2
        vertices_lod3         # LOD 3 — 10% triangles
        indices_lod3
```

**Naming**: `vertices_lod{N}` and `indices_lod{N}` where N matches the LOD
level (1, 2, 3). LOD 0 uses the existing `vertices`/`indices` arrays — no
suffix needed.

For curve geometries, LOD simplification doesn't apply (curves are cheap to
render and can't be meaningfully decimated). Sphere geometries also skip LOD.

**Attributes** on `/meshes/.zattrs` (in addition to existing ones):

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `lod_ratios` | [float, ...] | [1.0, 0.5, 0.25, 0.1] | Triangle ratio per LOD level |

The `lod_ratios` array tells the viewer what decimation was applied at each
level. The `_MeshChunkManager` currently hardcodes `(1.0, 0.5, 0.25, 0.1)`.
Storing the ratios in the file lets a producer choose different decimation
targets and lets the viewer know what it's loading.

### Producing mesh LOD

`save_meshes_to_zarr()` would accept an optional `lod_ratios` parameter.
For each chunk, after saving the full-detail mesh, it runs `simplify_mesh()`
at each ratio and saves the result as `vertices_lod{N}` / `indices_lod{N}`.

### Consuming mesh LOD

`_MeshChunkManager._get_simplified()` currently calls `simplify_mesh()` and
caches the result. With pre-computed LOD arrays, it would first check for
`vertices_lod{N}` in the chunk group and use that directly, skipping
decimation entirely. Falls back to runtime simplification if the arrays
aren't present.

This makes `fast_simplification` optional at runtime even when mesh LOD is
active — the work was done at scene build time.

## spatial_ref

A scalar int32 variable carrying CRS metadata as attributes. Follows the CF
grid_mapping convention used by rioxarray.

| Attribute | Type | Description |
|-----------|------|-------------|
| `crs_wkt` | string | WKT2 representation of the coordinate reference system |
| `GeoTransform` | string | Six space-separated values: `"x_origin dx rot_x y_origin rot_y dy"` |

The GeoTransform matches GDAL conventions. For a north-up raster with no
rotation: `x_origin` is the west edge, `dx` is positive pixel width,
`y_origin` is the north edge, `dy` is negative pixel height.

explore() uses this to set up CRS transforms for streaming tile I/O and
coordinate display.

## meshes

Placed geometry (buildings, roads, water features, point clouds) stored in a
nested group structure. Meshes are spatially partitioned into chunks that
align with the elevation grid's chunk layout.

**Group**: `/meshes/`

**Attributes** on `/meshes/.zattrs`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `pixel_spacing` | [psx, psy] | World units per pixel, matches elevation grid |
| `elevation_shape` | [H, W] | Shape of the elevation array |
| `elevation_chunks` | [chunk_h, chunk_w] | Chunk dimensions of the elevation array |
| `feature_keys` | [str, ...] | Optional list of geometry IDs that have been cached |

### Geometry groups

Each placed geometry type gets its own sub-group: `/meshes/{geometry_id}/`.
The geometry_id is a short string like `"building"`, `"road"`, `"water_major"`,
`"lidar"`. This ID maps to the GAS geometry ID in the OptiX scene.

**Attributes** on `/meshes/{geometry_id}/.zattrs`:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `color` | [r, g, b] or [r, g, b, a] | yes | Default color, floats 0-1 |
| `type` | string | no | `""` (triangles, default), `"curve"`, or `"sphere"` |

### Chunk sub-groups

Each geometry group contains one sub-group per spatial chunk:
`/meshes/{geometry_id}/{chunk_row}_{chunk_col}/`.

Chunk indices are computed from the triangle/segment centroids:
```
pixel_col = world_x / pixel_spacing_x
pixel_row = world_y / pixel_spacing_y
chunk_col = pixel_col // chunk_w
chunk_row = pixel_row // chunk_h
```

Vertex indices are remapped per-chunk so each chunk is self-contained (no
cross-chunk index references).

#### Triangle meshes (type = "" or absent)

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `vertices` | float32 | (N*3,) | Flat xyz: [x0, y0, z0, x1, y1, z1, ...] |
| `indices` | int32 | (T*3,) | Triangle indices into vertices/3 |

#### Curve geometries (type = "curve")

B-spline curve tubes used for roads and water features. OptiX renders these
as ROUND_QUADRATIC_BSPLINE primitives.

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `vertices` | float32 | (N*3,) | Control point xyz, flat |
| `widths` | float32 | (N,) | Per-control-point tube radius |
| `indices` | int32 | (S,) | Segment start indices (each segment uses 3 consecutive control points) |

#### Sphere geometries (type = "sphere")

Used for LiDAR point clouds and scattered point data.

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `centers` | float32 | (N*3,) | Sphere center xyz, flat |
| `radii` | float32 | (N,) | Per-sphere radius |
| `colors` | float32 | (N*4,) | Per-sphere RGBA, optional |
| `classification` | int32 | (N,) | ASPRS classification code per point, optional |
| `intensity` | float32 | (N,) | Normalized intensity [0-1] per point, optional |
| `rgb` | float32 | (N*3,) | RGB color per point [0-1], flat, optional |
| `return_number` | int32 | (N,) | Return number per point, optional |
| `number_of_returns` | int32 | (N,) | Total returns per pulse per point, optional |

The `colors` array stores the rendered RGBA used by the viewer (typically
derived from one of the attribute arrays via `build_colors()`). The raw
attribute arrays (`classification`, `intensity`, `rgb`, `return_number`)
are stored separately so the viewer can cycle color modes at runtime
(elevation, intensity, classification, RGB — toggled with the
point cloud color mode key).

**Attributes** on sphere geometry groups:

| Attribute | Type | Description |
|-----------|------|-------------|
| `color_mode` | string | Default color mode: `"elevation"`, `"intensity"`, `"classification"`, or `"rgb"` |
| `source_crs` | string | Optional WKT of the original LAS/LAZ CRS before reprojection |
| `point_count` | int | Total number of points (redundant with array length, but quick metadata lookup) |

When `classification` is present, the viewer can apply ASPRS standard colors
(ground=brown, vegetation=green, building=red, water=blue, etc.) from the
`CLASSIFICATION_COLORS` table in `pointcloud.py` without re-reading the
source file.

### Chunking alignment

The mesh chunk grid must match the elevation chunk grid exactly. This is what
lets the `_MeshChunkManager` load geometry for just the visible terrain
region, and what lets the LOD tile grid align with mesh chunks when
`tile_size == chunk_size`.

If a scene has 2048x2048 elevation chunks, the meshes are partitioned into
the same 2048x2048 pixel blocks. A building at world position (5000, 3000)
with pixel_spacing=30 is at pixel (166, 100), which falls in chunk (0, 0)
for 2048-wide chunks.

## overlays

Raster layers that can be cycled with G in the viewer. Each overlay is a 2D
array at the same resolution as the elevation grid.

**Group**: `/overlays/`

Each sub-group `/overlays/{layer_name}/` contains:

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `data` | float32 | (H, W) | Layer values |

**Attributes** on each layer sub-group:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `colormap` | string | no | Matplotlib colormap name (default: `"viridis"`) |
| `units` | string | no | Display units (e.g. `"degrees"`, `"m/s"`) |
| `value_range` | [min, max] | no | Expected value range for normalization |
| `alpha` | float | no | Default opacity 0-1 (default: 0.7) |

explore() currently receives overlays as a dict of 2D arrays passed through
the `overlay_layers` parameter in the Dataset. This group would let them
persist to disk.

xrspatial outputs (slope, aspect, curvature, hillshade, TPI, etc.) fit here
directly. The layer_name should match the xarray variable name when possible.

## wind

Wind velocity fields for the particle animation system.

**Group**: `/wind/`

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `u` | float32 | (Hg, Wg) | Zonal (east-west) wind component, m/s |
| `v` | float32 | (Hg, Wg) | Meridional (north-south) wind component, m/s |

The grid dimensions (Hg, Wg) don't have to match the elevation grid.
`fetch_wind()` typically returns a 20x20 grid interpolated across the scene
bounds.

**Attributes** on `/wind/.zattrs`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `grid_bounds` | {x0, y0, x1, y1} | World-space extent of the wind grid |
| `grid_size` | int | Grid resolution (e.g. 20) |
| `source_time` | string | ISO 8601 timestamp of the forecast data |
| `source` | string | Data source (e.g. `"open-meteo"`) |

## hydro

Hydrological flow data for the GPU particle advection system. Can be
pre-computed from terrain analysis or stored from a previous session.

**Group**: `/hydro/`

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `flow_accum` | float32 | (H, W) | Flow accumulation raster |
| `flow_dir_mfd` | float32 | (8, H, W) | MFD direction fractions per cell, 8 neighbor weights |
| `stream_order` | int32 | (H, W) | Stream order raster |
| `stream_link` | int32 | (H, W) | Optional stream link IDs |

Shape (H, W) matches the elevation grid.

**Attributes** on `/hydro/.zattrs`:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_particles` | int | 12000 | Particle count |
| `max_age` | int | 200 | Max particle lifespan in frames |
| `trail_len` | int | 20 | Position history length |
| `speed` | float | 0.75 | Velocity scale factor |
| `accum_threshold` | int | 50 | Minimum flow accumulation for particle spawning |
| `color` | [r, g, b] | [0.2, 0.5, 1.0] | Particle color |
| `alpha` | float | 0.5 | Particle opacity |
| `dot_radius` | int | 2 | Particle size in pixels |

When `hydro_data=True` is passed to explore(), the viewer computes MFD flow
from the terrain on first activation (Shift+Y). A pre-computed hydro group
lets the viewer skip that computation and start particles immediately.

## weather

Weather data for cloud rendering.

**Group**: `/weather/`

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `cloud_cover` | float32 | (Hg, Wg) | Cloud fraction 0-1 |
| `temperature` | float32 | (Hg, Wg) | Temperature in Kelvin |
| `humidity` | float32 | (Hg, Wg) | Relative humidity 0-100 |
| `pressure` | float32 | (Hg, Wg) | Atmospheric pressure in Pa |

Like wind, the grid dimensions are independent of the elevation grid.

**Attributes** on `/weather/.zattrs`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `grid_bounds` | {x0, y0, x1, y1} | World-space extent |
| `grid_size` | int | Grid resolution |
| `source_time` | string | ISO 8601 timestamp |

## camera

Initial camera state. Attributes only, no arrays.

**Group**: `/camera/`

**Attributes** on `/camera/.zattrs`:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | [x, y, z] | — | Starting position in world coords |
| `yaw` | float | 0 | Heading in degrees (0 = +X, 90 = +Y) |
| `pitch` | float | -30 | Vertical angle in degrees (negative = looking down) |
| `fov` | float | 60 | Field of view in degrees |
| `move_speed` | float | — | Units per second (auto-computed from scene if absent) |
| `look_speed` | float | 5.0 | Mouse/keyboard look sensitivity |

If `position` is absent, explore() computes a default position from the
terrain extent (existing behavior). If `position` is present, it overrides
the `start_position` parameter.

## render

Rendering defaults. Attributes only, no arrays.

**Group**: `/render/`

**Attributes** on `/render/.zattrs`:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh_type` | string | `"heightfield"` | `"heightfield"`, `"tin"`, or `"voxel"` |
| `subsample` | int | 1 | Initial resolution factor (1, 2, 4, 8) |
| `vertical_exaggeration` | float | 1.0 | Z scale multiplier |
| `colormap` | string | `"gray"` | Default colormap name |
| `color_stretch` | string | `"linear"` | `"linear"`, `"sqrt"`, `"cbrt"`, or `"log"` |
| `shadows` | bool | true | Shadow casting |
| `ambient` | float | 0.2 | Ambient light factor 0-1 |
| `sun_azimuth` | float | 225.0 | Sun angle in degrees |
| `sun_altitude` | float | 35.0 | Sun elevation in degrees |
| `fog_density` | float | 0.0 | Atmospheric fog amount 0-1 |
| `fog_color` | [r, g, b] | [0.7, 0.75, 0.85] | Fog RGB |
| `ao_enabled` | bool | true | Ambient occlusion |
| `ao_samples` | int | 4 | AO samples per frame |
| `gi_bounces` | int | 1 | Global illumination bounces (1-3) |
| `denoise` | bool | false | OptiX AI denoiser |
| `lod` | bool | false | Terrain LOD system |

These act as defaults — the user can still change everything at runtime
through keybindings. They're useful for reproducible scene setups: "this
scene looks best with sqrt stretch, 2x VE, and the terrain colormap."

## tour

Camera tour keyframes for scripted flyovers.

**Group**: `/tour/`

**Attributes** on `/tour/.zattrs`:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `fps` | int | 30 | Playback frame rate |
| `loop` | bool | false | Repeat indefinitely |

**Array**: `keyframes` — stored as a structured array or, for simplicity, as
three parallel arrays:

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `time` | float32 | (K,) | Keyframe time in seconds |
| `position` | float32 | (K, 3) | Camera xyz at each keyframe |
| `yaw` | float32 | (K,) | Heading at each keyframe |
| `pitch` | float32 | (K,) | Pitch at each keyframe |
| `fov` | float32 | (K,) | Optional per-keyframe FOV |
| `easing` | strings | (K,) | Optional easing function name per segment |

The existing tour format is a list of dicts:
```python
[
    {'time': 0,  'position': [100, 200, 50], 'yaw': 90,  'pitch': -20},
    {'time': 5,  'position': [300, 200, 80], 'yaw': 120, 'pitch': -30},
    {'time': 10, 'position': [300, 400, 60], 'yaw': 180, 'pitch': -25},
]
```

The parallel-array representation is more zarr-native and avoids encoding
JSON strings as attributes. explore() would convert between the two at
load/save time.

## observers

Pre-set observer positions for viewshed analysis and drone placement.

**Group**: `/observers/`

**Attributes** on `/observers/.zattrs`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `count` | int | Number of observers (1-8) |

Each observer is a sub-group `/observers/{slot}/` where slot is 1-8.

**Attributes** on each observer sub-group:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | [x, y, z] | — | Observer position in world coords |
| `observer_elev` | float | 0.05 | Height above terrain surface |
| `yaw` | float | 0 | Heading |
| `pitch` | float | 0 | Pitch |
| `drone_mode` | string | `"off"` | `"off"`, `"3rd_person"`, or `"fpv"` |
| `viewshed_enabled` | bool | false | Whether viewshed is active for this observer |

## Producing a scene zarr

### From quickstart (existing)

`quickstart()` already produces a zarr store with elevation + meshes:

```python
import rtxpy
rtxpy.quickstart("grand_canyon", features=["buildings", "roads", "water"])
# Creates grand_canyon_dem.zarr with elevation/ + meshes/
```

### From fetch + place (existing)

```python
dem = rtxpy.fetch_dem(bounds, "scene.zarr", source="copernicus")
ds = dem.rtx.triangulate()
ds.rtx.place_buildings(buildings_geojson)
ds.rtx.place_roads(roads_geojson)
ds.rtx.save_meshes("scene.zarr")
ds.rtx.explore(scene_zarr="scene.zarr")
```

### Full scene save (proposed)

A new `save_scene()` method on the accessor would serialize everything:

```python
ds.rtx.save_scene("scene.zarr",
    wind_data=wind,
    hydro_data=hydro,
    weather_data=weather,
    camera={'position': [100, 200, 50], 'yaw': 90, 'pitch': -20},
    render={'colormap': 'terrain', 'vertical_exaggeration': 2.0},
    tour=keyframes,
)
```

And `explore()` would accept a scene zarr directly:

```python
rtxpy.explore("scene.zarr")  # loads everything from disk
```

## Consuming a scene zarr

explore() currently loads terrain via xarray and meshes via
`load_meshes_from_zarr()`. The other groups would be loaded at viewer
init time and mapped to the corresponding `*_data` arguments.

When both a zarr group and an explicit argument are provided, the explicit
argument wins. This lets a user override a stored camera position or
colormap without editing the file.

### Load priority

1. Explicit kwargs to `explore()` (highest priority)
2. Scene zarr groups
3. Computed defaults (lowest priority)

## Chunking and alignment rules

All spatial data in the store shares a common coordinate system defined by
`spatial_ref` and the elevation grid's x/y coordinates.

The elevation chunk size sets the spatial partitioning unit. Mesh chunks,
LOD tiles, and the chunk manager's loading grid all align to this unit.
Mismatched chunk sizes will work (the mesh loader doesn't validate alignment)
but won't get the performance benefit of aligned loading.

Good chunk sizes for different DEM resolutions:

| DEM resolution | Typical extent | Chunk size | Rationale |
|----------------|---------------|------------|-----------|
| 1m lidar | 1000x1000 | 512x512 | 2 chunks, fast load |
| 10m (USGS 3DEP) | 5000x5000 | 1024x1024 | ~25 chunks, good LOD alignment |
| 30m (Copernicus) | 3000x3000 | 1024x1024 | ~9 chunks |
| 30m continental | 100000x50000 | 2048x2048 | Streaming required |

## Versioning

The `rtxpy_scene_version` attribute at the root controls format
compatibility. Version numbering follows semver-ish rules:

- Patch (1.0.x): new optional attributes on existing groups
- Minor (1.x.0): new optional groups
- Major (x.0.0): breaking changes to required groups

A reader should check the major version and refuse to load if it doesn't
match. Unknown groups and attributes should be ignored, not rejected.

## Validation

A `validate_scene(zarr_path)` function should check:

1. Root has `rtxpy_scene_version` attribute
2. `elevation` array exists with CF encoding attributes
3. `spatial_ref` exists with `crs_wkt` and `GeoTransform`
4. If `meshes` exists: has `pixel_spacing`, `elevation_shape`, `elevation_chunks`;
   each geometry group has `color`; chunk sub-groups have `vertices` + `indices`
5. If `overlays` exist: each has a `data` array with shape matching elevation
6. If `wind` exists: has `u` and `v` arrays and `grid_bounds` attribute
7. If `hydro` exists: has `flow_accum` and `flow_dir_mfd` arrays
8. If `tour` exists: has `time`, `position`, `yaw`, `pitch` arrays of equal length

Validation reports warnings for non-conforming optional data rather than
failing. The goal is "does this look like a valid scene" not "is this
bit-perfect."
