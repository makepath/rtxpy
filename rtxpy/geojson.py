"""GeoJSON parsing, CRS conversion, and mesh generation for rtxpy.

Converts GeoJSON vector features (Points, LineStrings, Polygons) into
3D triangle meshes positioned on terrain for ray-traced visualization.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _load_geojson(geojson):
    """Load GeoJSON and normalise to [(geometry, properties), ...].

    Parameters
    ----------
    geojson : str, Path, or dict
        File path or parsed GeoJSON object.

    Returns
    -------
    list of (dict, dict)
        Each entry is (geometry_dict, properties_dict).
    """
    if isinstance(geojson, (str, Path)):
        path = Path(geojson)
        if not path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {path}")
        with open(path) as f:
            geojson = json.load(f)

    if not isinstance(geojson, dict):
        raise TypeError(f"Expected dict, str, or Path, got {type(geojson)}")

    gtype = geojson.get("type")

    if gtype == "FeatureCollection":
        results = []
        for feature in geojson.get("features", []):
            results.extend(_load_geojson(feature))
        return results

    if gtype == "Feature":
        geom = geojson.get("geometry")
        props = geojson.get("properties") or {}
        if geom is None:
            return []
        return [(geom, props)]

    if gtype == "GeometryCollection":
        results = []
        for g in geojson.get("geometries", []):
            results.append((g, {}))
        return results

    # Bare geometry
    if gtype in ("Point", "MultiPoint", "LineString", "MultiLineString",
                  "Polygon", "MultiPolygon"):
        return [(geojson, {})]

    raise ValueError(f"Unsupported GeoJSON type: {gtype}")


def _flatten_multi(geometry):
    """Decompose Multi* geometry types into a list of primitive geometries.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry object.

    Returns
    -------
    list of dict
        List of primitive geometry dicts (Point, LineString, or Polygon).
    """
    gtype = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if gtype == "MultiPoint":
        return [{"type": "Point", "coordinates": c} for c in coords]
    if gtype == "MultiLineString":
        return [{"type": "LineString", "coordinates": c} for c in coords]
    if gtype == "MultiPolygon":
        return [{"type": "Polygon", "coordinates": c} for c in coords]
    # Already primitive
    return [geometry]


def _sanitize_label(value):
    """Clean a property value for use as a geometry ID component.

    Replaces non-alphanumeric characters with '-' and strips leading/trailing
    dashes.
    """
    if value is None:
        return "unknown"
    s = str(value)
    s = re.sub(r"[^a-zA-Z0-9]", "-", s)
    s = s.strip("-")
    return s or "unknown"


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def _geojson_to_world_coords(coords_lonlat, raster, terrain_data, psx, psy,
                              transformer=None, return_pixel_coords=False,
                              oob_counter=None):
    """Convert GeoJSON lon/lat coordinates to world coords with terrain Z.

    Pipeline
    --------
    GeoJSON (lon, lat) → raster CRS → pixel (col, row) → world (x, y, z)

    Parameters
    ----------
    coords_lonlat : array-like
        Nx2 or Nx3 array of [lon, lat, ...] coordinates.
    raster : xarray.DataArray
        The terrain raster (used for CRS and coordinate arrays).
    terrain_data : np.ndarray
        2D terrain elevation array (H, W).
    psx, psy : float
        Pixel spacing in world units (x and y).
    transformer : pyproj.Transformer, optional
        Pre-built CRS transformer.  Created internally if None and raster
        has CRS metadata.
    return_pixel_coords : bool, optional
        If True, return ``(world_coords, pixel_coords)`` where
        *pixel_coords* is an (N, 2) float64 array of (col, row).
        Default is False (return only world_coords).

    Returns
    -------
    np.ndarray or tuple
        (N, 3) float32 array of world coordinates (x, y, z).
        If *return_pixel_coords* is True, returns
        ``(world_coords, pixel_coords)`` instead.
    """
    coords = np.asarray(coords_lonlat, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    # Empty input guard
    if coords.size == 0 or (coords.ndim == 2 and coords.shape[1] == 0):
        empty_world = np.empty((0, 3), dtype=np.float32)
        if return_pixel_coords:
            return empty_world, np.empty((0, 2), dtype=np.float64)
        return empty_world

    lons = coords[:, 0]
    lats = coords[:, 1]

    H, W = terrain_data.shape

    # --- CRS conversion -------------------------------------------------
    has_crs = hasattr(raster, "rio") and raster.rio.crs is not None

    if has_crs:
        if transformer is None:
            try:
                from pyproj import Transformer
            except ImportError:
                raise ImportError(
                    "pyproj is required for CRS conversion. "
                    "Install with: pip install pyproj"
                )
            try:
                transformer = Transformer.from_crs(
                    "+proj=longlat +datum=WGS84 +no_defs",
                    raster.rio.crs, always_xy=True,
                )
            except Exception:
                transformer = Transformer.from_crs(
                    "EPSG:4326", raster.rio.crs, always_xy=True,
                )
        x_crs, y_crs = transformer.transform(lons, lats)
    else:
        warnings.warn(
            "Raster has no CRS metadata — assuming GeoJSON coordinates "
            "already match the raster coordinate system.",
            stacklevel=3,
        )
        x_crs = lons
        y_crs = lats

    # --- CRS coords → pixel coords -------------------------------------
    # raster x/y coordinate arrays
    x_coords = raster.coords[raster.dims[-1]].values  # columns
    y_coords = raster.coords[raster.dims[-2]].values  # rows

    dx = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1)
    dy = (y_coords[-1] - y_coords[0]) / max(len(y_coords) - 1, 1)

    cols = (x_crs - x_coords[0]) / dx
    rows = (y_crs - y_coords[0]) / dy

    # Clip to raster extent
    out_of_bounds = (
        (cols < -0.5) | (cols > W - 0.5) |
        (rows < -0.5) | (rows > H - 0.5)
    )
    n_oob = int(np.sum(out_of_bounds)) if np.any(out_of_bounds) else 0
    if n_oob > 0 and oob_counter is not None:
        oob_counter[0] += n_oob
    cols = np.clip(cols, 0, W - 1)
    rows = np.clip(rows, 0, H - 1)

    # --- Sample terrain Z (bilinear to match triangle mesh surface) ------
    x0 = np.clip(np.floor(cols).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(rows).astype(int), 0, H - 2)
    fx = cols - x0
    fy = rows - y0
    z00 = terrain_data[y0, x0].astype(np.float64)
    z10 = terrain_data[y0, x0 + 1].astype(np.float64)
    z01 = terrain_data[y0 + 1, x0].astype(np.float64)
    z11 = terrain_data[y0 + 1, x0 + 1].astype(np.float64)
    z00 = np.where(np.isnan(z00), 0.0, z00)
    z10 = np.where(np.isnan(z10), 0.0, z10)
    z01 = np.where(np.isnan(z01), 0.0, z01)
    z11 = np.where(np.isnan(z11), 0.0, z11)
    z_vals = (z00 * (1 - fx) * (1 - fy) +
              z10 * fx * (1 - fy) +
              z01 * (1 - fx) * fy +
              z11 * fx * fy)

    # --- Pixel coords → world coords -----------------------------------
    world_x = cols * psx
    world_y = rows * psy

    result = np.column_stack([world_x, world_y, z_vals]).astype(np.float32)

    if return_pixel_coords:
        pixel_coords = np.column_stack([cols, rows])
        return result, pixel_coords

    return result


def _build_transformer(raster):
    """Build a pyproj Transformer from WGS84 to the raster's CRS.

    Returns None if the raster has no CRS metadata.
    """
    has_crs = hasattr(raster, "rio") and raster.rio.crs is not None
    if not has_crs:
        return None
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError(
            "pyproj is required for CRS conversion. "
            "Install with: pip install pyproj"
        )
    try:
        return Transformer.from_crs(
            "+proj=longlat +datum=WGS84 +no_defs",
            raster.rio.crs,
            always_xy=True,
        )
    except Exception:
        return Transformer.from_crs(
            "EPSG:4326", raster.rio.crs, always_xy=True,
        )


# ---------------------------------------------------------------------------
# Mesh generators
# ---------------------------------------------------------------------------

def _make_marker_cube(size=1.0):
    """Create a unit cube mesh centered in XY with base at Z=0.

    Parameters
    ----------
    size : float
        Edge length of the cube.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array (8 verts × 3 = 24 floats).
    indices : np.ndarray
        Flat int32 index array (12 tris × 3 = 36 ints).
    """
    s = size / 2.0
    # 8 vertices: bottom 4 then top 4
    verts = np.array([
        -s, -s, 0,       s, -s, 0,       s,  s, 0,      -s,  s, 0,
        -s, -s, size,    s, -s, size,    s,  s, size,   -s,  s, size,
    ], dtype=np.float32)

    # 12 triangles (2 per face, CCW outward)
    idx = np.array([
        # top +Z
        4, 5, 6,  4, 6, 7,
        # bottom -Z
        0, 3, 2,  0, 2, 1,
        # front -Y
        0, 1, 5,  0, 5, 4,
        # back +Y
        2, 3, 7,  2, 7, 6,
        # right +X
        1, 2, 6,  1, 6, 5,
        # left -X
        3, 0, 4,  3, 4, 7,
    ], dtype=np.int32)

    return verts, idx


def _make_marker_orb(radius=1.0, stacks=8, sectors=12):
    """Create a UV sphere mesh centered at the origin.

    Parameters
    ----------
    radius : float
        Sphere radius.
    stacks : int
        Number of horizontal slices (latitude).
    sectors : int
        Number of vertical slices (longitude).

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    verts = []
    for i in range(stacks + 1):
        phi = np.pi * i / stacks  # 0 (top) to pi (bottom)
        for j in range(sectors + 1):
            theta = 2.0 * np.pi * j / sectors
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            verts.extend([x, y, z])

    indices = []
    for i in range(stacks):
        for j in range(sectors):
            v0 = i * (sectors + 1) + j
            v1 = v0 + 1
            v2 = v0 + (sectors + 1)
            v3 = v2 + 1
            if i != 0:
                indices.extend([v0, v2, v1])
            if i != stacks - 1:
                indices.extend([v1, v2, v3])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)


def _linestring_to_wall_mesh(world_coords, height, closed=False):
    """Extrude a polyline into a double-sided vertical wall mesh.

    Parameters
    ----------
    world_coords : np.ndarray
        (N, 3) array of world-space positions along the line.
    height : float
        Wall extrusion height above terrain.
    closed : bool
        If True, connect last point back to first (for polygon outlines).

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    pts = np.asarray(world_coords, dtype=np.float32)
    N = len(pts)
    if N < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    if closed and N > 2:
        # Append first point to close the ring
        pts = np.vstack([pts, pts[0:1]])
        N = len(pts)

    # 2 vertices per point (bottom + top)
    verts = np.empty(N * 2 * 3, dtype=np.float32)
    for i in range(N):
        base = i * 6
        verts[base + 0] = pts[i, 0]
        verts[base + 1] = pts[i, 1]
        verts[base + 2] = pts[i, 2]
        verts[base + 3] = pts[i, 0]
        verts[base + 4] = pts[i, 1]
        verts[base + 5] = pts[i, 2] + height

    # 4 triangles per segment (front + back for double-sided)
    num_segments = N - 1
    indices = np.empty(num_segments * 4 * 3, dtype=np.int32)
    for i in range(num_segments):
        b0 = i * 2       # bottom-left vertex index
        t0 = i * 2 + 1   # top-left
        b1 = (i + 1) * 2  # bottom-right
        t1 = (i + 1) * 2 + 1  # top-right

        tri_base = i * 12
        # Front face (CCW when viewed from outside)
        indices[tri_base + 0] = b0
        indices[tri_base + 1] = b1
        indices[tri_base + 2] = t1
        indices[tri_base + 3] = b0
        indices[tri_base + 4] = t1
        indices[tri_base + 5] = t0
        # Back face (reverse winding)
        indices[tri_base + 6] = b0
        indices[tri_base + 7] = t1
        indices[tri_base + 8] = b1
        indices[tri_base + 9] = b0
        indices[tri_base + 10] = t0
        indices[tri_base + 11] = t1

    return verts, indices


def _densify_on_terrain(world_coords, terrain_data, psx, psy, step=1.0):
    """Resample a polyline at regular intervals, re-sampling terrain Z.

    Adds intermediate vertices wherever a segment spans more than *step*
    pixels so the path closely follows the terrain surface.

    Parameters
    ----------
    world_coords : np.ndarray
        (N, 3) world-space positions (x=col*psx, y=row*psy, z=terrain_z).
    terrain_data : np.ndarray
        2D terrain elevation array (H, W).
    psx, psy : float
        Pixel spacing in world units.
    step : float
        Maximum distance in pixels between consecutive output vertices.

    Returns
    -------
    np.ndarray
        Densified (M, 3) float32 world-coordinate array with M >= N.
    """
    pts = np.asarray(world_coords, dtype=np.float64)
    if len(pts) < 2:
        return np.asarray(world_coords, dtype=np.float32)

    H, W = terrain_data.shape
    result = [pts[0]]

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        # Distance in pixel space
        dx_px = (p1[0] - p0[0]) / psx
        dy_px = (p1[1] - p0[1]) / psy
        dist_px = np.sqrt(dx_px ** 2 + dy_px ** 2)

        n_sub = max(1, int(np.ceil(dist_px / step)))

        for j in range(1, n_sub + 1):
            t = j / n_sub
            wx = p0[0] + t * (p1[0] - p0[0])
            wy = p0[1] + t * (p1[1] - p0[1])
            # Bilinear interpolation to match terrain triangle surface
            px = wx / psx
            py = wy / psy
            x0 = int(np.clip(np.floor(px), 0, W - 2))
            y0 = int(np.clip(np.floor(py), 0, H - 2))
            x1 = x0 + 1
            y1 = y0 + 1
            fx = px - x0
            fy = py - y0
            z00 = float(terrain_data[y0, x0])
            z10 = float(terrain_data[y0, x1])
            z01 = float(terrain_data[y1, x0])
            z11 = float(terrain_data[y1, x1])
            # Replace any NaN corners with 0
            if np.isnan(z00): z00 = 0.0
            if np.isnan(z10): z10 = 0.0
            if np.isnan(z01): z01 = 0.0
            if np.isnan(z11): z11 = 0.0
            z = (z00 * (1 - fx) * (1 - fy) +
                 z10 * fx * (1 - fy) +
                 z01 * (1 - fx) * fy +
                 z11 * fx * fy)
            result.append([wx, wy, z])

    return np.array(result, dtype=np.float32)


def _linestring_to_tube_mesh(world_coords, radius=0.5, hover=1.0,
                              segments=6, closed=False):
    """Extrude a polyline into a tube mesh hovering above terrain.

    Parameters
    ----------
    world_coords : np.ndarray
        (N, 3) array of world-space positions along the line.
    radius : float
        Tube cross-section radius.
    hover : float
        Height offset above terrain surface.
    segments : int
        Number of sides for the tube cross-section (6 = hexagonal).
    closed : bool
        If True, connect last point back to first (for polygon outlines).

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    pts = np.asarray(world_coords, dtype=np.float32).copy()
    N = len(pts)
    if N < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    if closed and N > 2:
        if not np.allclose(pts[0], pts[-1], atol=1e-3):
            pts = np.vstack([pts, pts[0:1]])
            N = len(pts)

    # Hover above terrain
    pts[:, 2] += hover

    # Tangent at each point (central difference, forward/backward at ends)
    tangents = np.empty_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if N > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(norms, 1e-8)

    S = segments
    angles = np.linspace(0, 2 * np.pi, S, endpoint=False)
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)

    up = np.array([0, 0, 1], dtype=np.float32)

    verts = np.empty((N * S, 3), dtype=np.float32)
    for i in range(N):
        t = tangents[i]
        right = np.cross(t, up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.cross(t, np.array([1, 0, 0], dtype=np.float32))
            rn = np.linalg.norm(right)
        right /= max(rn, 1e-8)
        frame_up = np.cross(right, t)
        frame_up /= max(np.linalg.norm(frame_up), 1e-8)

        verts[i * S:(i + 1) * S] = (
            pts[i]
            + radius * (np.outer(cos_a, right) + np.outer(sin_a, frame_up))
        )

    # Connect adjacent rings with quads (2 tris each)
    num_quads = (N - 1) * S
    tris = np.empty((num_quads * 2, 3), dtype=np.int32)
    for i in range(N - 1):
        for j in range(S):
            jn = (j + 1) % S
            v0 = i * S + j
            v1 = i * S + jn
            v2 = (i + 1) * S + j
            v3 = (i + 1) * S + jn
            qi = (i * S + j) * 2
            tris[qi] = [v0, v2, v3]
            tris[qi + 1] = [v0, v3, v1]

    return verts.ravel(), tris.ravel()


def _linestring_to_ribbon_mesh(world_coords, width=1.0, hover=0.2,
                                closed=False):
    """Extrude a polyline into a flat ribbon mesh on the terrain surface.

    Parameters
    ----------
    world_coords : np.ndarray
        (N, 3) array of world-space positions along the line.
    width : float
        Ribbon half-width (total width = 2 * width).
    hover : float
        Small Z offset above terrain to avoid z-fighting.
    closed : bool
        If True, connect last point back to first.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    pts = np.asarray(world_coords, dtype=np.float32).copy()
    N = len(pts)
    if N < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    if closed and N > 2:
        if not np.allclose(pts[0], pts[-1], atol=1e-3):
            pts = np.vstack([pts, pts[0:1]])
            N = len(pts)

    # Hover above terrain
    pts[:, 2] += hover

    # Tangent at each point (central difference, forward/backward at ends)
    tangents = np.empty((N, 2), dtype=np.float32)
    tangents[0] = pts[1, :2] - pts[0, :2]
    tangents[-1] = pts[-1, :2] - pts[-2, :2]
    if N > 2:
        tangents[1:-1] = pts[2:, :2] - pts[:-2, :2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(norms, 1e-8)

    # Perpendicular in XY plane (rotate 90 degrees)
    perp = np.empty_like(tangents)
    perp[:, 0] = -tangents[:, 1]
    perp[:, 1] = tangents[:, 0]

    # Two vertices per point: left and right of the centerline
    verts = np.empty((N * 2, 3), dtype=np.float32)
    verts[0::2] = pts + width * np.column_stack([perp, np.zeros(N)])
    verts[1::2] = pts - width * np.column_stack([perp, np.zeros(N)])

    # Quad strip: two triangles per segment
    num_quads = N - 1
    tris = np.empty((num_quads * 2, 3), dtype=np.int32)
    for i in range(num_quads):
        l0 = i * 2
        r0 = i * 2 + 1
        l1 = (i + 1) * 2
        r1 = (i + 1) * 2 + 1
        qi = i * 2
        tris[qi] = [l0, r1, l1]
        tris[qi + 1] = [l0, r0, r1]

    return verts.ravel(), tris.ravel()


def _polygon_to_ribbon_mesh(rings_world_coords, width=1.0, hover=0.2):
    """Build ribbon meshes for a polygon's exterior and interior rings.

    Parameters
    ----------
    rings_world_coords : list of np.ndarray
        Each element is an (N, 3) array for one ring.
    width : float
        Ribbon half-width.
    hover : float
        Small Z offset above terrain.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    all_verts = []
    all_indices = []
    vert_offset = 0

    for ring_coords in rings_world_coords:
        v, idx = _linestring_to_ribbon_mesh(
            ring_coords, width=width, hover=hover, closed=True,
        )
        if len(v) == 0:
            continue
        all_verts.append(v)
        all_indices.append(idx + vert_offset)
        vert_offset += len(v) // 3

    if not all_verts:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.concatenate(all_verts), np.concatenate(all_indices)


def _polygon_to_mesh(rings_world_coords, height):
    """Build wall meshes for a polygon's exterior and interior rings.

    Parameters
    ----------
    rings_world_coords : list of np.ndarray
        Each element is an (N, 3) array for one ring.
        First is the exterior ring, rest are holes.
    height : float
        Wall extrusion height.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    all_verts = []
    all_indices = []
    vert_offset = 0

    for ring_coords in rings_world_coords:
        v, idx = _linestring_to_wall_mesh(ring_coords, height, closed=True)
        if len(v) == 0:
            continue
        all_verts.append(v)
        # Offset indices
        all_indices.append(idx + vert_offset)
        vert_offset += len(v) // 3  # number of vertices

    if not all_verts:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.concatenate(all_verts), np.concatenate(all_indices)


def _polygon_to_tube_mesh(rings_world_coords, radius=0.5, hover=1.0,
                           segments=6):
    """Build tube meshes for a polygon's exterior and interior rings.

    Parameters
    ----------
    rings_world_coords : list of np.ndarray
        Each element is an (N, 3) array for one ring.
    radius : float
        Tube cross-section radius.
    hover : float
        Height offset above terrain surface.
    segments : int
        Number of sides for the tube cross-section.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 vertex array.
    indices : np.ndarray
        Flat int32 index array.
    """
    all_verts = []
    all_indices = []
    vert_offset = 0

    for ring_coords in rings_world_coords:
        v, idx = _linestring_to_tube_mesh(
            ring_coords, radius=radius, hover=hover,
            segments=segments, closed=True,
        )
        if len(v) == 0:
            continue
        all_verts.append(v)
        all_indices.append(idx + vert_offset)
        vert_offset += len(v) // 3

    if not all_verts:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.concatenate(all_verts), np.concatenate(all_indices)


def _ear_clip_2d(xy):
    """Triangulate a simple polygon using ear clipping (2D).

    Parameters
    ----------
    xy : np.ndarray
        (N, 2) array of 2D polygon vertices (no closing duplicate).

    Returns
    -------
    list of (int, int, int)
        Triangle index triples referencing the input vertex array.
    """
    N = len(xy)
    if N < 3:
        return []
    if N == 3:
        return [(0, 1, 2)]

    # Work with a mutable index list
    idx = list(range(N))
    tris = []

    # Signed area to determine winding
    def _signed_area_2(indices):
        s = 0.0
        n = len(indices)
        for k in range(n):
            i0 = indices[k]
            i1 = indices[(k + 1) % n]
            s += xy[i0, 0] * xy[i1, 1] - xy[i1, 0] * xy[i0, 1]
        return s

    # Cross product of vectors (b-a) and (c-a)
    def _cross2(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def _point_in_tri(p, a, b, c):
        d1 = _cross2(a, b, p)
        d2 = _cross2(b, c, p)
        d3 = _cross2(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    ccw = _signed_area_2(idx) > 0

    max_iter = N * N  # safety bound
    while len(idx) > 2 and max_iter > 0:
        max_iter -= 1
        found_ear = False
        n = len(idx)
        for k in range(n):
            prev_i = idx[(k - 1) % n]
            curr_i = idx[k]
            next_i = idx[(k + 1) % n]
            cross = _cross2(xy[prev_i], xy[curr_i], xy[next_i])
            # Ear vertex must be convex
            if (ccw and cross <= 0) or (not ccw and cross >= 0):
                continue
            # No other vertex inside this triangle
            ear_ok = True
            for m in range(n):
                mi = idx[m]
                if mi == prev_i or mi == curr_i or mi == next_i:
                    continue
                if _point_in_tri(xy[mi], xy[prev_i], xy[curr_i], xy[next_i]):
                    ear_ok = False
                    break
            if ear_ok:
                tris.append((prev_i, curr_i, next_i))
                idx.pop(k)
                found_ear = True
                break
        if not found_ear:
            # Degenerate polygon — fall back to fan from first vertex
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            break
    return tris


def _extrude_polygon(rings_world_coords, height):
    """Extrude a polygon into solid 3D geometry (walls + roof cap).

    Uses the exterior ring only (holes are ignored).  Walls are
    double-sided (both winding orders) for correct lighting from any
    viewing angle.  The roof cap uses ear-clipping triangulation to
    handle concave footprints (L-shaped buildings, etc.).

    Parameters
    ----------
    rings_world_coords : list of np.ndarray
        Each element is an (N, 3) array.  Only the first (exterior) ring
        is used.
    height : float
        Extrusion height above each vertex's z coordinate.

    Returns
    -------
    vertices : np.ndarray, flat float32
    indices  : np.ndarray, flat int32
    """
    ring = np.asarray(rings_world_coords[0], dtype=np.float32)
    if len(ring) < 3:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    # Strip duplicate closing vertex
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    N = len(ring)
    if N < 3:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    # Vertices: bottom ring (0..N-1), top ring (N..2N-1)
    top = ring.copy()
    top[:, 2] += height

    verts = np.empty(2 * N * 3, dtype=np.float32)
    verts[: N * 3] = ring.ravel()
    verts[N * 3: 2 * N * 3] = top.ravel()

    # --- Walls: double-sided (4 tris per edge) ---
    wall_tris = []
    for i in range(N):
        j = (i + 1) % N
        b0, b1, t0, t1 = i, j, i + N, j + N
        # Front face
        wall_tris.extend([b0, b1, t1])
        wall_tris.extend([b0, t1, t0])
        # Back face (reverse winding)
        wall_tris.extend([b0, t1, b1])
        wall_tris.extend([b0, t0, t1])

    # --- Roof cap: ear-clipping on 2D projection ---
    roof_xy = top[:, :2]  # project to XY for triangulation
    roof_ears = _ear_clip_2d(roof_xy)
    roof_tris = []
    for a, b, c in roof_ears:
        # Both sides of the roof cap
        roof_tris.extend([a + N, b + N, c + N])
        roof_tris.extend([a + N, c + N, b + N])

    idxs = np.array(wall_tris + roof_tris, dtype=np.int32)
    return verts, idxs


# ---------------------------------------------------------------------------
# Polygon fill utilities
# ---------------------------------------------------------------------------

def _points_in_polygon(points_xy, ring_xy):
    """Ray-casting point-in-polygon test.

    Parameters
    ----------
    points_xy : np.ndarray
        (M, 2) array of test point coordinates.
    ring_xy : np.ndarray
        (N, 2) array of polygon ring vertices (closed or unclosed).

    Returns
    -------
    np.ndarray
        Boolean mask of shape (M,).
    """
    ring = np.asarray(ring_xy, dtype=np.float64)
    pts = np.asarray(points_xy, dtype=np.float64)

    n = len(ring)
    inside = np.zeros(len(pts), dtype=bool)

    for i in range(n):
        j = (i + 1) % n
        xi, yi = ring[i]
        xj, yj = ring[j]

        # Test if horizontal ray from point intersects edge
        cond = ((yi > pts[:, 1]) != (yj > pts[:, 1]))
        if not np.any(cond):
            continue
        x_intersect = xi + (pts[cond, 1] - yi) / (yj - yi) * (xj - xi)
        crossing = pts[cond, 0] < x_intersect
        inside[cond] ^= crossing

    return inside


def _scatter_in_polygon(ring_pixel_coords, spacing, H, W):
    """Generate a grid of points inside a polygon.

    Parameters
    ----------
    ring_pixel_coords : np.ndarray
        (N, 2) pixel-space coordinates of the polygon exterior ring.
    spacing : float
        Grid spacing in pixel units.
    H, W : int
        Raster dimensions (for clipping).

    Returns
    -------
    np.ndarray
        (M, 2) array of pixel-space (col, row) positions inside the polygon.
    """
    ring = np.asarray(ring_pixel_coords, dtype=np.float64)

    xmin, ymin = ring.min(axis=0)
    xmax, ymax = ring.max(axis=0)

    # Clip to raster bounds
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, W - 1)
    ymax = min(ymax, H - 1)

    # Generate grid candidates
    xs = np.arange(xmin + spacing / 2, xmax, spacing)
    ys = np.arange(ymin + spacing / 2, ymax, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)

    grid_x, grid_y = np.meshgrid(xs, ys)
    candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Filter by polygon
    mask = _points_in_polygon(candidates, ring)
    result = candidates[mask]

    if len(result) > 10000:
        warnings.warn(
            f"Polygon fill generates {len(result)} instances "
            f"(spacing={spacing}). Consider increasing fill_spacing.",
            stacklevel=3,
        )

    return result
