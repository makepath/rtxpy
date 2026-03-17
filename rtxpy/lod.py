"""Level-of-detail utilities for terrain and instanced geometry.

Provides LOD level computation, mesh simplification via quadric
decimation, LOD chain generation, and box-filter downsampling for
multi-resolution rendering.
"""

import warnings

import numpy as np


def compute_lod_level(distance, lod_distances):
    """Return the LOD level for a given distance.

    Parameters
    ----------
    distance : float
        Distance from camera to object or tile center.
    lod_distances : list of float
        Ascending distance thresholds defining LOD transitions.
        For *N* thresholds, LOD levels range from 0 to *N*.
        Level 0 is the highest detail (closest to camera).

    Returns
    -------
    int
        LOD level (0 = highest detail).

    Examples
    --------
    >>> compute_lod_level(100, [500, 1000, 2000])
    0
    >>> compute_lod_level(750, [500, 1000, 2000])
    1
    >>> compute_lod_level(3000, [500, 1000, 2000])
    3
    """
    for i, threshold in enumerate(lod_distances):
        if distance < threshold:
            return i
    return len(lod_distances)


def compute_lod_level_with_hysteresis(distance, lod_distances, prev_lod,
                                      hysteresis=0.2):
    """Return the LOD level with hysteresis to prevent popping.

    Uses wider thresholds for downgrading (increasing LOD level) than
    upgrading (decreasing LOD level), creating a dead zone around each
    transition boundary.

    Parameters
    ----------
    distance : float
        Distance from camera to object or tile center.
    lod_distances : list of float
        Ascending distance thresholds defining LOD transitions.
    prev_lod : int
        Previous LOD level for this tile (-1 if never assigned).
    hysteresis : float
        Fractional band width.  A tile must move ``hysteresis`` past the
        threshold before switching.  E.g. 0.2 means 20% beyond.

    Returns
    -------
    int
        LOD level (0 = highest detail).
    """
    if prev_lod < 0:
        return compute_lod_level(distance, lod_distances)

    # Upgrade (reduce LOD number = more detail): require distance to be
    # clearly inside the better band (threshold * (1 - hysteresis)).
    # Downgrade (increase LOD number = less detail): require distance to
    # exceed threshold * (1 + hysteresis).
    new_lod = compute_lod_level(distance, lod_distances)
    if new_lod < prev_lod:
        # Upgrading — use tighter threshold
        lod_tight = compute_lod_level(
            distance / (1.0 - hysteresis), lod_distances)
        if lod_tight < prev_lod:
            return new_lod
        return prev_lod
    elif new_lod > prev_lod:
        # Downgrading — use looser threshold
        lod_loose = compute_lod_level(
            distance / (1.0 + hysteresis), lod_distances)
        if lod_loose > prev_lod:
            return new_lod
        return prev_lod
    return prev_lod


def compute_lod_distances(tile_diagonal, factor=3.0, max_lod=3):
    """Compute LOD distance thresholds from tile geometry.

    Each LOD transition occurs at ``tile_diagonal * factor * 2^level``.

    Parameters
    ----------
    tile_diagonal : float
        Diagonal size of a terrain tile in world units.
    factor : float
        Base multiplier for the first threshold.
    max_lod : int
        Number of LOD transitions (max LOD level = max_lod).

    Returns
    -------
    list of float
        Distance thresholds for LOD 0 → 1, 1 → 2, etc.
    """
    return [tile_diagonal * factor * (2 ** i) for i in range(max_lod)]


def compute_tile_roughness(tile_2d):
    """Compute terrain roughness as std deviation from bilinear fit.

    Fits a bilinear surface through the tile's four corners and
    measures how much the actual terrain deviates.  Flat or planar
    tiles score near zero; jagged ridgelines score high.

    Parameters
    ----------
    tile_2d : np.ndarray
        2D elevation array, shape ``(H, W)``.  May contain NaN.

    Returns
    -------
    float
        Standard deviation of elevation residuals (world-space units).
        Returns 0.0 for degenerate tiles (all NaN or < 2×2).
    """
    h, w = tile_2d.shape
    if h < 2 or w < 2:
        return 0.0

    # Corner elevations — fall back to tile nanmean if a corner is NaN
    corners = np.array([tile_2d[0, 0], tile_2d[0, -1],
                        tile_2d[-1, 0], tile_2d[-1, -1]])
    valid = ~np.isnan(corners)
    if not np.any(valid):
        return 0.0
    fill = float(np.mean(corners[valid]))
    c00 = corners[0] if valid[0] else fill
    c01 = corners[1] if valid[1] else fill
    c10 = corners[2] if valid[2] else fill
    c11 = corners[3] if valid[3] else fill

    # Bilinear interpolation between corners
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    bilinear = (c00 * (1 - xs) * (1 - ys) + c01 * xs * (1 - ys)
                + c10 * (1 - xs) * ys + c11 * xs * ys)

    residuals = tile_2d - bilinear
    return float(np.nanstd(residuals))


def simplify_mesh(vertices, indices, ratio):
    """Simplify a triangle mesh using quadric decimation.

    Requires ``trimesh`` with ``simplify_quadric_decimation`` support.
    Falls back to returning the original mesh if trimesh is unavailable
    or simplification fails.

    Parameters
    ----------
    vertices : np.ndarray
        Flat float32 vertex buffer, shape ``(N*3,)``.
    indices : np.ndarray
        Flat int32 index buffer, shape ``(M*3,)``.
    ratio : float
        Fraction of triangles to keep (0.0 to 1.0).

    Returns
    -------
    simplified_vertices : np.ndarray
        Flat float32 vertex buffer.
    simplified_indices : np.ndarray
        Flat int32 index buffer.
    """
    if ratio >= 1.0:
        return vertices, indices

    try:
        import trimesh
    except ImportError:
        return vertices, indices

    verts_2d = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces_2d = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    num_target = max(4, int(len(faces_2d) * ratio))

    try:
        mesh = trimesh.Trimesh(vertices=verts_2d, faces=faces_2d,
                               process=False)
        simplified = mesh.simplify_quadric_decimation(num_target)
        return (simplified.vertices.astype(np.float32).flatten(),
                simplified.faces.astype(np.int32).flatten())
    except Exception:
        return vertices, indices


def build_lod_chain(vertices, indices, ratios=(1.0, 0.5, 0.25, 0.1)):
    """Build a chain of progressively simplified meshes.

    Parameters
    ----------
    vertices : np.ndarray
        Full-detail flat float32 vertex buffer.
    indices : np.ndarray
        Full-detail flat int32 index buffer.
    ratios : tuple of float
        Triangle retention ratios per LOD level.  The first entry
        should be 1.0 (original mesh).

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        ``(vertices, indices)`` tuples, one per LOD level.
    """
    chain = [(vertices.copy(), indices.copy())]
    for ratio in ratios[1:]:
        v, i = simplify_mesh(vertices, indices, ratio)
        chain.append((v, i))
    return chain


def box_downsample_2x(arr):
    """Downsample a 2D array by 2x using NaN-aware box averaging.

    Trims to even dimensions, then averages each 2x2 block.  NaN
    values are ignored (``nanmean``), so water/no-data pixels don't
    contaminate adjacent land pixels.

    Parameters
    ----------
    arr : np.ndarray
        2D array to downsample.

    Returns
    -------
    np.ndarray
        Downsampled array, shape ``(H//2, W//2)``.
    """
    H, W = arr.shape
    hh = H - H % 2
    ww = W - W % 2
    if hh < 2 or ww < 2:
        return arr.copy()
    block = arr[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = np.nanmean(block, axis=(1, 3)).astype(arr.dtype)
    return out
