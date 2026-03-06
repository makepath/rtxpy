"""Level-of-detail utilities for terrain and instanced geometry.

Provides LOD level computation, mesh simplification via quadric
decimation, and LOD chain generation for multi-resolution rendering.
"""

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
