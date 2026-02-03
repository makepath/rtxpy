"""Mesh utilities for terrain triangulation, STL export, and OBJ loading.

This module provides functions for converting raster terrain data into
triangle meshes suitable for ray tracing, for exporting meshes to STL format,
and for loading external mesh files in OBJ format.
"""

import numba as nb
from numba import cuda
import numpy as np
from pathlib import Path

from .rtx import has_cupy

if has_cupy:
    import cupy


@cuda.jit
def _triangulate_terrain_gpu(verts, triangles, data, H, W, scale, stride):
    """GPU kernel for terrain triangulation."""
    globalId = stride + cuda.grid(1)
    if globalId < W * H:
        h = globalId // W
        w = globalId % W
        meshMapIndex = h * W + w

        val = data[h, w]

        offset = 3 * meshMapIndex
        verts[offset] = w
        verts[offset + 1] = h
        verts[offset + 2] = val * scale

        if w != W - 1 and h != H - 1:
            offset = 6 * (h * (W - 1) + w)
            triangles[offset + 0] = np.int32(meshMapIndex + W)
            triangles[offset + 1] = np.int32(meshMapIndex + W + 1)
            triangles[offset + 2] = np.int32(meshMapIndex)
            triangles[offset + 3] = np.int32(meshMapIndex + W + 1)
            triangles[offset + 4] = np.int32(meshMapIndex + 1)
            triangles[offset + 5] = np.int32(meshMapIndex)


@nb.njit(parallel=True)
def _triangulate_terrain_cpu(verts, triangles, data, H, W, scale):
    """CPU implementation of terrain triangulation using numba."""
    for h in nb.prange(H):
        for w in range(W):
            meshMapIndex = h * W + w

            val = data[h, w]

            offset = 3 * meshMapIndex
            verts[offset] = w
            verts[offset + 1] = h
            verts[offset + 2] = val * scale

            if w != W - 1 and h != H - 1:
                offset = 6 * (h * (W - 1) + w)
                triangles[offset + 0] = np.int32(meshMapIndex + W)
                triangles[offset + 1] = np.int32(meshMapIndex + W + 1)
                triangles[offset + 2] = np.int32(meshMapIndex)
                triangles[offset + 3] = np.int32(meshMapIndex + W + 1)
                triangles[offset + 4] = np.int32(meshMapIndex + 1)
                triangles[offset + 5] = np.int32(meshMapIndex)


def triangulate_terrain(verts, triangles, terrain, scale=1.0):
    """Convert a 2D terrain array into a triangle mesh.

    This function populates the provided vertex and triangle buffers with
    mesh data representing the terrain surface. Each cell in the terrain
    becomes a vertex, and adjacent cells are connected with two triangles.

    Parameters
    ----------
    verts : array-like
        Output vertex buffer of shape (H * W * 3,) as float32.
        Will be populated with x, y, z coordinates for each vertex.
    triangles : array-like
        Output triangle index buffer of shape ((H-1) * (W-1) * 2 * 3,) as int32.
        Will be populated with vertex indices for each triangle.
    terrain : array-like
        2D array of elevation values with shape (H, W). Can be a numpy array,
        cupy array, or xarray DataArray. If an xarray DataArray is passed,
        its underlying data array will be used.
    scale : float, optional
        Scale factor applied to elevation values. Default is 1.0.

    Returns
    -------
    int
        0 on success.

    Notes
    -----
    The function automatically selects CPU or GPU execution based on the
    input array type. If terrain is a cupy array, GPU execution is used;
    otherwise, CPU execution with parallel numba is used.
    """
    # Handle xarray DataArray by extracting underlying data
    # Check for xarray DataArray specifically (has 'dims' attribute)
    if hasattr(terrain, 'dims') and hasattr(terrain, 'coords'):
        data = terrain.data  # xarray stores cupy/numpy array in .data
    else:
        data = terrain
    H, W = terrain.shape

    if isinstance(data, np.ndarray):
        _triangulate_terrain_cpu(verts, triangles, data, H, W, scale)
    elif has_cupy and isinstance(data, cupy.ndarray):
        jobSize = H * W
        blockdim = 1024
        griddim = (jobSize + blockdim - 1) // 1024
        d = 100
        offset = 0
        while jobSize > 0:
            batch = min(d, griddim)
            _triangulate_terrain_gpu[batch, blockdim](
                verts, triangles, data, H, W, scale, offset
            )
            offset += batch * blockdim
            jobSize -= batch * blockdim
    else:
        raise TypeError(
            f"Unsupported terrain data type: {type(data)}. "
            "Expected numpy.ndarray or cupy.ndarray."
        )

    return 0


@nb.jit(nopython=True)
def _fill_stl_contents(content, verts, triangles, numTris):
    """Fill STL binary content from mesh data."""
    v = np.empty(12, np.float32)
    pad = np.zeros(2, np.int8)
    for i in range(numTris):
        t0 = triangles[3 * i + 0]
        t1 = triangles[3 * i + 1]
        t2 = triangles[3 * i + 2]
        # Normal (set to zero, viewers recalculate)
        v[3 * 0 + 0] = 0
        v[3 * 0 + 1] = 0
        v[3 * 0 + 2] = 0
        # Vertex 0
        v[3 * 1 + 0] = verts[3 * t0 + 0]
        v[3 * 1 + 1] = verts[3 * t0 + 1]
        v[3 * 1 + 2] = verts[3 * t0 + 2]
        # Vertex 1
        v[3 * 2 + 0] = verts[3 * t1 + 0]
        v[3 * 2 + 1] = verts[3 * t1 + 1]
        v[3 * 2 + 2] = verts[3 * t1 + 2]
        # Vertex 2
        v[3 * 3 + 0] = verts[3 * t2 + 0]
        v[3 * 3 + 1] = verts[3 * t2 + 1]
        v[3 * 3 + 2] = verts[3 * t2 + 2]

        offset = 50 * i
        content[offset:offset + 48] = v.view(np.uint8)
        content[offset + 48:offset + 50] = pad


def write_stl(filename, verts, triangles):
    """Save a triangle mesh to a binary STL file.

    STL is a simple, widely-supported 3D format. Windows has a built-in
    STL viewer, and most 3D applications can open STL files.

    Parameters
    ----------
    filename : str
        Output file path. Should end with '.stl'.
    verts : array-like
        Vertex buffer as float32 with 3 values per vertex (x, y, z).
    triangles : array-like
        Triangle index buffer as int32 with 3 indices per triangle.

    Notes
    -----
    If cupy arrays are provided, they will be automatically converted
    to numpy arrays for writing.
    """
    ib = triangles
    vb = verts

    # Convert cupy arrays to numpy if needed
    if has_cupy:
        if isinstance(ib, cupy.ndarray):
            ib = cupy.asnumpy(ib)
        if isinstance(vb, cupy.ndarray):
            vb = cupy.asnumpy(vb)

    header = np.zeros(80, np.uint8)
    nf = np.empty(1, np.uint32)
    numTris = triangles.shape[0] // 3
    nf[0] = numTris

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(nf)

        # Size of 1 triangle in STL is 50 bytes:
        # 12 floats (each 4 bytes) = 48 bytes + 2 bytes padding
        content = np.empty(numTris * 50, np.uint8)
        _fill_stl_contents(content, vb, ib, numTris)
        f.write(content)


def load_obj(filepath, scale=1.0, swap_yz=False):
    """Load a Wavefront OBJ file and return vertices and indices for ray tracing.

    This function parses OBJ files and converts them to the flattened vertex
    and index arrays expected by the RTX class. Supports triangular and
    quadrilateral faces (quads are automatically triangulated).

    Parameters
    ----------
    filepath : str or Path
        Path to the OBJ file to load.
    scale : float, optional
        Scale factor applied to all vertex coordinates. Default is 1.0.
    swap_yz : bool, optional
        If True, swap Y and Z coordinates. Useful when OBJ uses Y-up convention
        but the scene uses Z-up (common for terrain/DEM scenes). Default is False.

    Returns
    -------
    vertices : numpy.ndarray
        Flattened float32 array of vertex positions with shape (N*3,),
        where N is the number of vertices. Layout is [x0, y0, z0, x1, y1, z1, ...].
    indices : numpy.ndarray
        Flattened int32 array of triangle indices with shape (M*3,),
        where M is the number of triangles. Layout is [i0, i1, i2, i3, i4, i5, ...].

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file contains no valid geometry or has faces with fewer than
        3 vertices.

    Examples
    --------
    Load an OBJ file and add it to a scene:

    >>> from rtxpy import RTX, load_obj
    >>> verts, indices = load_obj("building.obj", scale=0.1)
    >>> rtx = RTX()
    >>> rtx.add_geometry("building", verts, indices)

    Load with coordinate swap for Z-up terrain scenes:

    >>> verts, indices = load_obj("model.obj", swap_yz=True)

    Notes
    -----
    - OBJ files use 1-based indexing; this function converts to 0-based.
    - Only vertex positions (v) and faces (f) are parsed. Texture coordinates (vt),
      normals (vn), materials, and other OBJ features are ignored.
    - Faces with more than 4 vertices are triangulated using a fan pattern.
    - Negative face indices (relative references) are supported.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v' and len(parts) >= 4:
                # Vertex: v x y z [w]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if swap_yz:
                    y, z = z, y
                vertices.append([x * scale, y * scale, z * scale])

            elif parts[0] == 'f' and len(parts) >= 4:
                # Face: f v1 v2 v3 ... or f v1/vt1/vn1 v2/vt2/vn2 ...
                face_indices = []
                for p in parts[1:]:
                    # Handle v, v/vt, v/vt/vn, or v//vn formats
                    idx_str = p.split('/')[0]
                    idx = int(idx_str)
                    # OBJ uses 1-based indexing, convert to 0-based
                    # Negative indices are relative to current vertex count
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    face_indices.append(idx)

                if len(face_indices) < 3:
                    continue

                # Triangulate: fan triangulation for polygons
                # Triangle: [0, 1, 2]
                # Quad: [0, 1, 2], [0, 2, 3]
                # Pentagon: [0, 1, 2], [0, 2, 3], [0, 3, 4]
                for i in range(1, len(face_indices) - 1):
                    faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])

    if not vertices:
        raise ValueError(f"No vertices found in OBJ file: {filepath}")
    if not faces:
        raise ValueError(f"No faces found in OBJ file: {filepath}")

    vertices_array = np.array(vertices, dtype=np.float32).flatten()
    indices_array = np.array(faces, dtype=np.int32).flatten()

    return vertices_array, indices_array


def make_transform(x=0.0, y=0.0, z=0.0, scale=1.0, rotation_z=0.0):
    """Create a 3x4 affine transform matrix for positioning geometry.

    This is a convenience function for creating transform matrices to use
    with RTX.add_geometry(). The transform applies scale, then rotation
    around the Z axis, then translation.

    Parameters
    ----------
    x : float, optional
        X translation. Default is 0.0.
    y : float, optional
        Y translation. Default is 0.0.
    z : float, optional
        Z translation. Default is 0.0.
    scale : float, optional
        Uniform scale factor. Default is 1.0.
    rotation_z : float, optional
        Rotation around Z axis in radians. Default is 0.0.

    Returns
    -------
    list
        12-float list representing a 3x4 row-major affine transform matrix.
        Format: [Xx, Xy, Xz, Tx, Yx, Yy, Yz, Ty, Zx, Zy, Zz, Tz]

    Examples
    --------
    Simple translation:

    >>> transform = make_transform(x=100, y=200, z=50)
    >>> rtx.add_geometry("tower", verts, indices, transform=transform)

    Scale and translate:

    >>> transform = make_transform(x=100, y=200, z=50, scale=0.1)

    Rotate 90 degrees and translate:

    >>> import math
    >>> transform = make_transform(x=100, y=200, rotation_z=math.pi/2)
    """
    import math
    c = math.cos(rotation_z)
    s = math.sin(rotation_z)

    # Scale * Rotation * Translation (applied right to left)
    # Rotation matrix around Z: [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return [
        scale * c, -scale * s, 0.0, x,
        scale * s,  scale * c, 0.0, y,
        0.0,        0.0,       scale, z,
    ]


def make_transforms_on_terrain(positions, terrain, scale=1.0, rotation_z=0.0):
    """Create transforms for placing objects at multiple positions on terrain.

    This convenience function samples terrain elevation at each (x, y) position
    and creates transform matrices suitable for RTX.add_geometry(transforms=...).

    Parameters
    ----------
    positions : array-like
        Sequence of (x, y) coordinate pairs where objects should be placed.
        Can be a list of tuples, numpy array of shape (N, 2), etc.
    terrain : array-like
        2D array of elevation values with shape (H, W). The terrain uses
        pixel coordinates where position (x, y) samples terrain[int(y), int(x)].
    scale : float, optional
        Uniform scale factor applied to all transforms. Default is 1.0.
    rotation_z : float or array-like, optional
        Rotation around Z axis in radians. Can be a single value applied to
        all instances, or an array of rotations (one per position). Default is 0.0.

    Returns
    -------
    list
        List of 12-float transform matrices, one per position.

    Examples
    --------
    Place cell towers at multiple locations:

    >>> tower_verts, tower_indices = load_obj("cell_tower.obj")
    >>> positions = [(100, 200), (300, 400), (500, 150)]
    >>> transforms = make_transforms_on_terrain(positions, dem, scale=0.1)
    >>> rtx.add_geometry("towers", tower_verts, tower_indices, transforms=transforms)

    With random rotations:

    >>> import numpy as np
    >>> rotations = np.random.uniform(0, 2*np.pi, len(positions))
    >>> transforms = make_transforms_on_terrain(positions, dem, rotation_z=rotations)
    """
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 2)

    n = len(positions)

    # Handle rotation_z as scalar or array
    if np.isscalar(rotation_z):
        rotations = np.full(n, rotation_z)
    else:
        rotations = np.asarray(rotation_z)
        if len(rotations) != n:
            raise ValueError(f"rotation_z length ({len(rotations)}) must match "
                           f"positions length ({n})")

    transforms = []
    for i, (x, y) in enumerate(positions):
        # Sample terrain elevation at this position
        # Terrain uses row, col indexing: terrain[row, col] = terrain[y, x]
        row = int(np.clip(y, 0, terrain.shape[0] - 1))
        col = int(np.clip(x, 0, terrain.shape[1] - 1))
        z = float(terrain[row, col])

        transforms.append(make_transform(x, y, z, scale, rotations[i]))

    return transforms
