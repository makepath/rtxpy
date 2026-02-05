"""Mesh utilities for terrain triangulation, mesh loading, and STL export.

This module provides functions for converting raster terrain data into
triangle meshes suitable for ray tracing, loading 3D model files (GLB, OBJ, etc.),
and exporting meshes to STL format.
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
            # Counter-clockwise winding for upward-facing normals (+Z)
            triangles[offset + 0] = np.int32(meshMapIndex)
            triangles[offset + 1] = np.int32(meshMapIndex + W + 1)
            triangles[offset + 2] = np.int32(meshMapIndex + W)
            triangles[offset + 3] = np.int32(meshMapIndex)
            triangles[offset + 4] = np.int32(meshMapIndex + 1)
            triangles[offset + 5] = np.int32(meshMapIndex + W + 1)


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
                # Counter-clockwise winding for upward-facing normals (+Z)
                triangles[offset + 0] = np.int32(meshMapIndex)
                triangles[offset + 1] = np.int32(meshMapIndex + W + 1)
                triangles[offset + 2] = np.int32(meshMapIndex + W)
                triangles[offset + 3] = np.int32(meshMapIndex)
                triangles[offset + 4] = np.int32(meshMapIndex + 1)
                triangles[offset + 5] = np.int32(meshMapIndex + W + 1)


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


def _lazy_import_trimesh():
    """Lazily import trimesh with helpful error message."""
    try:
        import trimesh
        return trimesh
    except ImportError:
        raise ImportError(
            "trimesh is required for loading GLB/glTF files. "
            "Install it with: pip install trimesh"
        )


def load_glb(filepath, scale=1.0, swap_yz=False, center_xy=False, base_at_zero=False):
    """Load a GLB/glTF mesh file and return vertices and indices for ray tracing.

    Uses trimesh to load the mesh and extracts geometry suitable for use with
    RTX.add_geometry().

    Parameters
    ----------
    filepath : str or Path
        Path to the GLB or glTF file.
    scale : float, optional
        Scale factor applied to all vertex positions. Default is 1.0.
    swap_yz : bool, optional
        If True, swap Y and Z coordinates. Useful when models use Y-up
        convention but rtxpy expects Z-up. Default is False.
    center_xy : bool, optional
        If True, center the model at the XY origin. Default is False.
    base_at_zero : bool, optional
        If True, translate the model so its lowest Z coordinate is at 0.
        Default is False.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 array of vertex positions with shape (num_verts * 3,).
        Contains x, y, z coordinates for each vertex.
    indices : np.ndarray
        Flat int32 array of triangle indices with shape (num_tris * 3,).
        Each group of 3 indices defines a triangle.

    Examples
    --------
    >>> verts, indices = load_glb('model.glb', scale=0.1, swap_yz=True)
    >>> rtx.add_geometry('model', verts, indices)

    >>> # With instancing
    >>> verts, indices = load_glb('tower.glb', swap_yz=True, base_at_zero=True)
    >>> transforms = make_transforms_on_terrain(positions, terrain)
    >>> rtx.add_geometry('towers', verts, indices, transforms=transforms)

    Notes
    -----
    GLB files may contain multiple meshes in a scene. This function combines
    all meshes into a single vertex/index buffer. Materials and textures are
    ignored - only geometry is extracted.
    """
    trimesh = _lazy_import_trimesh()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load the scene/mesh
    scene_or_mesh = trimesh.load(filepath)

    # Handle scenes with multiple meshes
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Combine all meshes in the scene
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        if not meshes:
            raise ValueError(f"No triangle meshes found in: {filepath}")

        # Concatenate all meshes
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not extract triangle mesh from: {filepath}")

    # Get vertices and faces
    vertices = mesh.vertices.copy().astype(np.float32)
    faces = mesh.faces.copy().astype(np.int32)

    # Apply coordinate transforms
    if swap_yz:
        # Swap Y and Z (convert Y-up to Z-up)
        vertices[:, [1, 2]] = vertices[:, [2, 1]]

    if center_xy:
        # Center at XY origin
        center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
        vertices[:, 0] -= center[0]
        vertices[:, 1] -= center[1]

    if base_at_zero:
        # Move base to Z=0
        vertices[:, 2] -= vertices[:, 2].min()

    # Apply scale
    if scale != 1.0:
        vertices *= scale

    # Flatten arrays for rtxpy format
    vertices_flat = vertices.flatten().astype(np.float32)
    indices_flat = faces.flatten().astype(np.int32)

    return vertices_flat, indices_flat


def load_mesh(filepath, scale=1.0, swap_yz=False, center_xy=False, base_at_zero=False):
    """Load a mesh file in any supported format (GLB, glTF, OBJ, STL, PLY, etc.).

    This is a convenience wrapper that uses trimesh to load meshes in various
    formats. For GLB/glTF files specifically, you can also use load_glb().

    Parameters
    ----------
    filepath : str or Path
        Path to the mesh file. Supported formats include:
        GLB, glTF, OBJ, STL, PLY, OFF, and others supported by trimesh.
    scale : float, optional
        Scale factor applied to all vertex positions. Default is 1.0.
    swap_yz : bool, optional
        If True, swap Y and Z coordinates. Useful when models use Y-up
        convention but rtxpy expects Z-up. Default is False.
    center_xy : bool, optional
        If True, center the model at the XY origin. Default is False.
    base_at_zero : bool, optional
        If True, translate the model so its lowest Z coordinate is at 0.
        Default is False.

    Returns
    -------
    vertices : np.ndarray
        Flat float32 array of vertex positions with shape (num_verts * 3,).
    indices : np.ndarray
        Flat int32 array of triangle indices with shape (num_tris * 3,).

    See Also
    --------
    load_glb : Specifically for GLB/glTF files.
    """
    return load_glb(filepath, scale=scale, swap_yz=swap_yz,
                    center_xy=center_xy, base_at_zero=base_at_zero)


def make_transform(x=0.0, y=0.0, z=0.0, scale=1.0, rotation_z=0.0):
    """Create a 3x4 transform matrix for positioning an instance.

    The transform matrix is stored as a flat list of 12 floats in row-major order,
    representing a 3x4 matrix [R|T] where R is the 3x3 rotation/scale matrix
    and T is the translation vector.

    Parameters
    ----------
    x : float, optional
        X position. Default is 0.
    y : float, optional
        Y position. Default is 0.
    z : float, optional
        Z position (elevation). Default is 0.
    scale : float, optional
        Uniform scale factor. Default is 1.0.
    rotation_z : float, optional
        Rotation around the Z axis in radians. Default is 0.

    Returns
    -------
    list of float
        12-element list representing the 3x4 transform matrix in row-major order:
        [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]

    Examples
    --------
    >>> transform = make_transform(x=100, y=50, z=200)
    >>> rtx.add_geometry("obj", verts, indices, transforms=[transform])

    >>> # With rotation and scale
    >>> transform = make_transform(x=10, y=20, z=5, scale=2.0, rotation_z=np.pi/4)
    """
    cos_z = np.cos(rotation_z)
    sin_z = np.sin(rotation_z)

    # Rotation matrix around Z axis, with scale
    # [cos -sin 0]   [s 0 0]   [s*cos  -s*sin  0]
    # [sin  cos 0] * [0 s 0] = [s*sin   s*cos  0]
    # [0    0   1]   [0 0 s]   [0       0      s]
    r00 = scale * cos_z
    r01 = -scale * sin_z
    r02 = 0.0
    r10 = scale * sin_z
    r11 = scale * cos_z
    r12 = 0.0
    r20 = 0.0
    r21 = 0.0
    r22 = scale

    return [r00, r01, r02, float(x),
            r10, r11, r12, float(y),
            r20, r21, r22, float(z)]


def make_transforms_on_terrain(positions, terrain, scale=1.0, rotation_z=0.0):
    """Create transform matrices for placing objects on terrain.

    For each (x, y) position, samples the terrain elevation and creates
    a transform matrix that places an object at that location on the terrain surface.

    Parameters
    ----------
    positions : list of tuple
        List of (x, y) positions in terrain pixel coordinates.
    terrain : array-like
        2D terrain elevation array with shape (H, W).
    scale : float, optional
        Uniform scale factor for all instances. Default is 1.0.
    rotation_z : float or str, optional
        Rotation around Z axis. Can be:
        - A float: same rotation (in radians) for all instances
        - 'random': random rotation for each instance
        Default is 0.0.

    Returns
    -------
    list of list
        List of 12-element transform matrices, one per position.

    Examples
    --------
    >>> positions = [(10, 20), (30, 40), (50, 60)]
    >>> transforms = make_transforms_on_terrain(positions, terrain)
    >>> rtx.add_geometry("trees", tree_verts, tree_indices, transforms=transforms)

    >>> # With random rotations
    >>> transforms = make_transforms_on_terrain(positions, terrain, rotation_z='random')
    """
    # Handle xarray DataArray
    if hasattr(terrain, 'data'):
        terrain_data = terrain.data
        if has_cupy:
            import cupy
            if isinstance(terrain_data, cupy.ndarray):
                terrain_data = cupy.asnumpy(terrain_data)
    else:
        terrain_data = terrain

    if has_cupy:
        import cupy
        if isinstance(terrain_data, cupy.ndarray):
            terrain_data = cupy.asnumpy(terrain_data)

    H, W = terrain_data.shape
    transforms = []

    for i, (x, y) in enumerate(positions):
        # Sample terrain elevation at this position
        # Clamp to valid range
        ix = int(np.clip(x, 0, W - 1))
        iy = int(np.clip(y, 0, H - 1))
        z = float(terrain_data[iy, ix])

        # Determine rotation
        if rotation_z == 'random':
            rot = np.random.uniform(0, 2 * np.pi)
        else:
            rot = float(rotation_z)

        transform = make_transform(x=x, y=y, z=z, scale=scale, rotation_z=rot)
        transforms.append(transform)

    return transforms
