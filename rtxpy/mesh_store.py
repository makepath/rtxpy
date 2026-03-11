"""Zarr-based mesh storage spatially partitioned to match DEM chunks.

Stores mesh data (vertices + indices) inside a zarr store alongside the DEM,
partitioned into spatial chunks that align with the elevation grid chunks.
This makes the zarr a self-contained scene file and enables chunk-level mesh
loading — load only the meshes for the terrain region you're viewing.

Supports both triangle meshes and curve geometries (B-spline tubes used for
roads and water features).

Zarr layout::

    scene.zarr/
      elevation/...               # existing DEM
      meshes/                     # mesh group
        .zattrs → {pixel_spacing, elevation_shape, elevation_chunks}
        building/
          .zattrs → {color: [0.6, 0.6, 0.6, 1.0]}
          0_0/vertices, indices   # chunk (row=0, col=0)
          ...
        road/
          .zattrs → {color: ..., type: "curve"}
          0_0/vertices, widths, indices
          ...
"""

import numpy as np
import zarr
from zarr.codecs import BloscCodec

_BLOSC = BloscCodec(cname='zstd', clevel=6, shuffle='bitshuffle')


def chunks_for_pixel_window(yi0, yi1, xi0, xi1, chunk_h, chunk_w):
    """Map a pixel-coordinate window to overlapping chunk indices.

    Parameters
    ----------
    yi0, yi1 : int
        Row pixel range [yi0, yi1).
    xi0, xi1 : int
        Column pixel range [xi0, xi1).
    chunk_h, chunk_w : int
        Chunk size in pixels (rows, cols).

    Returns
    -------
    list of (int, int)
        List of (chunk_row, chunk_col) tuples that overlap the window.
    """
    cr0 = max(yi0 // chunk_h, 0)
    cr1 = (yi1 - 1) // chunk_h  # inclusive
    cc0 = max(xi0 // chunk_w, 0)
    cc1 = (xi1 - 1) // chunk_w  # inclusive
    return [(cr, cc) for cr in range(cr0, cr1 + 1)
            for cc in range(cc0, cc1 + 1)]


def chunks_for_world_rect(x0, y0, x1, y1, psx, psy, chunk_h, chunk_w,
                          elev_shape):
    """Map a world-coordinate rectangle to overlapping chunk indices.

    Parameters
    ----------
    x0, y0 : float
        Lower-left corner in world coordinates.
    x1, y1 : float
        Upper-right corner in world coordinates.
    psx, psy : float
        Pixel spacing (world units per pixel).
    chunk_h, chunk_w : int
        Chunk size in pixels (rows, cols).
    elev_shape : tuple of int
        ``(H, W)`` of the elevation grid.

    Returns
    -------
    list of (int, int)
        List of ``(chunk_row, chunk_col)`` tuples overlapping the rect.
    """
    H, W = elev_shape
    # World coords -> pixel coords (clamp to grid)
    xi0 = max(int(x0 / psx), 0)
    xi1 = min(int(x1 / psx) + 1, W)
    yi0 = max(int(y0 / psy), 0)
    yi1 = min(int(y1 / psy) + 1, H)
    if xi0 >= xi1 or yi0 >= yi1:
        return []
    return chunks_for_pixel_window(yi0, yi1, xi0, xi1, chunk_h, chunk_w)


def save_meshes_to_zarr(zarr_path, meshes, colors, pixel_spacing,
                        elevation_shape, elevation_chunks,
                        curves=None, spheres=None):
    """Save mesh geometries into a zarr store, spatially partitioned by chunk.

    Parameters
    ----------
    zarr_path : str or Path
        Path to an existing zarr store (opened in ``r+`` mode).
    meshes : dict
        ``{geometry_id: (vertices_flat, indices_flat)}`` where vertices is
        float32 (x, y, z, x, y, z, ...) and indices is int32.
    colors : dict
        ``{geometry_id: (r, g, b) or (r, g, b, a)}``.
    pixel_spacing : tuple of float
        ``(pixel_spacing_x, pixel_spacing_y)`` — world-units per pixel.
    elevation_shape : tuple of int
        ``(H, W)`` of the DEM grid.
    elevation_chunks : tuple of int
        ``(chunk_h, chunk_w)`` of the DEM chunking.
    curves : dict, optional
        ``{geometry_id: (vertices_flat, widths_flat, indices_flat)}`` for
        B-spline curve tube geometries (roads, water features).
    spheres : dict, optional
        ``{geometry_id: (centers_flat, radii, colors_flat)}`` for sphere
        primitive geometries (LiDAR point clouds).
    """
    store = zarr.open(str(zarr_path), mode='r+')

    # Create or overwrite meshes group
    mg = store.create_group('meshes', overwrite=True)
    mg.attrs['pixel_spacing'] = list(pixel_spacing)
    mg.attrs['elevation_shape'] = list(elevation_shape)
    mg.attrs['elevation_chunks'] = list(elevation_chunks)

    psx, psy = pixel_spacing
    chunk_h, chunk_w = elevation_chunks

    for gid, (verts, indices) in meshes.items():
        verts = np.asarray(verts, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)

        # Color attribute
        gg = mg.create_group(gid)
        c = colors.get(gid, (0.6, 0.6, 0.6))
        gg.attrs['color'] = list(c)

        if len(indices) == 0:
            continue

        # Compute triangle centroids in pixel coords
        n_tris = len(indices) // 3
        tri_idx = indices.reshape(n_tris, 3)
        verts_xyz = verts.reshape(-1, 3)

        # Centroid of each triangle
        cx = (verts_xyz[tri_idx[:, 0], 0] +
              verts_xyz[tri_idx[:, 1], 0] +
              verts_xyz[tri_idx[:, 2], 0]) / 3.0
        cy = (verts_xyz[tri_idx[:, 0], 1] +
              verts_xyz[tri_idx[:, 1], 1] +
              verts_xyz[tri_idx[:, 2], 1]) / 3.0

        # Convert world coords → pixel → chunk indices
        px = cx / psx
        py = cy / psy
        chunk_rows = np.clip(py.astype(np.int64) // chunk_h, 0,
                             (elevation_shape[0] - 1) // chunk_h)
        chunk_cols = np.clip(px.astype(np.int64) // chunk_w, 0,
                             (elevation_shape[1] - 1) // chunk_w)

        # Unique chunks
        chunk_keys = chunk_rows * 100000 + chunk_cols  # perfect hash for sane sizes
        unique_keys = np.unique(chunk_keys)

        for uk in unique_keys:
            cr = int(uk // 100000)
            cc = int(uk % 100000)
            mask = chunk_keys == uk

            # Extract triangles for this chunk
            local_tri_idx = tri_idx[mask]  # (n_local, 3)

            # Remap vertex indices — only keep referenced vertices
            used_verts = np.unique(local_tri_idx.ravel())
            remap = np.empty(len(verts_xyz), dtype=np.int32)
            remap[used_verts] = np.arange(len(used_verts), dtype=np.int32)
            local_verts = verts_xyz[used_verts].ravel().astype(np.float32)
            local_indices = remap[local_tri_idx.ravel()].astype(np.int32)

            chunk_grp = gg.create_group(f'{cr}_{cc}')
            chunk_grp.create_array(
                'vertices', data=local_verts,
                chunks=(len(local_verts),), compressors=_BLOSC,
            )
            chunk_grp.create_array(
                'indices', data=local_indices,
                chunks=(len(local_indices),), compressors=_BLOSC,
            )

    # --- Curve geometries (roads, water) ---
    if curves:
        for gid, (verts, widths, indices) in curves.items():
            verts = np.asarray(verts, dtype=np.float32)
            widths = np.asarray(widths, dtype=np.float32)
            indices = np.asarray(indices, dtype=np.int32)

            gg = mg.create_group(gid)
            c = colors.get(gid, (0.6, 0.6, 0.6))
            gg.attrs['color'] = list(c)
            gg.attrs['type'] = 'curve'

            if len(indices) == 0:
                continue

            # Partition curve segments by control point centroid
            verts_xyz = verts.reshape(-1, 3)
            # Each segment starts at indices[i], uses 3 consecutive control
            # points (quadratic B-spline).  Use the middle control point as
            # the spatial key.
            mid_idx = np.clip(indices + 1, 0, len(verts_xyz) - 1)
            cx = verts_xyz[mid_idx, 0]
            cy = verts_xyz[mid_idx, 1]

            px = cx / psx
            py = cy / psy
            chunk_rows = np.clip(py.astype(np.int64) // chunk_h, 0,
                                 (elevation_shape[0] - 1) // chunk_h)
            chunk_cols = np.clip(px.astype(np.int64) // chunk_w, 0,
                                 (elevation_shape[1] - 1) // chunk_w)

            chunk_keys = chunk_rows * 100000 + chunk_cols
            unique_keys = np.unique(chunk_keys)

            for uk in unique_keys:
                cr = int(uk // 100000)
                cc = int(uk % 100000)
                mask = chunk_keys == uk

                local_seg_idx = indices[mask]
                # Each segment uses 3 control points: [i, i+1, i+2]
                used = set()
                for si in local_seg_idx:
                    used.update(range(si, min(si + 3, len(verts_xyz))))
                used_verts = np.array(sorted(used), dtype=np.int32)
                remap = np.empty(len(verts_xyz), dtype=np.int32)
                remap[used_verts] = np.arange(len(used_verts), dtype=np.int32)

                local_verts = verts_xyz[used_verts].ravel().astype(np.float32)
                local_widths = widths[used_verts].astype(np.float32)
                local_indices = remap[local_seg_idx].astype(np.int32)

                chunk_grp = gg.create_group(f'{cr}_{cc}')
                chunk_grp.create_array(
                    'vertices', data=local_verts,
                    chunks=(len(local_verts),), compressors=_BLOSC,
                )
                chunk_grp.create_array(
                    'widths', data=local_widths,
                    chunks=(len(local_widths),), compressors=_BLOSC,
                )
                chunk_grp.create_array(
                    'indices', data=local_indices,
                    chunks=(len(local_indices),), compressors=_BLOSC,
                )

    # --- Sphere geometries (point clouds) ---
    if spheres:
        for gid, (centers, radii, sph_colors) in spheres.items():
            centers = np.asarray(centers, dtype=np.float32)
            radii = np.asarray(radii, dtype=np.float32)
            sph_colors = np.asarray(sph_colors, dtype=np.float32)

            gg = mg.create_group(gid)
            c = colors.get(gid, (0.5, 0.5, 0.5, 5.0))
            gg.attrs['color'] = list(c)
            gg.attrs['type'] = 'sphere'

            if len(centers) == 0:
                continue

            # Single chunk per geometry (already spatially partitioned by tile)
            chunk_grp = gg.create_group('0_0')
            chunk_grp.create_array(
                'centers', data=centers,
                chunks=(len(centers),), compressors=_BLOSC,
            )
            chunk_grp.create_array(
                'radii', data=radii,
                chunks=(len(radii),), compressors=_BLOSC,
            )
            chunk_grp.create_array(
                'colors', data=sph_colors,
                chunks=(len(sph_colors),), compressors=_BLOSC,
            )

    total = len(meshes) + (len(curves) if curves else 0) + \
        (len(spheres) if spheres else 0)
    print(f"Saved {total} mesh geometries to {zarr_path}/meshes/")


def load_meshes_from_zarr(zarr_path, chunks=None):
    """Load mesh geometries from a zarr store.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr store.
    chunks : list of (int, int) or None
        Specific chunk indices ``[(row, col), ...]`` to load. ``None`` loads
        all chunks (full scene).

    Returns
    -------
    meshes : dict
        ``{geometry_id: (vertices_flat, indices_flat)}`` for triangle meshes.
    colors : dict
        ``{geometry_id: tuple}``.
    meta : dict
        ``{pixel_spacing, elevation_shape, elevation_chunks}`` from attrs.
    curves : dict
        ``{geometry_id: (vertices_flat, widths_flat, indices_flat)}`` for
        curve geometries.  Empty dict if none stored.
    spheres : dict
        ``{geometry_id: (centers_flat, radii, colors_flat)}`` for sphere
        primitive geometries.  Empty dict if none stored.
    """
    store = zarr.open(str(zarr_path), mode='r', use_consolidated=False)
    mg = store['meshes']

    meta = {
        'pixel_spacing': tuple(mg.attrs['pixel_spacing']),
        'elevation_shape': tuple(mg.attrs['elevation_shape']),
        'elevation_chunks': tuple(mg.attrs['elevation_chunks']),
    }

    # Build set of allowed chunk keys for filtering
    chunk_set = None
    if chunks is not None:
        chunk_set = {f'{cr}_{cc}' for cr, cc in chunks}

    meshes = {}
    curves = {}
    spheres = {}
    colors = {}

    for gid in mg:
        gg = mg[gid]
        if not hasattr(gg, 'attrs'):
            continue
        colors[gid] = tuple(gg.attrs.get('color', (0.6, 0.6, 0.6)))
        geom_type = gg.attrs.get('type', '')
        is_curve = geom_type == 'curve'
        is_sphere = geom_type == 'sphere'

        if is_sphere:
            # Sphere geometry: concatenate centers/radii/colors across chunks
            all_centers = []
            all_radii = []
            all_colors = []
            for key in sorted(gg):
                if key in ('centers', 'radii', 'colors'):
                    continue
                if chunk_set is not None and key not in chunk_set:
                    continue
                cg = gg[key]
                if 'centers' in cg:
                    all_centers.append(
                        np.array(cg['centers'], dtype=np.float32))
                if 'radii' in cg:
                    all_radii.append(
                        np.array(cg['radii'], dtype=np.float32))
                if 'colors' in cg:
                    all_colors.append(
                        np.array(cg['colors'], dtype=np.float32))

            if all_centers:
                spheres[gid] = (
                    np.concatenate(all_centers),
                    np.concatenate(all_radii),
                    np.concatenate(all_colors) if all_colors else
                    np.empty(0, dtype=np.float32),
                )
            else:
                spheres[gid] = (
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                )
            continue

        all_verts = []
        all_widths = []
        all_indices = []
        vert_offset = 0

        # Iterate sub-groups (chunk keys like "0_0", "0_1", ...)
        for key in sorted(gg):
            if key in ('vertices', 'indices', 'widths'):
                continue  # skip if somehow present at geometry level
            if chunk_set is not None and key not in chunk_set:
                continue

            cg = gg[key]
            v = np.array(cg['vertices'], dtype=np.float32)
            idx = np.array(cg['indices'], dtype=np.int32)

            # Offset indices by accumulated vertex count
            idx = idx + vert_offset
            all_verts.append(v)
            all_indices.append(idx)
            vert_offset += len(v) // 3

            if is_curve and 'widths' in cg:
                all_widths.append(np.array(cg['widths'], dtype=np.float32))

        if all_verts:
            cat_verts = np.concatenate(all_verts)
            cat_indices = np.concatenate(all_indices)
            if is_curve and all_widths:
                curves[gid] = (cat_verts,
                               np.concatenate(all_widths),
                               cat_indices)
            else:
                meshes[gid] = (cat_verts, cat_indices)
        else:
            if is_curve:
                curves[gid] = (np.empty(0, dtype=np.float32),
                               np.empty(0, dtype=np.float32),
                               np.empty(0, dtype=np.int32))
            else:
                meshes[gid] = (np.empty(0, dtype=np.float32),
                               np.empty(0, dtype=np.int32))

    return meshes, colors, meta, curves, spheres


# ---------------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------------

SCENE_VERSION = "1.0"


def validate_scene(zarr_path):
    """Validate a zarr store against the rtxpy scene specification.

    Returns a list of ``(level, message)`` tuples where *level* is
    ``"error"`` for spec violations that would prevent loading, or
    ``"warning"`` for non-conforming optional data.

    An empty list means the scene is valid.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr store.
    """
    issues = []
    try:
        store = zarr.open(str(zarr_path), mode='r', use_consolidated=False)
    except Exception as exc:
        return [("error", f"Cannot open zarr store: {exc}")]

    def _err(msg):
        issues.append(("error", msg))

    def _warn(msg):
        issues.append(("warning", msg))

    # -- Root attributes --
    if 'rtxpy_scene_version' not in store.attrs:
        _warn("Missing root attribute 'rtxpy_scene_version'")

    # -- elevation --
    if 'elevation' not in store:
        _err("Missing required array 'elevation'")
    else:
        elev = store['elevation']
        for attr in ('scale_factor', 'add_offset', '_FillValue'):
            if attr not in elev.attrs:
                _warn(f"elevation: missing CF attribute '{attr}'")

    # -- spatial_ref --
    if 'spatial_ref' not in store:
        _err("Missing required variable 'spatial_ref'")
    else:
        sr = store['spatial_ref']
        for attr in ('crs_wkt', 'GeoTransform'):
            if attr not in sr.attrs:
                _warn(f"spatial_ref: missing attribute '{attr}'")

    # -- elevation_roughness (optional) --
    if 'elevation_roughness' in store:
        er = store['elevation_roughness']
        if 'tile_size' not in er.attrs:
            _warn("elevation_roughness: missing 'tile_size' attribute")

    # -- mesh LOD arrays (checked within meshes below) --

    # -- meshes (optional) --
    if 'meshes' in store:
        mg = store['meshes']
        for attr in ('pixel_spacing', 'elevation_shape', 'elevation_chunks'):
            if attr not in mg.attrs:
                _warn(f"meshes: missing attribute '{attr}'")
        for gid in mg:
            gg = mg[gid]
            if not hasattr(gg, 'attrs'):
                continue
            if 'color' not in gg.attrs:
                _warn(f"meshes/{gid}: missing 'color' attribute")
            geom_type = gg.attrs.get('type', '')
            for key in gg:
                if key in ('vertices', 'indices', 'widths',
                           'centers', 'radii', 'colors'):
                    continue
                cg = gg[key]
                if not hasattr(cg, 'keys'):
                    continue
                if geom_type == 'sphere':
                    if 'centers' not in cg:
                        _warn(f"meshes/{gid}/{key}: missing 'centers'")
                    if 'radii' not in cg:
                        _warn(f"meshes/{gid}/{key}: missing 'radii'")
                    # Point cloud attribute length consistency
                    if 'centers' in cg:
                        n_pts = cg['centers'].shape[0] // 3
                        for pc_attr in ('classification', 'intensity',
                                        'return_number', 'number_of_returns'):
                            if pc_attr in cg:
                                if cg[pc_attr].shape[0] != n_pts:
                                    _warn(
                                        f"meshes/{gid}/{key}: "
                                        f"'{pc_attr}' length "
                                        f"{cg[pc_attr].shape[0]} != "
                                        f"point count {n_pts}"
                                    )
                else:
                    if 'vertices' not in cg:
                        _warn(f"meshes/{gid}/{key}: missing 'vertices'")
                    if 'indices' not in cg:
                        _warn(f"meshes/{gid}/{key}: missing 'indices'")
                    if geom_type == 'curve' and 'widths' not in cg:
                        _warn(f"meshes/{gid}/{key}: missing 'widths'")
                    # Check mesh LOD consistency — if any LOD level
                    # exists, both vertices and indices must be present
                    if geom_type not in ('curve', 'sphere'):
                        for lod_n in range(1, 10):
                            vk = f'vertices_lod{lod_n}'
                            ik = f'indices_lod{lod_n}'
                            has_v = vk in cg
                            has_i = ik in cg
                            if has_v != has_i:
                                missing = ik if has_v else vk
                                _warn(f"meshes/{gid}/{key}: has "
                                      f"partial LOD {lod_n} "
                                      f"(missing '{missing}')")
                            if not has_v and not has_i:
                                break  # no more LOD levels

    # -- overlays (optional) --
    if 'overlays' in store:
        elev_shape = None
        if 'elevation' in store:
            elev_shape = store['elevation'].shape
        for name in store['overlays']:
            og = store['overlays'][name]
            if 'data' not in og:
                _warn(f"overlays/{name}: missing 'data' array")
            elif elev_shape is not None:
                if og['data'].shape != elev_shape:
                    _warn(f"overlays/{name}: shape {og['data'].shape} "
                          f"doesn't match elevation {elev_shape}")

    # -- wind (optional) --
    if 'wind' in store:
        wg = store['wind']
        for arr in ('u', 'v'):
            if arr not in wg:
                _warn(f"wind: missing array '{arr}'")
        if 'grid_bounds' not in wg.attrs:
            _warn("wind: missing 'grid_bounds' attribute")

    # -- hydro (optional) --
    if 'hydro' in store:
        hg = store['hydro']
        for arr in ('flow_accum', 'flow_dir_mfd'):
            if arr not in hg:
                _warn(f"hydro: missing array '{arr}'")

    # -- tour (optional) --
    if 'tour' in store:
        tg = store['tour']
        required_tour = ('time', 'position', 'yaw', 'pitch')
        lengths = {}
        for arr in required_tour:
            if arr not in tg:
                _warn(f"tour: missing array '{arr}'")
            else:
                lengths[arr] = tg[arr].shape[0]
        if len(set(lengths.values())) > 1:
            _warn(f"tour: arrays have mismatched lengths: {lengths}")

    return issues
