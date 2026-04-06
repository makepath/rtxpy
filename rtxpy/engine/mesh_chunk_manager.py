"""Dynamically loads/unloads mesh chunks based on camera position."""

import numpy as np

from ..rtx import has_cupy
from .helpers import _bilinear_terrain_z

if has_cupy:
    import cupy as cp


class _MeshChunkManager:
    """Dynamically loads/unloads mesh chunks based on camera position.

    Manages chunk lifecycle: reads per-chunk mesh data from a zarr store,
    caches it in memory, and merges visible chunks per geometry ID into
    the RTX scene.  Only nearby chunks (within ``radius`` of the camera)
    are kept in the scene; the rest are removed.
    """

    def __init__(self, zarr_path, psx, psy):
        import zarr as _zarr
        store = _zarr.open(str(zarr_path), mode='r', use_consolidated=False)
        mg = store['meshes']

        self._elev_shape = tuple(mg.attrs['elevation_shape'])
        self._elev_chunks = tuple(mg.attrs['elevation_chunks'])
        self._chunk_h, self._chunk_w = self._elev_chunks
        self._psx = psx
        self._psy = psy
        self._n_chunk_rows = (self._elev_shape[0] + self._chunk_h - 1) // self._chunk_h
        self._n_chunk_cols = (self._elev_shape[1] + self._chunk_w - 1) // self._chunk_w

        # Per-gid colors from zarr attrs
        self._colors = {}
        self._gids = []
        for gid in mg:
            gg = mg[gid]
            if hasattr(gg, 'attrs'):
                self._colors[gid] = tuple(gg.attrs.get('color', (0.6, 0.6, 0.6)))
                self._gids.append(gid)

        # Cache: (cr, cc) -> {gid: (verts, indices)} or None if empty
        self._cache = {}
        self._visible = set()
        self._active_gids = set()  # gids currently in the RTX scene
        self.radius = 2
        self._zarr_path = zarr_path

        # Distance-aware loading parameters
        self._chunk_world_w = self._chunk_w * psx
        self._chunk_world_h = self._chunk_h * psy
        self.max_distance = None  # None = use radius-based fallback
        self.per_tick_load_limit = 2  # max new zarr reads per tick
        self.max_chunks = 25  # max visible chunks

        # LOD-aware loading state
        self._lod_distances = None  # set from LOD manager
        self._tile_lods = None  # {(tr,tc): lod_level} from aligned LOD manager
        self._last_cam_pos = None  # for movement detection
        self._cam_moving = False

        # Mesh simplification for placed geometry at higher LOD levels.
        # LOD 0 = full detail (ratio 1.0), LOD 1 = 50%, LOD 2 = 25%, LOD 3+ = 10%.
        self._simplify_ratios = (1.0, 0.5, 0.25, 0.1)
        # Cache: (cr, cc, gid, lod) -> (simplified_verts, simplified_indices)
        self._simplify_cache = {}

    def _load_chunk(self, cr, cc):
        """Load a single chunk from zarr into cache."""
        if (cr, cc) in self._cache:
            return
        from ..mesh_store import load_meshes_from_zarr
        meshes, _, _, curves, _spheres = load_meshes_from_zarr(
            self._zarr_path, chunks=[(cr, cc)])
        # Merge curves into the same dict with a marker
        combined = {}
        for gid, data in meshes.items():
            combined[gid] = data  # (verts, indices)
        for gid, data in curves.items():
            combined[gid] = data  # (verts, widths, indices)
        self._cache[(cr, cc)] = combined

    def _chunk_center(self, cr, cc):
        """World-coordinate center of chunk (cr, cc)."""
        cx = (cc * self._chunk_w + self._chunk_w * 0.5) * self._psx
        cy = (cr * self._chunk_h + self._chunk_h * 0.5) * self._psy
        return cx, cy

    def _get_simplified(self, cr, cc, gid, lod, verts, indices):
        """Return (possibly simplified) mesh for a chunk at a given LOD.

        LOD 0 returns the original mesh.  LOD 1+ applies quadric
        decimation with ratio from ``_simplify_ratios``, caching the
        result for reuse across frames.
        """
        ratio_idx = min(lod, len(self._simplify_ratios) - 1)
        ratio = self._simplify_ratios[ratio_idx]
        if ratio >= 1.0:
            return verts, indices
        key = (cr, cc, gid, lod)
        cached = self._simplify_cache.get(key)
        if cached is not None:
            return cached
        from ..lod import simplify_mesh
        sv, si = simplify_mesh(verts, indices, ratio)
        self._simplify_cache[key] = (sv, si)
        return sv, si

    def update(self, cam_x, cam_y, viewer):
        """Called per tick. Returns True if meshes changed."""
        import math
        from ..lod import compute_lod_level

        # Detect camera movement for LOD-aware load deferral
        move_thresh = self._chunk_world_w * 0.1
        if self._last_cam_pos is not None:
            dx = cam_x - self._last_cam_pos[0]
            dy = cam_y - self._last_cam_pos[1]
            self._cam_moving = (dx * dx + dy * dy) > move_thresh * move_thresh
        self._last_cam_pos = (cam_x, cam_y)

        max_dist = self.max_distance
        lod_dists = self._lod_distances
        tile_lods = self._tile_lods  # {(tr,tc): lod} when grids aligned
        chunk_dists = {}
        chunk_lods = {}  # per-chunk LOD level

        if tile_lods is not None:
            # Grids aligned: reuse LOD manager's tile assignments directly.
            # tile_lods keys are (tile_row, tile_col) which map 1:1 to
            # chunk (cr, cc) when tile_size == chunk_size.
            new_visible = set()
            max_lod = len(lod_dists) if lod_dists else 999
            for (cr, cc), lod in tile_lods.items():
                if cr >= self._n_chunk_rows or cc >= self._n_chunk_cols:
                    continue
                if lod > max_lod:
                    continue
                chunk_lods[(cr, cc)] = lod
                cx, cy = self._chunk_center(cr, cc)
                chunk_dists[(cr, cc)] = math.sqrt(
                    (cam_x - cx) ** 2 + (cam_y - cy) ** 2)
                new_visible.add((cr, cc))
            # Cap at max_chunks, keeping closest
            if len(new_visible) > self.max_chunks:
                by_dist = sorted(new_visible, key=lambda c: chunk_dists[c])
                new_visible = set(by_dist[:self.max_chunks])
        elif max_dist is not None:
            # Distance-aware: compute visible chunks from world-rect
            from ..mesh_store import chunks_for_world_rect
            x0 = cam_x - max_dist
            y0 = cam_y - max_dist
            x1 = cam_x + max_dist
            y1 = cam_y + max_dist
            candidates = chunks_for_world_rect(
                x0, y0, x1, y1,
                self._psx, self._psy,
                self._chunk_h, self._chunk_w,
                self._elev_shape)
            for cr, cc in candidates:
                cx, cy = self._chunk_center(cr, cc)
                chunk_dists[(cr, cc)] = math.sqrt(
                    (cam_x - cx) ** 2 + (cam_y - cy) ** 2)
            if lod_dists:
                max_lod = len(lod_dists)
                candidates = [
                    c for c in candidates
                    if compute_lod_level(chunk_dists[c], lod_dists) <= max_lod
                ]
            candidates.sort(key=lambda c: chunk_dists[c])
            new_visible = set(candidates[:self.max_chunks])
            # Compute per-chunk LOD for mesh simplification
            if lod_dists:
                for cr, cc in new_visible:
                    chunk_lods[(cr, cc)] = compute_lod_level(
                        chunk_dists[(cr, cc)], lod_dists)
        else:
            # Legacy radius-based ring
            cc_cam = int(cam_x / self._psx) // self._chunk_w
            cr_cam = int(cam_y / self._psy) // self._chunk_h
            cr0 = max(cr_cam - self.radius, 0)
            cr1 = min(cr_cam + self.radius, self._n_chunk_rows - 1)
            cc0 = max(cc_cam - self.radius, 0)
            cc1 = min(cc_cam + self.radius, self._n_chunk_cols - 1)
            new_visible = set()
            for cr in range(cr0, cr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    new_visible.add((cr, cc))
                    cx, cy = self._chunk_center(cr, cc)
                    chunk_dists[(cr, cc)] = math.sqrt(
                        (cam_x - cx) ** 2 + (cam_y - cy) ** 2)

        # Check if any visible chunks are uncached (deferred from prior tick)
        has_deferred = any((cr, cc) not in self._cache for cr, cc in new_visible)

        if new_visible == self._visible and not has_deferred:
            return False

        # Evict simplification cache entries for chunks leaving visible set
        departed = self._visible - new_visible
        if departed and self._simplify_cache:
            for k in [k for k in self._simplify_cache
                      if (k[0], k[1]) in departed]:
                del self._simplify_cache[k]

        self._visible = new_visible

        # Load uncached chunks, prioritized by distance (closest first).
        # Limited to per_tick_load_limit new zarr reads per tick.
        uncached = [(cr, cc) for cr, cc in new_visible
                    if (cr, cc) not in self._cache]
        if uncached and chunk_dists:
            uncached.sort(key=lambda c: chunk_dists.get(c, 0))
        loads = 0
        for cr, cc in uncached:
            if loads >= self.per_tick_load_limit:
                break
            # When moving, defer distant (LOD 1+) chunks
            lod = chunk_lods.get((cr, cc))
            if lod is None and lod_dists and (cr, cc) in chunk_dists:
                lod = compute_lod_level(chunk_dists[(cr, cc)], lod_dists)
            if self._cam_moving and lod is not None and lod > 0:
                continue
            self._load_chunk(cr, cc)
            loads += 1

        # Merge visible chunks per gid.  Iterate chunks first so we only
        # touch gids that actually have data (skips empty lookups).
        # Per-gid accumulators: {gid: (all_verts, all_widths, all_indices,
        #                              vert_offset, is_curve)}
        merge_acc = {}
        for cr, cc in sorted(new_visible):
            chunk_data = self._cache.get((cr, cc))
            if not chunk_data:
                continue
            clod = chunk_lods.get((cr, cc), 0)
            for gid, data in chunk_data.items():
                if len(data) == 3:
                    verts, widths, indices = data
                    if len(indices) == 0:
                        continue
                    acc = merge_acc.get(gid)
                    if acc is None:
                        acc = ([], [], [], [0], True)
                        merge_acc[gid] = acc
                    acc[0].append(verts)
                    acc[1].append(widths)
                    acc[2].append(indices + acc[3][0])
                    acc[3][0] += len(verts) // 3
                else:
                    verts, indices = data
                    if len(indices) == 0:
                        continue
                    if clod > 0:
                        verts, indices = self._get_simplified(
                            cr, cc, gid, clod, verts, indices)
                    acc = merge_acc.get(gid)
                    if acc is None:
                        acc = ([], [], [], [0], False)
                        merge_acc[gid] = acc
                    acc[0].append(verts)
                    acc[2].append(indices + acc[3][0])
                    acc[3][0] += len(verts) // 3

        merged = {}
        for gid, (all_verts, all_widths, all_indices, _, is_curve) in merge_acc.items():
            if all_verts:
                if is_curve:
                    merged[gid] = (np.concatenate(all_verts),
                                   np.concatenate(all_widths),
                                   np.concatenate(all_indices))
                else:
                    merged[gid] = (np.concatenate(all_verts),
                                   np.concatenate(all_indices))

        # Remove gids no longer present
        rtx = viewer.rtx
        accessor = viewer._accessor
        for gid in list(self._active_gids):
            if gid not in merged:
                rtx.remove_geometry(gid)
                if accessor is not None:
                    accessor._baked_meshes.pop(gid, None)
                    accessor._geometry_colors.pop(gid, None)
                self._active_gids.discard(gid)

        # Get current (possibly subsampled) terrain data
        terrain_np = viewer.raster.data
        if hasattr(terrain_np, 'get'):
            terrain_np = terrain_np.get()
        else:
            terrain_np = np.asarray(terrain_np)
        H, W = terrain_np.shape
        ve = viewer.vertical_exaggeration

        # Get full-res terrain for computing original base_z
        base_terrain = viewer._base_raster.data
        if hasattr(base_terrain, 'get'):
            base_terrain_np = base_terrain.get()
        else:
            base_terrain_np = np.asarray(base_terrain)
        base_psx = viewer._base_pixel_spacing_x
        base_psy = viewer._base_pixel_spacing_y

        # Upload terrain to GPU once (use cached if available)
        gpu_terrain = None
        gpu_base_terrain = None
        if has_cupy:
            if viewer._gpu_terrain is None:
                viewer._gpu_terrain = cp.asarray(terrain_np)
            gpu_terrain = viewer._gpu_terrain
            if viewer._gpu_base_terrain is None:
                viewer._gpu_base_terrain = cp.asarray(base_terrain_np)
            gpu_base_terrain = viewer._gpu_base_terrain

        # Add/update merged gids
        for gid, data in merged.items():
            is_curve = len(data) == 3
            if is_curve:
                verts, widths, indices = data
            else:
                verts, indices = data

            # Apply VE to Z coordinates and cache base_z for VE rescaling.
            # orig_base_z and new_base_z both sample the same full-res terrain
            # at the same XY positions, so (new_base_z + z_offset) == stored_z.
            # The only transformation needed is: final_z = stored_z * ve.
            # We still compute base_z once for the baked mesh cache (used by
            # _rebuild_vertical_exaggeration to rescale without re-reading zarr).
            n_verts = len(verts) // 3
            use_gpu = (gpu_terrain is not None
                       and gpu_base_terrain is not None
                       and n_verts > 10000)

            if use_gpu:
                # cp.asarray copies H→D (verts is numpy), so we can
                # mutate it in-place without an extra GPU copy.
                updated_verts_gpu = cp.asarray(verts)
                if ve != 1.0:
                    updated_verts_gpu[2::3] *= ve

                if is_curve:
                    rtx.add_curve_geometry(
                        gid, updated_verts_gpu,
                        cp.asarray(widths), cp.asarray(indices))
                else:
                    rtx.add_geometry(gid, updated_verts_gpu, cp.asarray(indices))
                self._active_gids.add(gid)

                if accessor is not None:
                    accessor._geometry_colors[gid] = self._colors.get(gid, (0.6, 0.6, 0.6))
                    vx = cp.asarray(verts[0::3])
                    vy = cp.asarray(verts[1::3])
                    orig_base_z_np = _bilinear_terrain_z(
                        gpu_base_terrain, vx, vy,
                        base_psx, base_psy).get()
                    if is_curve:
                        accessor._baked_meshes[gid] = (
                            verts.copy(), widths.copy(), indices.copy(), orig_base_z_np)
                    else:
                        accessor._baked_meshes[gid] = (verts.copy(), indices.copy(), orig_base_z_np)
            else:
                updated_verts = verts.copy()
                if ve != 1.0:
                    updated_verts[2::3] *= ve

                if is_curve:
                    rtx.add_curve_geometry(gid, updated_verts, widths, indices)
                else:
                    rtx.add_geometry(gid, updated_verts, indices)
                self._active_gids.add(gid)

                if accessor is not None:
                    accessor._geometry_colors[gid] = self._colors.get(gid, (0.6, 0.6, 0.6))
                    orig_base_z = _bilinear_terrain_z(
                        base_terrain_np, verts[0::3], verts[1::3],
                        base_psx, base_psy)
                    if is_curve:
                        accessor._baked_meshes[gid] = (
                            verts.copy(), widths.copy(), indices.copy(), orig_base_z)
                    else:
                        accessor._baked_meshes[gid] = (verts.copy(), indices.copy(), orig_base_z)

        if accessor is not None:
            accessor._geometry_colors_dirty = True

        # Refresh viewer geometry tracking (same pattern as FIRMS toggle)
        from ..viewer.terrain_lod import is_terrain_lod_gid
        viewer._all_geometries = rtx.list_geometries()
        groups = set()
        for g in viewer._all_geometries:
            if is_terrain_lod_gid(g):
                continue
            parts = g.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base = parts[0]
            else:
                base = g
            if base != 'terrain':
                groups.add(base)
        viewer._geometry_layer_order = ['none', 'all'] + sorted(groups)

        # Apply current visibility mode
        layer_idx = viewer._geometry_layer_idx
        if layer_idx < len(viewer._geometry_layer_order):
            layer_name = viewer._geometry_layer_order[layer_idx]
        else:
            layer_name = 'none'
            viewer._geometry_layer_idx = 0

        for geom_id in viewer._all_geometries:
            if layer_name == 'none':
                rtx.set_geometry_visible(geom_id, False)
            elif layer_name == 'all':
                rtx.set_geometry_visible(geom_id, True)
            else:
                parts = geom_id.rsplit('_', 1)
                base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else geom_id
                visible = (base_name == layer_name or geom_id == layer_name)
                rtx.set_geometry_visible(geom_id, visible)

        n_tris = 0
        n_segs = 0
        for g in merged:
            if len(merged[g]) == 3:
                n_segs += len(merged[g][2])
            else:
                n_tris += len(merged[g][1]) // 3
        parts = []
        if n_tris > 0:
            parts.append(f"{n_tris:,} triangles")
        if n_segs > 0:
            parts.append(f"{n_segs:,} curve segments")
        print(f"Mesh chunks: loaded {len(new_visible)} chunks, "
              f"{len(merged)} geometries ({', '.join(parts)})")
        return True
