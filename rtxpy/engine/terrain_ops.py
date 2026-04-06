"""Terrain LOD and mesh management subsystem for InteractiveViewer."""

import time

import numpy as np

from ..rtx import has_cupy
from .helpers import _bilinear_terrain_z, _add_overlay

if has_cupy:
    import cupy as cp


class TerrainOps:
    """Terrain LOD setup, resolution rebuild, VE rebuild, and scene mesh
    upload methods.

    Composition subsystem -- holds a reference to the full
    ``InteractiveViewer`` as ``self.v``.
    """

    def __init__(self, viewer):
        self.v = viewer

    def _update_scene_meshes(self, smm):
        """Upload scene mesh updates from the SceneMeshManager.

        Handles VE application, GPU upload, geometry color tracking,
        baked mesh caching, and geometry layer refresh.

        Parameters
        ----------
        smm : SceneMeshManager
            The scene mesh manager with dirty merged buffers.

        Returns
        -------
        bool
            True if any geometry was added or removed.
        """
        v = self.v
        prev_active = set(smm.active_gids)
        merged = smm.get_merged()
        rtx = v.rtx
        accessor = v._accessor
        ve = v.vertical_exaggeration
        changed = False

        # Remove gids no longer present
        for gid in smm.get_removed_gids(prev_active):
            if rtx.has_geometry(gid):
                rtx.remove_geometry(gid)
                changed = True
            if accessor is not None:
                accessor._baked_meshes.pop(gid, None)
                accessor._geometry_colors.pop(gid, None)

        # Get terrain data for Z re-snap / baked mesh cache
        base_terrain = v._base_raster.data
        if hasattr(base_terrain, 'get'):
            base_terrain_np = base_terrain.get()
        else:
            base_terrain_np = np.asarray(base_terrain)
        base_psx = v._base_pixel_spacing_x
        base_psy = v._base_pixel_spacing_y

        colors = smm.colors

        # Add/update merged gids
        for gid, data in merged.items():
            is_curve = len(data) == 3
            if is_curve:
                verts, widths, indices = data
            else:
                verts, indices = data

            updated_verts = verts.copy()
            if ve != 1.0:
                updated_verts[2::3] *= ve

            if is_curve:
                rtx.add_curve_geometry(gid, updated_verts, widths, indices)
            else:
                rtx.add_geometry(gid, updated_verts, indices)
            changed = True

            if accessor is not None:
                accessor._geometry_colors[gid] = colors.get(
                    gid, (0.6, 0.6, 0.6))
                orig_base_z = _bilinear_terrain_z(
                    base_terrain_np, verts[0::3], verts[1::3],
                    base_psx, base_psy)
                if is_curve:
                    accessor._baked_meshes[gid] = (
                        verts.copy(), widths.copy(), indices.copy(),
                        orig_base_z)
                else:
                    accessor._baked_meshes[gid] = (
                        verts.copy(), indices.copy(), orig_base_z)

        if changed and accessor is not None:
            accessor._geometry_colors_dirty = True
            v._geometry_colors_builder = accessor._build_geometry_colors_gpu

        # Refresh geometry layer tracking
        if changed:
            from ..viewer.terrain_lod import is_terrain_lod_gid
            v._all_geometries = rtx.list_geometries()
            groups = set()
            for g in v._all_geometries:
                if is_terrain_lod_gid(g):
                    continue
                parts = g.rsplit('_', 1)
                base = parts[0] if len(parts) == 2 and parts[1].isdigit() else g
                if base != 'terrain':
                    groups.add(base)
            v._geometry_layer_order = ['none', 'all'] + sorted(groups)

        return changed

    def _enable_terrain_lod(self):
        """Set up per-tile LOD terrain rendering.

        Creates a :class:`TerrainLODManager` that renders each tile at
        a distance-appropriate resolution.  Called once during __init__.
        """
        v = self.v
        from ..viewer.terrain_lod import TerrainLODManager

        # Get full-res terrain as numpy
        base = v._base_raster
        terrain_data = base.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        # Fill any remaining NaN at edges so both the LOD tile builder and
        # the render kernel see clean elevation.  Without this, NaN pixels
        # render as blue ocean water.  Use scipy nearest-neighbor fill
        # (fast, handles diagonal NaN borders from UTM reprojection) with
        # iterative neighbor-mean fallback.
        if np.any(np.isnan(terrain_np)):
            terrain_np = terrain_np.copy()
            nan_mask = np.isnan(terrain_np)
            if nan_mask.any():
                try:
                    from scipy.ndimage import distance_transform_edt
                    ind = distance_transform_edt(
                        nan_mask, return_distances=False,
                        return_indices=True)
                    terrain_np = terrain_np.copy()
                    terrain_np[nan_mask] = terrain_np[tuple(ind)][nan_mask]
                except ImportError:
                    # Fallback: iterative neighbor averaging
                    for _ in range(200):
                        still_nan = np.isnan(terrain_np)
                        if not still_nan.any():
                            break
                        padded = np.pad(terrain_np, 1, mode='edge')
                        neighbors = np.stack([
                            padded[:-2, 1:-1], padded[2:, 1:-1],
                            padded[1:-1, :-2], padded[1:-1, 2:],
                        ], axis=0)
                        with np.errstate(all='ignore'):
                            fill_vals = np.nanmean(neighbors, axis=0)
                        terrain_np = np.where(
                            still_nan & np.isfinite(fill_vals),
                            fill_vals, terrain_np)
            # Update raster so render kernel sees clean data too
            is_cupy = hasattr(base.data, 'get')
            if is_cupy:
                import cupy
                v.raster = v.raster.copy(
                    data=cupy.asarray(terrain_np))
            else:
                v.raster = v.raster.copy(data=terrain_np)

        # Remove the single terrain geometry
        if v.rtx.has_geometry('terrain'):
            v.rtx.remove_geometry('terrain')
        if v.rtx.has_geometry('terrain_skirt'):
            v.rtx.remove_geometry('terrain_skirt')

        # Check for a ChunkDataSource provided via explore(terrain_source=...)
        # During __init__, these are pre-set as class attributes since
        # post-init assignment hasn't happened yet.
        chunk_source = (getattr(v, '_terrain_source', None)
                        or getattr(v.__class__, '_pre_terrain_source', None))

        # Choose tile size.  When a chunk source is active, its chunk grid
        # defines the tile grid.  When a zarr chunk manager is active, align
        # to the zarr elevation chunk size so terrain tiles and mesh chunks
        # share the same spatial grid — one distance lookup drives both.
        H, W = terrain_np.shape
        if chunk_source is not None:
            tile_size = chunk_source.chunk_shape[0]
            print(f"LOD tile size {tile_size} (from chunk source)")
        elif (v._chunk_manager is not None
                and v._chunk_manager._chunk_h == v._chunk_manager._chunk_w):
            tile_size = v._chunk_manager._chunk_h
            print(f"LOD tile size {tile_size} (aligned to zarr chunk grid)")
        else:
            tile_size = max(32, min(256, max(H, W) // 8))
            print(f"LOD tile size {tile_size}")

        mgr = TerrainLODManager(
            terrain_np,
            tile_size=tile_size,
            pixel_spacing_x=v._base_pixel_spacing_x,
            pixel_spacing_y=v._base_pixel_spacing_y,
            max_lod=3,
            base_subsample=v.subsample_factor,
            chunk_source=chunk_source,
        )
        # Carry forward any world offset from a previous terrain reload
        ox = v.terrain._world_offset_x
        oy = v.terrain._world_offset_y
        if ox != 0.0 or oy != 0.0:
            mgr.set_offset(ox, oy)
        # Enable tile streaming if a data callback was provided
        tile_data_fn = (getattr(v, '_tile_data_fn', None)
                        or getattr(v.__class__, '_pre_tile_data_fn', None))
        if tile_data_fn is not None:
            mgr.set_tile_data_fn(tile_data_fn)
            # Pass CRS coordinate transform so tile_data_fn receives
            # actual CRS coordinates (e.g. UTM) instead of viewer
            # world-space coords (pixel * abs(spacing) + offset).
            try:
                x = base.coords['x'].values
                y = base.coords['y'].values
                if len(x) >= 2 and len(y) >= 2:
                    crs_dx = float(x[1] - x[0])
                    crs_dy = float(y[1] - y[0])
                    mgr.set_crs_transform(float(x[0]), float(y[0]),
                                          crs_dx, crs_dy)
                    v._minimap_crs_transform = (
                        float(x[0]), float(y[0]), crs_dx, crs_dy)
            except (KeyError, AttributeError):
                pass
        # Wire scene zarr for placed geometry loading through LOD manager
        scene_zarr = (getattr(v, '_scene_zarr', None)
                      or getattr(v.__class__, '_pre_scene_zarr', None))
        if scene_zarr is not None:
            mgr.set_scene_zarr(scene_zarr)

        v._terrain_lod_manager = mgr
        v.lod_enabled = True

        # Create per-tile overlay and texture managers for LOD-aware compositing
        from ..viewer.overlay_tiles import OverlayTileManager, TextureTileManager
        v._overlay_tile_mgr = OverlayTileManager(tile_size)
        v._texture_tile_mgr = TextureTileManager(tile_size)
        # Register tile lifecycle callbacks so both managers stay
        # in sync with the LOD tile set automatically.
        otm = v._overlay_tile_mgr
        ttm = v._texture_tile_mgr

        # With LOD active, basemap goes through per-tile lazy fetch — stop
        # any monolithic XYZ tile fetch that was started before LOD enable.
        if v._tile_service is not None:
            v._tile_service._generation += 1  # cancel in-flight fetches

        # Lazy per-tile basemap fetching.  Each tile's CRS bounds are
        # converted to WGS84 and XYZ map tiles are fetched in background
        # threads, then stored in the TextureTileManager.
        _crs_origin = mgr._crs_origin      # (crs_x0, crs_y0) or None
        _crs_spacing = mgr._crs_spacing    # (crs_dx, crs_dy) or None
        _ts = tile_size
        _viewer = v
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _basemap_executor = _TPE(max_workers=4)
        _basemap_pending = set()  # tiles currently being fetched

        def _on_tile_added(tr, tc, elev):
            # Don't call otm/ttm.invalidate() here — set_tile() already
            # marks dirty when actual data arrives.  Invalidating on
            # every terrain LOD change forces composite rebuild + GPU
            # upload every frame during camera movement.

            # Lazy-fetch basemap for this tile in background.
            # Use elev.shape to get actual tile dimensions (edge tiles
            # may be smaller than tile_size).
            if (_crs_origin is not None
                    and _crs_spacing is not None
                    and _viewer._tiles_enabled
                    and _viewer._tile_service is not None
                    and (tr, tc) not in _basemap_pending
                    and not ttm.has_tile(tr, tc)):
                th = elev.shape[0] if elev is not None else _ts
                tw = elev.shape[1] if elev is not None else _ts
                _basemap_pending.add((tr, tc))
                _basemap_executor.submit(
                    _fetch_tile_basemap, tr, tc, th, tw)

        def _fetch_tile_basemap(tr, tc, th, tw):
            """Fetch basemap RGB for a single LOD tile and store it."""
            try:
                crs_x0, crs_y0 = _crs_origin
                crs_dx, crs_dy = _crs_spacing
                c0 = tc * _ts
                r0 = tr * _ts
                # CRS bounds: pixel-center to pixel-center so linspace
                # produces exact pixel coordinates.  Using c0+tw would
                # overshoot by one pixel and stretch the basemap.
                cx0 = crs_x0 + c0 * crs_dx
                cy0 = crs_y0 + r0 * crs_dy
                cx1 = crs_x0 + (c0 + tw - 1) * crs_dx
                cy1 = crs_y0 + (r0 + th - 1) * crs_dy
                x_min, x_max = min(cx0, cx1), max(cx0, cx1)
                y_min, y_max = min(cy0, cy1), max(cy0, cy1)
                rgb = _viewer._tile_service.fetch_rgb_for_bounds(
                    x_min, y_min, x_max, y_max, th, tw)
                if rgb is not None and not np.all(rgb == 0):
                    ttm.set_tile(tr, tc, rgb)
            except Exception:
                pass
            finally:
                _basemap_pending.discard((tr, tc))

        def _on_tile_removed(tr, tc):
            # Only remove basemap texture — it's re-fetched lazily
            # via _on_tile_added when the tile returns.  Overlay data
            # is bulk-populated from populate_from_array() and won't
            # be re-created on re-add, so we must keep it.
            ttm.remove_tile(tr, tc)

        mgr.set_tile_callbacks(
            on_added=_on_tile_added,
            on_removed=_on_tile_removed,
        )
        # If an overlay already exists, slice it into per-tile chunks
        if v._active_overlay_data is not None:
            n_tr = (H + tile_size - 1) // tile_size
            n_tc = (W + tile_size - 1) // tile_size
            active_name = None
            if v._terrain_layer_idx < len(v._terrain_layer_order):
                active_name = v._terrain_layer_order[
                    v._terrain_layer_idx]
            for name, data in v._overlay_layers.items():
                if name == active_name:
                    v._overlay_tile_mgr.populate_from_array(
                        data, tile_size, n_tr, n_tc)
                    lut = v._overlay_color_luts.get(name)
                    if lut is not None:
                        v._overlay_tile_mgr.set_color_lut(lut)

        # Basemap tiles are fetched lazily via _on_tile_added callback —
        # no monolithic texture slicing needed.

        # Wire LOD manager into HydroManager for streaming support
        v.hydro_mgr.set_lod_manager(mgr)
        if tile_data_fn is not None:
            v.hydro_mgr.set_tile_data_fn(tile_data_fn)
        try:
            x = base.coords['x'].values
            y = base.coords['y'].values
            if len(x) >= 2 and len(y) >= 2:
                v.hydro_mgr.set_crs_transform(
                    float(x[0]), float(y[0]),
                    float(x[1] - x[0]), float(y[1] - y[0]))
        except (KeyError, AttributeError):
            pass

        # When tile_data_fn streaming is active, use TIN meshes for ALL
        # tiles (including LOD 0) so initial-extent and streaming tiles
        # render identically.  chunk_source.supports_streaming is different
        # — those are still bounded in-grid tiles, so heightfield is fine.
        has_tile_data_fn = mgr._tile_data_fn is not None
        if not has_tile_data_fn:
            mgr.enable_heightfield_lod0()

        # Enable threaded building and batched upload BEFORE the initial
        # build so all tiles go through the batch path from the start.
        # This avoids creating N individual GAS entries that persist
        # alongside the batched ones.
        mgr.enable_threaded_building()
        mgr.enable_batched_upload()

        # Force initial tile build — no build limit so all in-bounds
        # tiles appear on the first frame (no progressive pop-in on
        # enable).  Streaming tiles build progressively after launch.
        # Use terrain center as fallback if camera position isn't set yet
        # (called from __init__ before run() sets the start position).
        cam_pos = v.position
        if cam_pos is None:
            H, W = v.terrain_shape
            cx = W * v.pixel_spacing_x * 0.5
            cy = H * v.pixel_spacing_y * 0.5
            cam_pos = np.array([cx, cy, 0.0])
        saved_limit = mgr.per_tick_build_limit
        saved_streaming = mgr._streaming
        mgr._streaming = False  # only in-bounds tiles on initial build
        mgr.per_tick_build_limit = 10000
        mgr.update(cam_pos, v.rtx,
                    ve=v.vertical_exaggeration, force=True,
                    camera_front=v._get_front(), fov=v.camera.fov)
        mgr.per_tick_build_limit = saved_limit
        mgr._streaming = saved_streaming
        # Force one more update so streaming tiles begin building
        if saved_streaming:
            mgr._last_update_pos = None
        # Only render if camera position has been initialised (run() sets it).
        # During __init__ the position is still None.
        if v.position is not None:
            v._update_frame()

    def _rebuild_at_resolution(self, factor):
        """Rebuild terrain mesh at a different subsample factor.

        Subsamples the original raster by ``factor`` (1 = full res, 2 = half,
        etc.), rebuilds the terrain geometry, re-snaps any placed meshes to the
        new surface, and refreshes the minimap.

        Parameters
        ----------
        factor : int
            Subsample factor (1, 2, 4, or 8).
        """
        v = self.v
        from .. import mesh as mesh_mod

        v.subsample_factor = factor

        # Update LOD manager's base subsample and force tile rebuild
        if v._terrain_lod_manager is not None:
            v._terrain_lod_manager.set_base_subsample(factor)
            v._terrain_lod_manager.update(
                v.position, v.rtx,
                ve=v.vertical_exaggeration, force=True,
                camera_front=v._get_front(), fov=v.camera.fov)

        base = v._base_raster

        # 1. Subsample the raster
        if factor > 1:
            sub = base.isel(
                {base.dims[0]: slice(None, None, factor),
                 base.dims[1]: slice(None, None, factor)}
            )
        else:
            sub = base

        v.raster = sub
        v._wind_terrain_np = None  # invalidate cached terrain
        v._hydro_terrain_np = None
        v._d_base_frame = None     # invalidate GPU wind/hydro buffers
        v._d_wind_scratch = None
        v._d_cloud_fog_map = None  # invalidate cached cloud fog map
        H, W = sub.shape
        v.terrain_shape = (H, W)

        # 2. Update pixel spacing
        v.pixel_spacing_x = v._base_pixel_spacing_x * factor
        v.pixel_spacing_y = v._base_pixel_spacing_y * factor

        # 3. Get terrain_np for elevation stats
        ve = v.vertical_exaggeration
        terrain_data = sub.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        v.elev_min = float(np.nanmin(terrain_np)) * ve
        v.elev_max = float(np.nanmax(terrain_np)) * ve
        v.elev_mean = float(np.nanmean(terrain_np)) * ve

        # Update land-only color range with VE
        f = v.subsample_factor
        wm = v._water_mask[::f, ::f] if f > 1 else v._water_mask
        land_pixels = terrain_np[~wm[:terrain_np.shape[0], :terrain_np.shape[1]]]
        if land_pixels.size > 0:
            v._land_color_range = (float(np.nanmin(land_pixels)) * ve,
                                      float(np.nanmax(land_pixels)) * ve)

        # 5. Subsample overlay layers
        if v._base_overlay_layers:
            v._overlay_layers = {}
            for name, data in v._base_overlay_layers.items():
                if factor > 1:
                    v._overlay_layers[name] = data[::factor, ::factor]
                else:
                    v._overlay_layers[name] = data
            v._overlay_names = list(v._overlay_layers.keys())
            # Rebuild terrain layer order with new overlay names
            v._terrain_layer_order = ['elevation'] + list(v._overlay_names)
            if v._terrain_layer_idx >= len(v._terrain_layer_order):
                v._terrain_layer_idx = 0
            # Reset active overlay data if an overlay is selected
            terrain_name = v._terrain_layer_order[v._terrain_layer_idx]
            if terrain_name != 'elevation' and terrain_name in v._overlay_layers:
                v._active_overlay_data = v._overlay_layers[terrain_name]
                v._overlay_as_water = (
                    terrain_name.startswith('flood_')
                    or (terrain_name == 'stream_link'
                        and v._hydro_enabled))
                v._active_overlay_color_lut = v._overlay_color_luts.get(
                    terrain_name)

        # 6. Invalidate scene mesh state (re-snap Z at new resolution).
        #    Raw zarr data in _cache is resolution-independent, so we keep
        #    it and only clear the active/visible state so re-merge is cheap.
        smm = (v._terrain_lod_manager.scene_mesh_manager
               if v._terrain_lod_manager is not None else None)
        if smm is not None:
            for gid in list(smm.active_gids):
                if hasattr(v, '_baked_meshes'):
                    v._baked_meshes.pop(gid, None)
                if v._accessor is not None:
                    v._accessor._baked_meshes.pop(gid, None)
                if v.rtx is not None and v.rtx.has_geometry(gid):
                    v.rtx.remove_geometry(gid)
            smm.clear_active()
        elif v._chunk_manager is not None:
            for gid in list(v._chunk_manager._active_gids):
                if hasattr(v, '_baked_meshes'):
                    v._baked_meshes.pop(gid, None)
                if v._accessor is not None:
                    v._accessor._baked_meshes.pop(gid, None)
                if v.rtx is not None and v.rtx.has_geometry(gid):
                    v.rtx.remove_geometry(gid)
            v._chunk_manager._visible.clear()
            v._chunk_manager._active_gids.clear()
            if hasattr(v, 'position'):
                v._chunk_manager.update(v.position[0], v.position[1], v)

        # 7. Re-snap placed meshes to new terrain surface
        # Invalidate GPU terrain cache (terrain changed) and upload once
        v._gpu_terrain = None
        if v.rtx is not None:
            gpu_terrain = None
            if has_cupy:
                gpu_terrain = cp.asarray(terrain_np)
                v._gpu_terrain = gpu_terrain
            from ..viewer.terrain_lod import is_terrain_lod_gid
            for geom_id in v.rtx.list_geometries():
                if is_terrain_lod_gid(geom_id):
                    continue
                # Baked meshes — re-snap Z to new terrain surface + VE
                if hasattr(v, '_baked_meshes') and geom_id in v._baked_meshes:
                    baked = v._baked_meshes[geom_id]
                    is_curve = (len(baked) == 4)
                    baked_key = (factor, geom_id)
                    if baked_key in v._baked_mesh_cache:
                        cached = v._baked_mesh_cache[baked_key]
                        if is_curve:
                            scaled_v, orig_w, orig_idx = cached
                            v.rtx.add_curve_geometry(
                                geom_id, scaled_v, orig_w, orig_idx)
                        else:
                            scaled_v, orig_idx = cached
                            v.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    else:
                        if is_curve:
                            orig_v, orig_w, orig_idx, orig_base_z = baked
                        elif len(baked) == 3:
                            orig_v, orig_idx, orig_base_z = baked
                        else:
                            orig_v, orig_idx = baked
                            orig_base_z = None

                        n_verts = len(orig_v) // 3
                        use_gpu = (gpu_terrain is not None
                                   and orig_base_z is not None
                                   and n_verts > 1000)

                        if use_gpu:
                            vx = cp.asarray(orig_v[0::3])
                            vy = cp.asarray(orig_v[1::3])
                            new_base_z = _bilinear_terrain_z(
                                gpu_terrain, vx, vy,
                                v.pixel_spacing_x, v.pixel_spacing_y)
                            z_offset = cp.asarray(orig_v[2::3]) - cp.asarray(orig_base_z)
                            new_z = (new_base_z + z_offset) * ve
                            scaled_v_gpu = cp.asarray(orig_v.copy())
                            scaled_v_gpu[2::3] = new_z
                            if is_curve:
                                v._baked_mesh_cache[baked_key] = (
                                    scaled_v_gpu.get().copy(), orig_w, orig_idx)
                                v.rtx.add_curve_geometry(
                                    geom_id, scaled_v_gpu,
                                    cp.asarray(orig_w),
                                    cp.asarray(orig_idx))
                            else:
                                v._baked_mesh_cache[baked_key] = (
                                    scaled_v_gpu.get().copy(), orig_idx)
                                v.rtx.add_geometry(geom_id, scaled_v_gpu,
                                                      cp.asarray(orig_idx))
                        else:
                            scaled_v = orig_v.copy()
                            if orig_base_z is not None:
                                vx = orig_v[0::3]
                                vy = orig_v[1::3]
                                new_base_z = _bilinear_terrain_z(
                                    terrain_np, vx, vy,
                                    v.pixel_spacing_x, v.pixel_spacing_y)
                                z_offset = orig_v[2::3] - orig_base_z
                                scaled_v[2::3] = (new_base_z + z_offset) * ve
                            else:
                                scaled_v[2::3] *= ve
                            if is_curve:
                                v._baked_mesh_cache[baked_key] = (
                                    scaled_v.copy(), orig_w, orig_idx)
                                v.rtx.add_curve_geometry(
                                    geom_id, scaled_v, orig_w, orig_idx)
                            else:
                                v._baked_mesh_cache[baked_key] = (
                                    scaled_v.copy(), orig_idx)
                                v.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    continue
                # Instanced meshes — update transform Z from terrain
                transform = v.rtx.get_geometry_transform(geom_id)
                if transform is None:
                    continue
                wx, wy = transform[3], transform[7]
                z = float(_bilinear_terrain_z(
                    terrain_np,
                    np.array([wx], dtype=np.float32),
                    np.array([wy], dtype=np.float32),
                    v.pixel_spacing_x, v.pixel_spacing_y)[0]) * ve
                transform[11] = z
                v.rtx.update_transform(geom_id, transform)

        # 8. Re-snap all observer drones to new terrain
        for obs in v._observers.values():
            if obs.drone_placed and obs.position is not None:
                v._update_observer_drone_for(obs)

        # 9. Recompute minimap
        v._minimap_bg_extent = None
        v._compute_minimap_background()

        # 10. Clear viewshed cache (no longer matches terrain)
        v._viewshed_cache = None
        for obs in v._observers.values():
            obs.viewshed_cache = None
            if obs.viewshed_enabled:
                obs.viewshed_enabled = False
        if v.viewshed_enabled:
            v.viewshed_enabled = False
            print("  Viewshed disabled (terrain changed). Press V to recalculate.")

        print(f"Resolution: {W}x{H} (subsample {factor}x)")
        v._update_frame()

    def _rebuild_vertical_exaggeration(self, ve):
        """Rebuild terrain mesh with a new vertical exaggeration factor.

        Parameters
        ----------
        ve : float
            Vertical exaggeration multiplier applied to elevation values.
        """
        v = self.v
        from .. import mesh as mesh_mod

        v.vertical_exaggeration = ve

        # Force re-upload of all LOD tiles with new VE
        if v._terrain_lod_manager is not None:
            v._terrain_lod_manager._tile_lods.clear()
            v._terrain_lod_manager.update(
                v.position, v.rtx, ve=ve, force=True,
                camera_front=v._get_front(), fov=v.camera.fov)

        terrain_data = v.raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        # Update elevation stats (scaled)
        v.elev_min = float(np.nanmin(terrain_np)) * ve
        v.elev_max = float(np.nanmax(terrain_np)) * ve
        v.elev_mean = float(np.nanmean(terrain_np)) * ve

        # Update land-only color range with VE
        f = v.subsample_factor
        wm = v._water_mask[::f, ::f] if f > 1 else v._water_mask
        land_pixels = terrain_np[~wm[:terrain_np.shape[0], :terrain_np.shape[1]]]
        if land_pixels.size > 0:
            v._land_color_range = (float(np.nanmin(land_pixels)) * ve,
                                      float(np.nanmax(land_pixels)) * ve)

        # Re-snap placed meshes to scaled terrain
        # Invalidate GPU terrain cache (VE changed terrain Z) and upload once
        v._gpu_terrain = None
        if v.rtx is not None:
            gpu_terrain = None
            if has_cupy:
                gpu_terrain = cp.asarray(terrain_np)
                v._gpu_terrain = gpu_terrain
            from ..viewer.terrain_lod import is_terrain_lod_gid
            for geom_id in v.rtx.list_geometries():
                if is_terrain_lod_gid(geom_id):
                    continue
                # Baked meshes (merged buildings/curves) — re-snap Z to terrain + VE
                if hasattr(v, '_baked_meshes') and geom_id in v._baked_meshes:
                    baked = v._baked_meshes[geom_id]
                    is_curve = (len(baked) == 4)
                    if is_curve:
                        orig_v, orig_w, orig_idx, orig_base_z = baked
                    elif len(baked) == 3:
                        orig_v, orig_idx, orig_base_z = baked
                    else:
                        orig_v, orig_idx = baked
                        orig_base_z = None

                    n_verts = len(orig_v) // 3
                    use_gpu = (gpu_terrain is not None
                               and orig_base_z is not None
                               and n_verts > 1000)

                    if use_gpu:
                        vx = cp.asarray(orig_v[0::3])
                        vy = cp.asarray(orig_v[1::3])
                        cur_base_z = _bilinear_terrain_z(
                            gpu_terrain, vx, vy,
                            v.pixel_spacing_x, v.pixel_spacing_y)
                        z_offset = cp.asarray(orig_v[2::3]) - cp.asarray(orig_base_z)
                        new_z = (cur_base_z + z_offset) * ve
                        scaled_v_gpu = cp.asarray(orig_v.copy())
                        scaled_v_gpu[2::3] = new_z
                        if is_curve:
                            v.rtx.add_curve_geometry(
                                geom_id, scaled_v_gpu,
                                cp.asarray(orig_w),
                                cp.asarray(orig_idx))
                        else:
                            v.rtx.add_geometry(geom_id, scaled_v_gpu,
                                                  cp.asarray(orig_idx))
                    else:
                        scaled_v = orig_v.copy()
                        if orig_base_z is not None:
                            vx = orig_v[0::3]
                            vy = orig_v[1::3]
                            cur_base_z = _bilinear_terrain_z(
                                terrain_np, vx, vy,
                                v.pixel_spacing_x, v.pixel_spacing_y)
                            z_offset = orig_v[2::3] - orig_base_z
                            scaled_v[2::3] = (cur_base_z + z_offset) * ve
                        else:
                            scaled_v[2::3] *= ve
                        if is_curve:
                            v.rtx.add_curve_geometry(
                                geom_id, scaled_v, orig_w, orig_idx)
                        else:
                            v.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    continue
                # Instanced meshes — update transform Z from terrain
                transform = v.rtx.get_geometry_transform(geom_id)
                if transform is None:
                    continue
                wx, wy = transform[3], transform[7]
                z = float(_bilinear_terrain_z(
                    terrain_np,
                    np.array([wx], dtype=np.float32),
                    np.array([wy], dtype=np.float32),
                    v.pixel_spacing_x, v.pixel_spacing_y)[0]) * ve
                transform[11] = z
                v.rtx.update_transform(geom_id, transform)

        # Re-snap all observer drones to updated terrain
        for obs in v._observers.values():
            if obs.drone_placed and obs.position is not None:
                v._update_observer_drone_for(obs)

        # Clear viewshed cache
        v._viewshed_cache = None
        for obs in v._observers.values():
            obs.viewshed_cache = None
            if obs.viewshed_enabled:
                obs.viewshed_enabled = False
        if v.viewshed_enabled:
            v.viewshed_enabled = False
            print("  Viewshed disabled (terrain changed). Press V to recalculate.")

        print(f"Vertical exaggeration: {ve:.2f}x")
        v._update_frame()
