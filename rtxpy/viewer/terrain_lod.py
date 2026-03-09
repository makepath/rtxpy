"""Distance-based terrain LOD manager.

Divides a terrain raster into a grid of tiles.  Each tile is assigned
an LOD level based on its distance to the camera, controlling the mesh
resolution (LOD 0 = full detail, LOD *k* = 2^k subsampling).  Tiles
are built as individual GAS entries in the OptiX IAS so the raytracer
traverses only the detail actually needed.

Adjacent tiles with different LOD levels are stitched at boundaries
(boundary vertex Z values interpolated from the coarser neighbor's
pyramid data) to eliminate T-junction cracks.

Mesh construction (triangulation, normals, skirt, vertex transforms)
runs in a background ``ThreadPoolExecutor`` so the viewer tick loop
stays responsive while tiles build.  Completed meshes are collected
next tick and uploaded to the GPU.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

from ..lod import (compute_lod_distances,
                    compute_lod_level_with_hysteresis,
                    compute_tile_roughness)


class TerrainLODManager:
    """Manages per-tile terrain LOD in the interactive viewer.

    Parameters
    ----------
    terrain_np : np.ndarray
        Full-resolution elevation array, shape ``(H, W)``.
    tile_size : int
        Tile edge length in full-resolution pixels (default 128).
    pixel_spacing_x, pixel_spacing_y : float
        World-space size of one raster pixel.
    max_lod : int
        Maximum LOD level.  Level *k* subsamples the tile by ``2^k``.
    lod_distance_factor : float
        Controls how far each LOD band extends.  First transition
        at ``tile_diagonal * factor``, doubling per level.
    base_subsample : int
        Global base subsample factor (from R/Shift+R).  LOD 0 uses
        this value; LOD *k* uses ``base_subsample * 2^k``.
    """

    __slots__ = (
        '_terrain_np', '_tile_size', '_psx', '_psy',
        '_max_lod', '_lod_distances', '_lod_distance_factor',
        '_H', '_W', '_n_tile_rows', '_n_tile_cols',
        '_tile_centers', '_tile_lods', '_tile_cache',
        '_active_tiles', '_last_update_pos', '_last_update_fwd',
        '_update_threshold_sq',
        '_base_subsample', '_pyramid_cache', '_has_in_flight_work',
        '_tile_half_diag_sq', '_tile_half_diag',
        '_max_dist_sq', '_max_dist',
        '_tile_world_x', '_tile_world_y',
        '_offset_x', '_offset_y',
        'max_tiles', 'per_tick_build_limit',
        '_tile_roughness',
        '_tile_data_fn', '_streaming',
        '_tile_data_cache', '_io_futures',
        '_executor', '_pending_futures', '_threaded',
        '_batched', '_lod_tile_meshes', '_dirty_lods', '_batch_gids',
        '_hf_enabled', '_hf_gid', '_hf_dirty', '_hf_tile_size',
        '_build_retries',
    )

    def __init__(self, terrain_np, tile_size=128,
                 pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 max_lod=3, lod_distance_factor=3.0,
                 base_subsample=1):
        self._terrain_np = terrain_np
        self._tile_size = tile_size
        self._psx = pixel_spacing_x
        self._psy = pixel_spacing_y
        self._max_lod = max_lod
        self._lod_distance_factor = lod_distance_factor
        self._base_subsample = base_subsample

        H, W = terrain_np.shape
        self._H = H
        self._W = W
        self._n_tile_rows = (H + tile_size - 1) // tile_size
        self._n_tile_cols = (W + tile_size - 1) // tile_size

        # World-space offset applied to tile vertices (for stable camera
        # across terrain reloads).  Must be set before _recompute_tile_centers.
        self._offset_x = 0.0
        self._offset_y = 0.0

        # Streaming: when set, tiles outside terrain_np bounds are fetched
        # from this callback instead of from the pyramid cache.
        self._tile_data_fn = None
        self._streaming = False
        self._tile_data_cache = {}  # cache_key -> np.ndarray (prefetched I/O)
        self._io_futures = {}       # cache_key -> Future (in-flight prefetches)

        # Pre-compute tile centres in world coordinates
        self._tile_centers = {}
        self._recompute_tile_centers()

        # LOD distance thresholds (store squared for fast comparison)
        tile_diag_sq = ((tile_size * pixel_spacing_x) ** 2
                        + (tile_size * pixel_spacing_y) ** 2)
        self._tile_half_diag_sq = tile_diag_sq * 0.25
        self._tile_half_diag = np.sqrt(tile_diag_sq * 0.25)
        self._lod_distances = compute_lod_distances(
            np.sqrt(tile_diag_sq), factor=lod_distance_factor,
            max_lod=max_lod)
        max_dist = (self._lod_distances[-1] * 3.0
                    if self._lod_distances else float('inf'))
        self._max_dist_sq = max_dist * max_dist
        self._max_dist = max_dist
        # Tile world-space dimensions (for streaming radius computation)
        self._tile_world_x = tile_size * abs(pixel_spacing_x)
        self._tile_world_y = tile_size * abs(pixel_spacing_y)

        # Lazy pyramid cache: level -> downsampled array (built on demand)
        self._pyramid_cache = {}

        # Per-tile roughness scale factors for terrain-adaptive LOD.
        # Rough tiles keep finer detail at greater distance; smooth tiles
        # drop to coarser LOD sooner.  Computed from full-res terrain.
        self._tile_roughness = {}  # (tr, tc) -> scale float
        self._compute_all_roughness()

        # Progressive loading limits
        self.max_tiles = float('inf')  # no cap; frustum + distance cull suffice
        self.per_tick_build_limit = 8  # max new tile meshes built per update

        # Per-tile state
        self._tile_lods = {}   # (tr, tc) -> current LOD level
        self._tile_cache = {}  # (tr, tc, lod, base_sub) -> (verts, indices, normals)
        self._active_tiles = set()  # GAS IDs currently in the scene
        self._build_retries = {}  # cache_key -> int (failed build count)

        # Movement/rotation threshold before re-evaluating LOD
        self._last_update_pos = None
        self._last_update_fwd = None
        update_threshold = np.sqrt(tile_diag_sq) * 0.25
        self._update_threshold_sq = update_threshold * update_threshold
        self._has_in_flight_work = False

        # Threaded mesh building: tiles are built in background threads
        # and collected next tick.  Disabled by default for test compat;
        # engine.py enables it after construction.
        self._executor = None
        self._pending_futures = {}  # cache_key -> Future
        self._threaded = False

        # Batched upload: tiles at the same LOD are concatenated into a
        # single GAS to reduce IAS instance count.  Disabled by default;
        # engine.py enables it after construction.
        self._batched = False
        self._lod_tile_meshes = {}  # {lod: {(tr,tc): (verts, indices, normals)}}
        self._dirty_lods = set()    # LOD levels needing batch rebuild
        self._batch_gids = set()    # batch GAS IDs currently in the scene

        # Heightfield for LOD 0: in-bounds LOD 0 tiles use heightfield
        # ray marching instead of explicit triangles.  Disabled by default;
        # engine.py enables it after construction.
        self._hf_enabled = False
        self._hf_gid = 'terrain_lod_hf'
        self._hf_dirty = False
        self._hf_tile_size = 32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable_threaded_building(self, max_workers=4):
        """Enable background mesh building with a thread pool.

        Call once after construction to move tile triangulation off the
        main thread.  Completed meshes are collected in :meth:`update`.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._threaded = True

    def enable_batched_upload(self):
        """Enable batched GAS upload for LOD tiles.

        Instead of creating individual GAS entries per tile, tiles at
        the same LOD level are concatenated into a single GAS.  This
        reduces IAS instance count from ~50-100 to ~4 (one per active
        LOD level), cutting per-ray IAS traversal overhead.

        Batch GAS entries are rebuilt when the set of tiles at any LOD
        level changes (tile added, removed, or LOD transition).
        """
        self._batched = True

    def enable_heightfield_lod0(self, hf_tile_size=32):
        """Use heightfield ray marching for LOD 0 in-bounds tiles.

        LOD 0 tiles are full-resolution regular grids — the ideal case
        for heightfield ray marching.  Heightfield uses bilinear
        interpolation for smooth normals and ~4 bytes/pixel vs ~16
        bytes/pixel for explicit triangles.

        LOD 1+ tiles and out-of-bounds streaming tiles continue to use
        TIN (triangle) meshes.

        Parameters
        ----------
        hf_tile_size : int
            AABB tile dimension within the heightfield GAS (default 32).
        """
        self._hf_enabled = True
        self._hf_tile_size = hf_tile_size
        self._hf_dirty = True

    def shutdown(self):
        """Shut down the background thread pool and clear batch state."""
        if self._executor is not None:
            for fut in self._pending_futures.values():
                fut.cancel()
            self._pending_futures.clear()
            for fut in self._io_futures.values():
                fut.cancel()
            self._io_futures.clear()
            self._tile_data_cache.clear()
            self._executor.shutdown(wait=False)
            self._executor = None
        self._threaded = False
        self._lod_tile_meshes.clear()
        self._dirty_lods.clear()
        self._batch_gids.clear()
        self._batched = False
        self._hf_dirty = False
        self._hf_enabled = False

    @property
    def n_tiles(self):
        """Total number of terrain tiles."""
        return self._n_tile_rows * self._n_tile_cols

    @property
    def tile_lods(self):
        """Current LOD assignment per tile: ``{(row, col): level}``."""
        return dict(self._tile_lods)

    def set_base_subsample(self, factor):
        """Update the global base subsample and invalidate cache."""
        if factor != self._base_subsample:
            self._base_subsample = factor
            self._cancel_pending()
            self._pyramid_cache.clear()
            self._tile_cache.clear()
            self._tile_data_cache.clear()
            self._lod_tile_meshes.clear()
            self._dirty_lods.clear()
            self._tile_lods.clear()
            if self._hf_enabled:
                self._hf_dirty = True

    def set_offset(self, offset_x, offset_y):
        """Set world-space offset for tile vertices.

        Used after terrain reload to keep the camera position stable:
        tile vertices are shifted so the same geographic point maps to
        the same world-space position as before the reload.
        """
        if offset_x != self._offset_x or offset_y != self._offset_y:
            self._offset_x = offset_x
            self._offset_y = offset_y
            self._cancel_pending()
            self._tile_cache.clear()
            self._tile_data_cache.clear()
            self._lod_tile_meshes.clear()
            self._dirty_lods.clear()
            self._tile_lods.clear()
            if self._hf_enabled:
                self._hf_dirty = True
            self._recompute_tile_centers()

    def set_tile_data_fn(self, fn):
        """Set a callback for loading tile elevation data on demand.

        When set, the tile grid becomes unbounded — tiles beyond the
        initial terrain array are fetched via this callback as the camera
        moves, enabling seamless terrain streaming.

        Parameters
        ----------
        fn : callable or None
            ``fn(x_min, y_min, x_max, y_max, target_samples)``
            returns ``np.ndarray`` of shape ``(H, W)`` (float32
            elevations, row 0 = northernmost) or ``None``.
            Called synchronously; consider wrapping in a thread pool
            for slow I/O backends (zarr over network).
        """
        self._tile_data_fn = fn
        self._streaming = fn is not None

    def _tile_center(self, tr, tc):
        """Return the world-space center of tile (tr, tc).

        Uses precomputed values for in-bounds tiles, otherwise
        computes on-the-fly for streaming tiles.
        """
        cached = self._tile_centers.get((tr, tc))
        if cached is not None:
            return cached
        ts = self._tile_size
        cx = (tc * ts + ts * 0.5) * self._psx + self._offset_x
        cy = (tr * ts + ts * 0.5) * self._psy + self._offset_y
        return (cx, cy)

    def _recompute_tile_centers(self):
        """Recompute tile centre positions from current grid dimensions."""
        self._tile_centers.clear()
        ts = self._tile_size
        H, W = self._H, self._W
        ox, oy = self._offset_x, self._offset_y
        for tr in range(self._n_tile_rows):
            for tc in range(self._n_tile_cols):
                r0 = tr * ts
                c0 = tc * ts
                r1 = min(r0 + ts, H)
                c1 = min(c0 + ts, W)
                cx = (c0 + c1) * 0.5 * self._psx + ox
                cy = (r0 + r1) * 0.5 * self._psy + oy
                self._tile_centers[(tr, tc)] = (cx, cy)

    def set_terrain(self, terrain_np, offset_x=None, offset_y=None):
        """Replace the terrain data (e.g. after dynamic reload)."""
        self._terrain_np = terrain_np
        H, W = terrain_np.shape
        self._H = H
        self._W = W
        self._n_tile_rows = (H + self._tile_size - 1) // self._tile_size
        self._n_tile_cols = (W + self._tile_size - 1) // self._tile_size
        if offset_x is not None:
            self._offset_x = offset_x
        if offset_y is not None:
            self._offset_y = offset_y
        self._recompute_tile_centers()
        self._compute_all_roughness()
        self._cancel_pending()
        self._pyramid_cache.clear()
        self._tile_cache.clear()
        self._tile_data_cache.clear()
        self._lod_tile_meshes.clear()
        self._dirty_lods.clear()
        self._tile_lods.clear()
        if self._hf_enabled:
            self._hf_dirty = True

    def update(self, camera_pos, rtx, ve=1.0, force=False,
               camera_front=None, fov=60.0):
        """Re-evaluate LOD per tile and rebuild changed tiles.

        Only tiles within LOD distance range are considered.  Tiles are
        sorted by distance and capped at ``max_tiles``.  At most
        ``per_tick_build_limit`` new meshes are built per call so the
        viewer stays responsive while tiles load progressively.

        Frustum culling prevents *building* tiles outside the view cone
        but never *removes* already-built tiles — they stay in the scene
        until they leave the distance range.  This avoids flicker when
        the camera rotates.

        LOD transitions use hysteresis (20% dead zone) to prevent
        popping when the camera oscillates near a threshold boundary.

        Parameters
        ----------
        camera_pos : array-like
            Camera position ``[x, y, z]`` in world coordinates.
        rtx : RTX
            Scene handle for adding/removing tile GAS.
        ve : float
            Current vertical exaggeration.
        force : bool
            If True, rebuild all tiles regardless of movement threshold.
        camera_front : array-like or None
            Camera forward direction ``[x, y, z]``.  When provided,
            tiles outside the view frustum are skipped for building.
        fov : float
            Horizontal field of view in degrees (used for frustum cull).

        Returns
        -------
        bool
            True if any tile GAS was added or updated.
        """
        cam_x, cam_y = float(camera_pos[0]), float(camera_pos[1])

        # Skip if camera hasn't moved or rotated enough
        if not force and not self._has_in_flight_work and self._last_update_pos is not None:
            dx = cam_x - self._last_update_pos[0]
            dy = cam_y - self._last_update_pos[1]
            moved = dx * dx + dy * dy >= self._update_threshold_sq
            # Check rotation change via forward-direction dot product
            rotated = False
            if not moved and camera_front is not None and self._last_update_fwd is not None:
                dot = (float(camera_front[0]) * self._last_update_fwd[0]
                       + float(camera_front[1]) * self._last_update_fwd[1])
                rotated = dot < 0.95  # ~18° change
            if not moved and not rotated:
                return False

        self._last_update_pos = (cam_x, cam_y)
        if camera_front is not None:
            self._last_update_fwd = (float(camera_front[0]),
                                     float(camera_front[1]))

        # Frustum culling setup: project forward to XY plane.
        # When looking steeply down (small XY component), the 2D
        # projection is unreliable — disable culling in that case.
        frustum_cos = None
        fwd_x = fwd_y = 0.0
        if camera_front is not None:
            fwd_x = float(camera_front[0])
            fwd_y = float(camera_front[1])
            fwd_len_sq = fwd_x * fwd_x + fwd_y * fwd_y
            # Only cull when the camera has a meaningful horizontal
            # component (fwd_len > 0.3 ≈ pitch shallower than ~73°).
            if fwd_len_sq > 0.09:  # 0.3^2
                fwd_len = math.sqrt(fwd_len_sq)
                fwd_x /= fwd_len
                fwd_y /= fwd_len
                # Widen frustum for steeper pitch: as fwd_len shrinks
                # toward 0.3 the ground footprint of the view cone grows.
                pitch_margin = 20.0 * (1.0 - fwd_len) / 0.7  # 0..~20°
                half_angle = math.radians(fov * 0.5 + 20.0 + pitch_margin)
                frustum_cos = math.cos(half_angle)

        tile_half_diag_sq = self._tile_half_diag_sq
        max_dist_sq = self._max_dist_sq

        # Pass 1: find all tiles within distance range, and flag which
        # are also inside the view frustum (candidates for building).
        # Uses squared distances to avoid sqrt per tile.
        in_frustum = []    # tiles that passed distance + frustum culling
        in_range_ids = set()  # all distance-valid tiles (frustum-independent)

        lod_distances = self._lod_distances
        max_lod = self._max_lod
        tile_lods = self._tile_lods

        # Determine tile iteration range.  For streaming mode, compute
        # tiles near the camera (unbounded grid).  Otherwise use the
        # fixed grid from the initial terrain array.
        if self._streaming:
            ts = self._tile_size
            tc_f = (cam_x - self._offset_x) / (ts * self._psx) - 0.5
            tr_f = (cam_y - self._offset_y) / (ts * self._psy) - 0.5
            tc_cam = int(round(tc_f))
            tr_cam = int(round(tr_f))
            r_tc = int(self._max_dist / self._tile_world_x) + 1
            r_tr = int(self._max_dist / self._tile_world_y) + 1
            tile_iter = ((tr, tc)
                         for tr in range(tr_cam - r_tr, tr_cam + r_tr + 1)
                         for tc in range(tc_cam - r_tc, tc_cam + r_tc + 1))
        else:
            tile_iter = ((tr, tc)
                         for tr in range(self._n_tile_rows)
                         for tc in range(self._n_tile_cols))

        for tr, tc in tile_iter:
            cx, cy = self._tile_center(tr, tc)
            dx_t = cx - cam_x
            dy_t = cy - cam_y
            dist_sq = dx_t * dx_t + dy_t * dy_t
            if dist_sq > max_dist_sq:
                continue

            # Track all in-range tiles (prevents eviction on rotate).
            gid = _tile_gid(tr, tc)
            in_range_ids.add(gid)

            # Frustum check — skip LOD/roughness for tiles behind the
            # camera.  Tiles close enough to overlap the view origin
            # (dist_sq <= tile_half_diag_sq) are always kept.
            dist = math.sqrt(dist_sq)
            if frustum_cos is not None and dist_sq > tile_half_diag_sq:
                inv_dist = 1.0 / dist
                cos_angle = (dx_t * fwd_x + dy_t * fwd_y) * inv_dist
                tile_margin = self._tile_half_diag * inv_dist
                if cos_angle + tile_margin < frustum_cos:
                    continue

            prev_lod = tile_lods.get((tr, tc), -1)
            # Terrain-adaptive LOD: scale distance by roughness so
            # rough tiles appear closer (keep detail) and smooth tiles
            # appear farther (drop detail sooner).
            roughness_scale = self._tile_roughness.get((tr, tc), 1.0)
            effective_dist = dist / roughness_scale if roughness_scale > 0 else dist
            lod = compute_lod_level_with_hysteresis(
                effective_dist, lod_distances, prev_lod)
            lod = min(lod, max_lod)
            entry = (dist_sq, tr, tc, lod, gid)
            in_frustum.append(entry)

        # Build candidates are the in-frustum tiles, sorted by distance
        in_frustum.sort()
        if len(in_frustum) > self.max_tiles:
            in_frustum = in_frustum[:self.max_tiles]

        # Two-pass build: fill holes (never-built tiles) before
        # spending budget on LOD transitions.  A tile at the wrong LOD
        # is much better than a missing tile.
        unbuilt = []
        lod_updates = []
        for entry in in_frustum:
            _dist_sq, tr, tc, lod, gid = entry
            prev_lod = tile_lods.get((tr, tc), -1)
            if lod == prev_lod and not force:
                continue
            if gid not in self._active_tiles:
                unbuilt.append(entry)
            else:
                lod_updates.append(entry)

        changed = False

        # --- Phase A: collect completed async builds ---
        self._collect_completed_builds()

        # --- Phase B: process tile queue (build/upload/stage) ---
        changed, pending = self._process_tile_queue(
            unbuilt + lod_updates, rtx, ve)

        # --- Phase C: prefetch I/O for streaming tiles ---
        if (self._streaming and self._threaded
                and self._executor is not None):
            self._prefetch_streaming_tiles(unbuilt + lod_updates)

        # --- Phase D: remove stale tiles ---
        changed |= self._remove_stale_tiles(rtx, in_range_ids)

        # --- Phase E: rebuild heightfield and batch GAS ---
        if self._hf_enabled and self._hf_dirty:
            changed |= self._rebuild_heightfield(rtx, ve)
        if self._batched:
            changed |= self._rebuild_batches(rtx)

        self._has_in_flight_work = (pending or bool(self._pending_futures)
                             or bool(self._io_futures))
        return changed

    # ------------------------------------------------------------------
    # update() helper phases
    # ------------------------------------------------------------------

    _MAX_BUILD_RETRIES = 3

    def _collect_completed_builds(self):
        """Phase A: harvest completed async mesh builds and I/O prefetches."""
        if self._pending_futures:
            for cache_key, fut in list(self._pending_futures.items()):
                if not fut.done():
                    continue
                del self._pending_futures[cache_key]
                exc = fut.exception()
                if exc is not None:
                    retries = self._build_retries.get(cache_key, 0) + 1
                    self._build_retries[cache_key] = retries
                    if retries >= self._MAX_BUILD_RETRIES:
                        logger.warning(
                            "tile %s build failed %d times, giving up: %s",
                            cache_key[:2], retries, exc)
                    else:
                        logger.debug(
                            "tile %s build failed (attempt %d): %s",
                            cache_key[:2], retries, exc)
                    continue
                self._build_retries.pop(cache_key, None)
                result = fut.result()
                if result is not None and result[0] is not None:
                    self._tile_cache[cache_key] = result

        if self._io_futures:
            for cache_key, fut in list(self._io_futures.items()):
                if not fut.done():
                    continue
                del self._io_futures[cache_key]
                exc = fut.exception()
                if exc is not None:
                    logger.debug("tile %s I/O prefetch failed: %s",
                                 cache_key[:2], exc)
                    continue
                result = fut.result()
                if result is not None:
                    self._tile_data_cache[cache_key] = result

    def _process_tile_queue(self, queue, rtx, ve):
        """Phase B: build, upload, or stage tiles from the priority queue.

        Returns (changed, pending) where *changed* is True if any tile
        GAS was added and *pending* is True if work was deferred.
        """
        # Pre-populate pyramid levels so background threads don't race
        if self._threaded:
            for lvl in range(self._max_lod + 1):
                self._get_pyramid_level(lvl)

        changed = False
        builds = 0
        pending = False
        batched = self._batched
        hf_enabled = self._hf_enabled
        tile_lods = self._tile_lods

        for _dist_sq, tr, tc, lod, gid in queue:
            # Heightfield handles LOD 0 in-bounds tiles — no mesh
            # building needed; the heightfield GAS covers them.
            if hf_enabled and lod == 0:
                in_bounds = (0 <= tr < self._n_tile_rows
                             and 0 <= tc < self._n_tile_cols)
                if in_bounds:
                    prev_lod = tile_lods.get((tr, tc), -1)
                    if prev_lod > 0 and batched:
                        self._unstage_tile(tr, tc, prev_lod)
                    self._tile_lods[(tr, tc)] = 0
                    self._active_tiles.add(gid)
                    self._hf_dirty = True
                    continue

            cache_key = (tr, tc, lod, self._base_subsample)

            # Already in the cache — upload/stage immediately (free)
            cached = self._tile_cache.get(cache_key)
            if cached is not None:
                if batched:
                    self._stage_tile(gid, tr, tc, lod, cached, ve)
                else:
                    changed |= self._upload_tile(
                        gid, tr, tc, lod, cached, ve, rtx)
                continue

            # Already building in a background thread — skip
            if cache_key in self._pending_futures:
                pending = True
                continue

            # Give up on tiles that have failed too many times
            if self._build_retries.get(cache_key, 0) >= self._MAX_BUILD_RETRIES:
                continue

            # Respect per-tick build limit
            if builds >= self.per_tick_build_limit:
                pending = True
                continue
            builds += 1

            if self._threaded and self._executor is not None:
                fut = self._executor.submit(
                    self._get_tile_mesh, tr, tc, lod)
                self._pending_futures[cache_key] = fut
                pending = True
            else:
                result = self._get_tile_mesh(tr, tc, lod)
                if result[0] is not None:
                    if batched:
                        self._stage_tile(gid, tr, tc, lod, result, ve)
                    else:
                        changed |= self._upload_tile(
                            gid, tr, tc, lod, result, ve, rtx)

        return changed, pending

    def _prefetch_streaming_tiles(self, queue, max_prefetches=8):
        """Phase C: submit I/O prefetches for out-of-bounds streaming tiles.

        Overlaps zarr reads with the current batch of mesh builds so
        tile data is ready when the build budget frees up next tick.
        At most *max_prefetches* new I/O futures are submitted per tick
        to avoid overwhelming the thread pool or network.
        """
        submitted = 0
        for _dist_sq, tr, tc, lod, _gid in queue:
            if submitted >= max_prefetches:
                break
            cache_key = (tr, tc, lod, self._base_subsample)
            if (0 <= tr < self._n_tile_rows
                    and 0 <= tc < self._n_tile_cols):
                continue  # in-bounds tiles use pyramid, not I/O
            if (cache_key in self._tile_cache
                    or cache_key in self._pending_futures
                    or cache_key in self._tile_data_cache
                    or cache_key in self._io_futures):
                continue
            self._io_futures[cache_key] = self._executor.submit(
                self._fetch_tile_data, tr, tc, lod)
            submitted += 1

    def _remove_stale_tiles(self, rtx, in_range_ids):
        """Phase D: remove tiles that left the distance range.

        Never removes tiles merely outside the frustum (avoids flicker
        when the camera rotates).  Evicts mesh cache entries to bound
        memory.

        Returns True if any tile GAS was removed.
        """
        stale_ids = self._active_tiles - in_range_ids
        if not stale_ids:
            return False

        changed = False
        batched = self._batched
        hf_enabled = self._hf_enabled

        if not batched:
            for old_id in stale_ids:
                rtx.remove_geometry(old_id)
                changed = True

        stale_keys = {k for k in self._tile_lods
                      if _tile_gid(*k) in stale_ids}
        stale_rc = {(k[0], k[1]) for k in stale_keys}
        for k in stale_keys:
            prev_lod = self._tile_lods.get(k, -1)
            if hf_enabled and prev_lod == 0:
                self._hf_dirty = True
            elif batched and prev_lod >= 0:
                self._unstage_tile(k[0], k[1], prev_lod)
            del self._tile_lods[k]
        # Evict all cached LOD variants for stale tiles (single pass)
        for cache_k in [ck for ck in self._tile_cache
                        if (ck[0], ck[1]) in stale_rc]:
            del self._tile_cache[cache_k]
        # Cancel in-flight builds for stale tiles (single pass)
        for fk in [fk for fk in self._pending_futures
                   if (fk[0], fk[1]) in stale_rc]:
            self._pending_futures.pop(fk).cancel()
        # Cancel in-flight I/O prefetches (single pass)
        for fk in [fk for fk in self._io_futures
                   if (fk[0], fk[1]) in stale_rc]:
            self._io_futures.pop(fk).cancel()
        # Evict prefetched tile data (single pass)
        for dk in [dk for dk in self._tile_data_cache
                   if (dk[0], dk[1]) in stale_rc]:
            del self._tile_data_cache[dk]
        self._active_tiles -= stale_ids
        return changed

    def remove_all(self, rtx):
        """Remove all LOD tile geometries from the scene."""
        self._cancel_pending()
        # Remove batch and heightfield GAS entries
        for gid in list(self._batch_gids):
            rtx.remove_geometry(gid)
        self._batch_gids.clear()
        self._lod_tile_meshes.clear()
        self._dirty_lods.clear()
        if self._hf_enabled:
            self._hf_dirty = False
        # Remove individual tile GAS entries (may be no-ops in batched mode)
        for tile_id in list(self._active_tiles):
            rtx.remove_geometry(tile_id)
        self._active_tiles.clear()
        self._tile_lods.clear()
        self._last_update_pos = None
        self._last_update_fwd = None

    def get_metrics(self):
        """Return structured LOD metrics for programmatic monitoring.

        Returns
        -------
        dict
            ``active_tiles``: number of tiles with an assigned LOD,
            ``cached_variants``: total mesh variants stored,
            ``in_flight_builds``: pending mesh build futures,
            ``in_flight_io``: pending I/O prefetch futures,
            ``pyramid_levels``: number of cached pyramid levels,
            ``pyramid_bytes``: approximate memory used by pyramid cache,
            ``lod_counts``: dict mapping LOD level to tile count.
        """
        from collections import Counter
        pyr_bytes = sum(arr.nbytes for arr in self._pyramid_cache.values()
                        if hasattr(arr, 'nbytes'))
        return {
            'active_tiles': len(self._tile_lods),
            'cached_variants': len(self._tile_cache),
            'in_flight_builds': len(self._pending_futures),
            'in_flight_io': len(self._io_futures),
            'pyramid_levels': len(self._pyramid_cache),
            'pyramid_bytes': pyr_bytes,
            'lod_counts': dict(Counter(self._tile_lods.values())),
        }

    def get_stats(self):
        """Return a summary string of LOD state."""
        if not self._tile_lods:
            return "LOD: no tiles"
        from collections import Counter
        counts = Counter(self._tile_lods.values())
        parts = []
        for lvl, cnt in sorted(counts.items()):
            label = "HF" if self._hf_enabled and lvl == 0 else f"L{lvl}"
            parts.append(f"{label}:{cnt}")
        active = len(self._tile_lods)
        if self._batched or self._hf_enabled:
            n_gas = len(self._batch_gids)
            suffix = f", {n_gas} GAS"
        else:
            suffix = ""
        if self._streaming:
            return f"LOD: {active} tiles ({', '.join(parts)}{suffix})"
        total = self._n_tile_rows * self._n_tile_cols
        return f"LOD: {active}/{total} tiles ({', '.join(parts)}{suffix})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_all_roughness(self):
        """Compute roughness scale factors for all in-bounds tiles.

        Uses ``compute_tile_roughness`` on full-resolution tile slices,
        then maps raw roughness to a scale factor in ``[0.5, 2.0]``
        via exponential interpolation::

            scale = 0.5 * 4^t    where t = normalized roughness in [0, 1]

        Smooth tiles get scale < 1 (LOD demoted earlier); rough tiles
        get scale > 1 (LOD promoted — finer detail kept at greater
        distance).  When all tiles have equal roughness, every tile
        gets scale 1.0 (neutral).
        """
        ts = self._tile_size
        raw = {}
        for tr in range(self._n_tile_rows):
            for tc in range(self._n_tile_cols):
                r0 = tr * ts
                c0 = tc * ts
                r1 = min(r0 + ts, self._H)
                c1 = min(c0 + ts, self._W)
                raw[(tr, tc)] = compute_tile_roughness(
                    self._terrain_np[r0:r1, c0:c1])

        if not raw:
            self._tile_roughness = {}
            return

        vals = np.array(list(raw.values()))
        r_min = float(np.min(vals))
        r_max = float(np.max(vals))

        # If the roughest tile is negligible compared to the terrain's
        # elevation range, all tiles are effectively flat — skip
        # adaptation to avoid amplifying float32 noise.
        elev_range = float(
            np.nanmax(self._terrain_np) - np.nanmin(self._terrain_np))
        roughness_floor = max(1e-4, elev_range * 1e-3)
        if r_max < roughness_floor or r_max - r_min < 1e-10:
            self._tile_roughness = {k: 1.0 for k in raw}
            return

        # Log-normalize roughness so the distribution isn't compressed
        # by extreme outliers (e.g. one sharp peak among gentle hills).
        log_vals = np.log1p(vals)
        log_min = float(np.min(log_vals))
        log_max = float(np.max(log_vals))
        log_range = log_max - log_min

        self._tile_roughness = {}
        if log_range < 1e-10:
            for k in raw:
                self._tile_roughness[k] = 1.0
            return
        for k, r in raw.items():
            t = (np.log1p(r) - log_min) / log_range  # 0..1
            # Exponential: 0.5 * 4^t  →  t=0 → 0.5, t=0.5 → 1.0, t=1 → 2.0
            self._tile_roughness[k] = 0.5 * (4.0 ** t)

    def _cancel_pending(self):
        """Cancel all in-flight background build and I/O futures."""
        for fut in self._pending_futures.values():
            fut.cancel()
        self._pending_futures.clear()
        for fut in self._io_futures.values():
            fut.cancel()
        self._io_futures.clear()
        self._build_retries.clear()

    def _upload_tile(self, gid, tr, tc, lod, mesh_data, ve, rtx):
        """Stitch, apply VE, and upload a completed tile mesh.

        Returns True if the tile was successfully added.
        """
        # own=False: rtx.add_geometry copies into GPU buffers internally
        verts, indices, norms = self._prepare_tile(
            mesh_data, tr, tc, lod, ve, own=False)
        rc = rtx.add_geometry(gid, verts, indices, normals=norms)
        if rc == 0 or rc is None:
            self._active_tiles.add(gid)
            self._tile_lods[(tr, tc)] = lod
            return True
        return False

    def _needs_stitch(self, tr, tc, lod):
        """Check whether any neighbor requires boundary stitching."""
        for nr, nc in ((tr - 1, tc), (tr + 1, tc),
                       (tr, tc - 1), (tr, tc + 1)):
            nlod = self._tile_lods.get((nr, nc), -1)
            if nlod < 0 or nlod == lod:
                continue
            if nlod > lod:
                return True
            if self._hf_enabled and nlod == 0 and lod > 0:
                return True
        return False

    def _prepare_tile(self, mesh_data, tr, tc, lod, ve, own=True):
        """Prepare cached mesh for upload: stitch boundaries and apply VE.

        Parameters
        ----------
        own : bool
            If True (default), returned arrays are guaranteed to be
            independent copies safe for long-term storage.  If False,
            cache references may be returned when no mutation is needed
            (suitable for immediate GPU upload that copies internally).
        """
        cached_verts, indices, cached_normals = mesh_data
        in_bounds = (0 <= tr < self._n_tile_rows
                     and 0 <= tc < self._n_tile_cols)
        needs_stitch = in_bounds and self._needs_stitch(tr, tc, lod)
        needs_ve = ve != 1.0

        if not needs_stitch and not needs_ve:
            if own:
                return (cached_verts.copy(), indices.copy(),
                        cached_normals.copy())
            return cached_verts, indices, cached_normals

        verts = cached_verts.copy()
        if needs_stitch:
            self._stitch_tile_boundary(verts, tr, tc, lod)
        if needs_ve:
            verts[2::3] *= ve
            norms = cached_normals.copy()
            norms[2::3] /= ve
            length = np.sqrt(norms[0::3]**2 + norms[1::3]**2 + norms[2::3]**2)
            length[length < 1e-10] = 1.0
            norms[0::3] /= length
            norms[1::3] /= length
            norms[2::3] /= length
        else:
            norms = cached_normals.copy() if own else cached_normals
        return verts, indices.copy() if own else indices, norms

    def _stage_tile(self, gid, tr, tc, lod, mesh_data, ve):
        """Stage a tile mesh for batch upload (deferred GAS build).

        The tile's stitched+VE-transformed mesh is stored per LOD level.
        The batch GAS for that level is rebuilt by :meth:`_rebuild_batches`
        at the end of the update tick.
        """
        # own=True: staged data persists across ticks, needs owned copies
        verts, indices, normals = self._prepare_tile(mesh_data, tr, tc, lod, ve)
        # If tile was at a different LOD, unstage from old level
        prev_lod = self._tile_lods.get((tr, tc), -1)
        if prev_lod >= 0 and prev_lod != lod:
            self._unstage_tile(tr, tc, prev_lod)
        # Stage into the new LOD's batch
        if lod not in self._lod_tile_meshes:
            self._lod_tile_meshes[lod] = {}
        self._lod_tile_meshes[lod][(tr, tc)] = (verts, indices, normals)
        self._dirty_lods.add(lod)
        self._active_tiles.add(gid)
        self._tile_lods[(tr, tc)] = lod

    def _tile_grid_dims(self, tr, tc, lod):
        """Compute grid dimensions (th, tw) for a tile without building it."""
        bs = self._base_subsample
        pyr_level = min(lod, self._max_lod)
        pyr_sub = bs * (2 ** pyr_level)
        pyr = self._get_pyramid_level(pyr_level)
        pH, pW = pyr.shape
        subsample = bs * (2 ** lod)
        extra = subsample // pyr_sub

        ts = self._tile_size
        r0_pyr = (tr * ts) // pyr_sub
        c0_pyr = (tc * ts) // pyr_sub
        r1_full = min((tr + 1) * ts, self._H)
        c1_full = min((tc + 1) * ts, self._W)
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub + 1, pH)
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub + 1, pW)

        th = len(range(r0_pyr, r1_pyr, extra))
        tw = len(range(c0_pyr, c1_pyr, extra))
        return th, tw

    def _stitch_tile_boundary(self, verts, tr, tc, lod):
        """Snap boundary vertex Z to match neighbors at different LODs.

        For each edge where the neighbor has a different LOD (or is a
        heightfield tile), the boundary vertices are interpolated from
        the reference pyramid level.  This closes T-junction cracks
        without needing skirt geometry.
        """
        th, tw = self._tile_grid_dims(tr, tc, lod)
        n_verts = len(verts) // 3
        if n_verts != th * tw or th < 2 or tw < 2:
            return

        neighbors = {
            'top': (tr - 1, tc),
            'bottom': (tr + 1, tc),
            'left': (tr, tc - 1),
            'right': (tr, tc + 1),
        }

        for edge, (nr, nc) in neighbors.items():
            neighbor_lod = self._tile_lods.get((nr, nc), -1)
            if neighbor_lod < 0 or neighbor_lod == lod:
                continue

            # Determine reference LOD for stitching
            if neighbor_lod > lod:
                # Neighbor is coarser — stitch to its grid
                ref_lod = neighbor_lod
            elif self._hf_enabled and neighbor_lod == 0 and lod > 0:
                # Neighbor is heightfield LOD 0 — stitch to full-res
                ref_lod = 0
            else:
                continue

            ref_z = self._get_boundary_z_ref(tr, tc, edge, ref_lod)
            if ref_z is None or len(ref_z) < 1:
                continue

            n_ref = len(ref_z)

            if edge in ('top', 'bottom'):
                n_self = tw
                row = 0 if edge == 'top' else th - 1
                edge_indices = row * tw + np.arange(tw)
            else:
                n_self = th
                col = 0 if edge == 'left' else tw - 1
                edge_indices = np.arange(th) * tw + col

            if n_self < 2:
                continue

            # Interpolate reference Z at our boundary positions
            positions = (np.arange(n_self, dtype=np.float64)
                         * (n_ref - 1) / (n_self - 1))
            z_interp = np.interp(
                positions,
                np.arange(n_ref, dtype=np.float64),
                ref_z.astype(np.float64))
            verts[edge_indices * 3 + 2] = z_interp.astype(np.float32)

    def _get_boundary_z_ref(self, tr, tc, edge, ref_lod):
        """Get Z values from a reference pyramid level at a shared boundary.

        Returns the 1-D array of elevation values that the reference
        (coarser or heightfield) tile would have along the shared edge.
        """
        pyr_level = min(ref_lod, self._max_lod)
        pyr = self._get_pyramid_level(pyr_level)
        pH, pW = pyr.shape
        pyr_sub = self._base_subsample * (2 ** pyr_level)
        subsample_ref = self._base_subsample * (2 ** ref_lod)
        extra = subsample_ref // pyr_sub

        ts = self._tile_size
        r0_full = tr * ts
        c0_full = tc * ts
        r1_full = min(r0_full + ts, self._H)
        c1_full = min(c0_full + ts, self._W)
        c0_pyr = c0_full // pyr_sub
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub + 1, pW)
        r0_pyr = r0_full // pyr_sub
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub + 1, pH)

        if edge == 'top':
            r = r0_full // pyr_sub
            if r >= pH:
                return None
            return pyr[r, c0_pyr:c1_pyr:extra].copy()
        elif edge == 'bottom':
            r = min(r1_full // pyr_sub, pH - 1)
            return pyr[r, c0_pyr:c1_pyr:extra].copy()
        elif edge == 'left':
            c = c0_full // pyr_sub
            if c >= pW:
                return None
            return pyr[r0_pyr:r1_pyr:extra, c].copy()
        elif edge == 'right':
            c = min(c1_full // pyr_sub, pW - 1)
            return pyr[r0_pyr:r1_pyr:extra, c].copy()
        return None

    def _unstage_tile(self, tr, tc, lod):
        """Remove a tile from its LOD batch and mark the level dirty."""
        tiles = self._lod_tile_meshes.get(lod)
        if tiles is not None:
            tiles.pop((tr, tc), None)
            self._dirty_lods.add(lod)

    def _rebuild_heightfield(self, rtx, ve):
        """Rebuild the heightfield GAS for LOD 0 in-bounds tiles.

        Uploads the subsampled elevation array and builds AABB custom
        primitives with an active mask derived from current LOD 0 tile
        assignments.  Only AABB tiles overlapping LOD 0 LOD-tiles get
        valid bounds; all others are degenerate (zero-volume).

        Returns True if the heightfield GAS was added or updated.
        """
        self._hf_dirty = False

        # Gather in-bounds LOD 0 tiles
        lod0_tiles = [
            (tr, tc) for (tr, tc), lod in self._tile_lods.items()
            if lod == 0
            and 0 <= tr < self._n_tile_rows
            and 0 <= tc < self._n_tile_cols
        ]

        gid = self._hf_gid
        if not lod0_tiles:
            # No LOD 0 in-bounds tiles — remove heightfield GAS
            if gid in self._batch_gids:
                rtx.remove_geometry(gid)
                self._batch_gids.discard(gid)
                return True
            return False

        # Get subsampled elevation (pyramid level 0)
        bs = self._base_subsample
        pyr = self._get_pyramid_level(0)
        pH, pW = pyr.shape

        hf_ts = self._hf_tile_size
        num_tiles_x = math.ceil((pW - 1) / hf_ts)
        num_tiles_y = math.ceil((pH - 1) / hf_ts)

        # Build active mask: AABB tiles covered by LOD 0 LOD-tiles.
        # Use 2D view + slice assignment instead of per-element Python loop.
        active_mask = np.zeros((num_tiles_y, num_tiles_x), dtype=bool)
        ts = self._tile_size
        for tr, tc in lod0_tiles:
            r0_full = tr * ts
            c0_full = tc * ts
            r1_full = min(r0_full + ts, self._H)
            c1_full = min(c0_full + ts, self._W)
            # Map to heightfield grid pixel coords
            r0_hf = r0_full // bs
            c0_hf = c0_full // bs
            r1_hf = min((r1_full + bs - 1) // bs, pH)
            c1_hf = min((c1_full + bs - 1) // bs, pW)
            # Map to AABB tile indices and mark active via slice
            ty0 = r0_hf // hf_ts
            tx0 = c0_hf // hf_ts
            ty1 = min(max(ty0, (r1_hf - 1) // hf_ts) + 1, num_tiles_y)
            tx1 = min(max(tx0, (c1_hf - 1) // hf_ts) + 1, num_tiles_x)
            active_mask[ty0:ty1, tx0:tx1] = True
        active_mask = active_mask.ravel()

        # IAS transform encodes world offset
        transform = [
            1.0, 0.0, 0.0, self._offset_x,
            0.0, 1.0, 0.0, self._offset_y,
            0.0, 0.0, 1.0, 0.0,
        ]

        rc = rtx.add_heightfield_geometry(
            gid, pyr, pH, pW,
            spacing_x=self._psx * bs,
            spacing_y=self._psy * bs,
            ve=ve,
            tile_size=hf_ts,
            active_mask=active_mask,
            transform=transform)

        if rc == 0 or rc is None:
            self._batch_gids.add(gid)
            return True
        return False

    def _rebuild_batches(self, rtx):
        """Rebuild batch GAS entries for dirty LOD levels.

        Concatenates all staged tile meshes at each dirty LOD level
        into a single vertex/index/normal buffer and uploads as one
        GAS.  Empty levels have their batch GAS removed.

        Returns True if any GAS was added or updated.
        """
        if not self._dirty_lods:
            return False
        changed = False
        for lod in self._dirty_lods:
            batch_gid = _batch_gid(lod)
            tiles = self._lod_tile_meshes.get(lod, {})
            if not tiles:
                # All tiles left this LOD — remove the batch GAS
                if batch_gid in self._batch_gids:
                    rtx.remove_geometry(batch_gid)
                    self._batch_gids.discard(batch_gid)
                    changed = True
                continue

            all_verts = []
            all_indices = []
            all_normals = []
            vert_offset = 0
            for (v, idx, n) in tiles.values():
                all_verts.append(v)
                shifted = idx + vert_offset
                all_indices.append(shifted)
                all_normals.append(n)
                vert_offset += len(v) // 3

            batch_verts = np.concatenate(all_verts)
            batch_indices = np.concatenate(all_indices)
            batch_normals = np.concatenate(all_normals)
            rc = rtx.add_geometry(
                batch_gid, batch_verts, batch_indices,
                normals=batch_normals)
            if rc == 0 or rc is None:
                self._batch_gids.add(batch_gid)
                changed = True
        self._dirty_lods.clear()
        return changed

    def _get_tile_mesh(self, tr, tc, lod):
        """Build or retrieve cached tile mesh.

        Returns references to cached arrays — caller must copy before
        mutating (e.g. for VE scaling).

        If the requested LOD produces a tile too small to triangulate
        (< 2×2 grid — common for edge tiles at high subsampling),
        falls back to progressively lower LOD levels until one works.
        """
        cache_key = (tr, tc, lod, self._base_subsample)
        cached = self._tile_cache.get(cache_key)
        if cached is not None:
            return cached

        in_bounds = (0 <= tr < self._n_tile_rows
                     and 0 <= tc < self._n_tile_cols)

        if in_bounds:
            # Try requested LOD, fall back to lower levels for small edge tiles
            for try_lod in range(lod, -1, -1):
                verts, indices, normals = self._build_tile_mesh(tr, tc, try_lod)
                if verts is not None:
                    entry = (verts, indices, normals)
                    self._tile_cache[cache_key] = entry
                    return entry
        elif self._tile_data_fn is not None:
            verts, indices, normals = self._build_streaming_tile_mesh(tr, tc, lod)
            if verts is not None:
                entry = (verts, indices, normals)
                self._tile_cache[cache_key] = entry
                return entry

        return None, None, None

    def _get_pyramid_level(self, level):
        """Return the downsampled terrain array for *level*, building lazily.

        Level 0 is the terrain at ``base_subsample`` resolution.  Each
        subsequent level halves via 2x2 NaN-aware box averaging.
        Evicts cached levels above ``_max_lod`` to bound memory.
        """
        cached = self._pyramid_cache.get(level)
        if cached is not None:
            return cached

        if level == 0:
            bs = self._base_subsample
            if bs > 1:
                arr = self._terrain_np[::bs, ::bs].copy()
            else:
                arr = self._terrain_np
        else:
            prev = self._get_pyramid_level(level - 1)
            arr = _box_downsample_2x(prev)

        self._pyramid_cache[level] = arr

        # Evict levels beyond current max_lod to bound memory
        for stale in [k for k in self._pyramid_cache
                      if k > self._max_lod]:
            del self._pyramid_cache[stale]

        return arr

    def _build_tile_mesh(self, tr, tc, lod):
        """Triangulate a single tile at the given LOD level.

        Pyramid levels are built lazily on first access (box-filter
        averaged).  This avoids pre-computing the full pyramid for
        large zarr-backed terrains.
        """
        from .. import mesh as mesh_mod

        subsample = self._base_subsample * (2 ** lod)

        # Pyramid level for this LOD (level 0 = base_subsample)
        pyr_level = min(lod, self._max_lod)
        pyr_arr = self._get_pyramid_level(pyr_level)
        pyr_sub = self._base_subsample * (2 ** pyr_level)
        pH, pW = pyr_arr.shape

        # Map full-res tile pixel coords to pyramid pixel coords.
        # Use ceil division for the end boundary so adjacent tiles
        # always share exactly one overlapping row/column, even when
        # tile_size isn't evenly divisible by pyr_sub.
        r0_full = tr * self._tile_size
        c0_full = tc * self._tile_size
        r1_full = min(r0_full + self._tile_size, self._H)
        c1_full = min(c0_full + self._tile_size, self._W)
        r0_pyr = r0_full // pyr_sub
        c0_pyr = c0_full // pyr_sub
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub + 1, pH)
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub + 1, pW)

        # If this LOD needs further subsampling beyond the pyramid level
        # (happens if lod > max precomputed level), stride the remainder.
        extra_sub = subsample // pyr_sub
        tile = pyr_arr[r0_pyr:r1_pyr:extra_sub, c0_pyr:c1_pyr:extra_sub]
        th, tw = tile.shape
        if th < 2 or tw < 2:
            return None, None, None

        # Triangulate using the fast numba/CUDA path
        num_verts = th * tw
        num_tris = (th - 1) * (tw - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        indices = np.zeros(num_tris * 3, dtype=np.int32)
        mesh_mod.triangulate_terrain(verts, indices, tile, scale=1.0)

        # Compute smooth per-vertex normals from the elevation grid
        eff_sub = pyr_sub * extra_sub
        normals = mesh_mod.compute_terrain_normals(
            tile, th, tw,
            psx=eff_sub * self._psx,
            psy=eff_sub * self._psy)

        # Transform from local grid coords to world coords.
        # Each pixel in the tile spans subsample full-res pixels.
        # The world offset keeps positions stable across terrain reloads.
        verts[0::3] = verts[0::3] * eff_sub * self._psx + c0_full * self._psx + self._offset_x
        verts[1::3] = verts[1::3] * eff_sub * self._psy + r0_full * self._psy + self._offset_y

        return verts, indices, normals

    def _fetch_tile_data(self, tr, tc, lod):
        """Fetch raw elevation data for a streaming tile (I/O only).

        Called from the thread pool to prefetch zarr reads ahead of
        mesh building.  Returns ``np.ndarray`` or ``None``.
        """
        ts = self._tile_size
        subsample = self._base_subsample * (2 ** lod)
        target_samples = max(2, ts // subsample)

        c0 = tc * ts
        r0 = tr * ts
        wx0 = c0 * self._psx + self._offset_x
        wy0 = r0 * self._psy + self._offset_y
        wx1 = (c0 + ts) * self._psx + self._offset_x
        wy1 = (r0 + ts) * self._psy + self._offset_y

        x_min, x_max = min(wx0, wx1), max(wx0, wx1)
        y_min, y_max = min(wy0, wy1), max(wy0, wy1)

        try:
            tile = self._tile_data_fn(x_min, y_min, x_max, y_max,
                                      target_samples)
        except Exception:
            return None
        if tile is None:
            return None
        return np.asarray(tile, dtype=np.float32)

    def _build_streaming_tile_mesh(self, tr, tc, lod):
        """Build a tile mesh from streamed data (outside initial terrain).

        Uses prefetched tile data from ``_tile_data_cache`` when
        available, otherwise fetches synchronously via ``_tile_data_fn``.
        """
        from .. import mesh as mesh_mod

        # Check prefetch cache first (populated by I/O futures)
        cache_key = (tr, tc, lod, self._base_subsample)
        tile = self._tile_data_cache.pop(cache_key, None)

        if tile is None:
            # No prefetched data — fetch synchronously
            tile = self._fetch_tile_data(tr, tc, lod)
            if tile is None:
                return None, None, None

        ts = self._tile_size

        # World-space bounds of this tile
        c0 = tc * ts
        r0 = tr * ts
        wx0 = c0 * self._psx + self._offset_x
        wy0 = r0 * self._psy + self._offset_y
        wx1 = (c0 + ts) * self._psx + self._offset_x
        wy1 = (r0 + ts) * self._psy + self._offset_y
        th, tw = tile.shape
        if th < 2 or tw < 2:
            return None, None, None

        # Triangulate
        num_verts = th * tw
        num_tris = (th - 1) * (tw - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        indices = np.zeros(num_tris * 3, dtype=np.int32)
        mesh_mod.triangulate_terrain(verts, indices, tile, scale=1.0)

        # Compute smooth normals using effective pixel spacing
        if tw > 1:
            eff_psx = (wx1 - wx0) / (tw - 1)
        else:
            eff_psx = 1.0
        if th > 1:
            eff_psy = (wy1 - wy0) / (th - 1)
        else:
            eff_psy = 1.0
        normals = mesh_mod.compute_terrain_normals(
            tile, th, tw, psx=eff_psx, psy=eff_psy)

        # Transform to world coordinates.
        # triangulate_terrain places vertex at (col, row, z) with scale=1.
        # Map col ∈ [0, tw-1] → [wx0, wx1], row ∈ [0, th-1] → [wy0, wy1].
        if tw > 1:
            verts[0::3] = verts[0::3] / (tw - 1) * (wx1 - wx0) + wx0
        else:
            verts[0::3] = (wx0 + wx1) * 0.5
        if th > 1:
            verts[1::3] = verts[1::3] / (th - 1) * (wy1 - wy0) + wy0
        else:
            verts[1::3] = (wy0 + wy1) * 0.5

        return verts, indices, normals


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _box_downsample_2x(arr):
    """Downsample a 2D array by 2× using NaN-aware box averaging."""
    import warnings

    H, W = arr.shape
    # Trim to even dimensions
    hh = H - H % 2
    ww = W - W % 2
    if hh < 2 or ww < 2:
        return arr.copy()
    block = arr[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2)
    # nanmean emits "Mean of empty slice" for all-NaN blocks (water);
    # the resulting NaN is correct, so suppress the warning.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = np.nanmean(block, axis=(1, 3)).astype(arr.dtype)
    return out


def _tile_gid(tr, tc):
    """Geometry ID for a terrain LOD tile."""
    return f'terrain_lod_r{tr}_c{tc}'


def _batch_gid(lod):
    """Geometry ID for a batched terrain LOD GAS at *lod* level."""
    return f'terrain_lod_batch_L{lod}'


def is_terrain_lod_gid(gid):
    """Return True if *gid* belongs to a terrain LOD tile or batch."""
    return gid.startswith('terrain_lod_')


