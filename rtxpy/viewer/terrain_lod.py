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

from ..lod import (box_downsample_2x as _box_downsample_2x,
                    compute_lod_distances,
                    compute_lod_level_with_hysteresis,
                    compute_tile_roughness)


class TerrainLODManager:
    """Manages per-tile terrain LOD in the interactive viewer.

    Parameters
    ----------
    terrain_np : np.ndarray
        Full-resolution elevation array, shape ``(H, W)``.
        Ignored when *chunk_source* is provided.
    tile_size : int
        Tile edge length in full-resolution pixels (default 128).
        Ignored when *chunk_source* is provided (chunk shape is used).
    pixel_spacing_x, pixel_spacing_y : float
        World-space size of one raster pixel.
        Ignored when *chunk_source* is provided.
    max_lod : int
        Maximum LOD level.  Level *k* subsamples the tile by ``2^k``.
    lod_distance_factor : float
        Controls how far each LOD band extends.  First transition
        at ``tile_diagonal * factor``, doubling per level.
    base_subsample : int
        Global base subsample factor (from R/Shift+R).  LOD 0 uses
        this value; LOD *k* uses ``base_subsample * 2^k``.
    chunk_source : ChunkDataSource or None
        When provided, terrain data is read from this source instead
        of *terrain_np*.  The source's chunk grid defines the tile
        grid, and its pixel spacing overrides *pixel_spacing_x/y*.
    """

    __slots__ = (
        '_terrain_np', '_chunk_source', '_tile_size', '_psx', '_psy',
        '_max_lod', '_lod_distances', '_lod_distance_factor',
        '_H', '_W', '_n_tile_rows', '_n_tile_cols',
        '_tile_centers', '_tile_lods', '_tile_cache',
        '_active_tiles', '_last_update_pos', '_last_update_fwd',
        '_update_threshold_sq',
        '_base_subsample', '_pyramid_cache', '_has_in_flight_work',
        '_tile_half_diag_sq', '_tile_half_diag',
        '_max_dist_sq', '_max_dist', '_streaming_max_dist',
        '_tile_world_x', '_tile_world_y',
        '_offset_x', '_offset_y',
        'max_tiles', 'per_tick_build_limit',
        '_tile_roughness',
        '_tile_data_fn', '_streaming', '_crs_origin', '_crs_spacing',
        '_tile_data_cache', '_io_futures',
        '_executor', '_pending_futures', '_threaded',
        '_batched', '_lod_tile_meshes', '_dirty_lods', '_batch_gids',
        '_batch_cache',
        '_hf_enabled', '_hf_gid', '_hf_dirty', '_hf_tile_size',
        '_hf_last_lod0_set', '_hf_last_ve', '_hf_cached_pyr_id',
        '_build_retries',
        '_on_tile_added', '_on_tile_removed',
        '_scene_mesh_mgr',
        '_lod_changed_this_tick',
    )

    def __init__(self, terrain_np=None, tile_size=128,
                 pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 max_lod=3, lod_distance_factor=3.0,
                 base_subsample=1, chunk_source=None):

        # --- Resolve data source ---
        self._chunk_source = chunk_source

        if chunk_source is not None:
            # Chunk source drives grid geometry
            H, W = chunk_source.shape
            ch, cw = chunk_source.chunk_shape
            tile_size = ch  # one chunk = one tile
            psx, psy = chunk_source.pixel_spacing
            pixel_spacing_x = psx
            pixel_spacing_y = psy
            # Keep terrain_np as None — all reads go through chunk_source
            self._terrain_np = None
        elif terrain_np is not None:
            H, W = terrain_np.shape
            self._terrain_np = terrain_np
        else:
            raise ValueError(
                "Either terrain_np or chunk_source must be provided")

        self._tile_size = tile_size
        self._psx = pixel_spacing_x
        self._psy = pixel_spacing_y
        self._max_lod = max_lod
        self._lod_distance_factor = lod_distance_factor
        self._base_subsample = base_subsample

        self._H = H
        self._W = W
        self._n_tile_rows = (H + tile_size - 1) // tile_size
        self._n_tile_cols = (W + tile_size - 1) // tile_size

        # World-space offset applied to tile vertices (for stable camera
        # across terrain reloads).  Must be set before _recompute_tile_centers.
        self._offset_x = 0.0
        self._offset_y = 0.0

        # Streaming: when chunk_source.supports_streaming is True, OR when
        # set_tile_data_fn() provides a callback, tiles outside the initial
        # grid are fetched on demand.
        self._tile_data_fn = None
        self._streaming = (chunk_source is not None
                           and chunk_source.supports_streaming)
        self._tile_data_cache = {}  # cache_key -> np.ndarray (prefetched I/O)
        self._io_futures = {}       # cache_key -> Future (in-flight prefetches)
        # CRS coordinate origin and signed spacing — used to convert pixel
        # indices to CRS coordinates for tile_data_fn calls.  When a
        # chunk_source provides crs_transform, it is used automatically.
        crs = chunk_source.crs_transform if chunk_source is not None else None
        if crs is not None:
            self._crs_origin = (crs[0], crs[1])
            self._crs_spacing = (crs[2], crs[3])
        else:
            self._crs_origin = None   # (crs_x0, crs_y0) or None
            self._crs_spacing = None  # (crs_dx, crs_dy) signed, or None

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
        self._streaming_max_dist = (self._lod_distances[-1] * 1.5
                                    if self._lod_distances else max_dist)
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
        self._lod_changed_this_tick = set()  # (tr, tc) tiles that changed LOD

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
        self._batch_cache = {}      # {lod: (verts, indices, normals)} cached concat

        # Heightfield for LOD 0: in-bounds LOD 0 tiles use heightfield
        # ray marching instead of explicit triangles.  Disabled by default;
        # engine.py enables it after construction.
        self._hf_enabled = False
        self._hf_gid = 'terrain_lod_hf'
        self._hf_dirty = False
        self._hf_tile_size = 32
        self._hf_last_lod0_set = frozenset()
        self._hf_last_ve = None
        self._hf_cached_pyr_id = None  # id() of pyramid array for skip

        # Tile lifecycle callbacks — used by OverlayTileManager to
        # keep per-tile overlay data in sync with the tile set.
        self._on_tile_added = None    # fn(tr, tc, elevation_tile)
        self._on_tile_removed = None  # fn(tr, tc)

        # Scene mesh manager: placed geometry loaded from scene zarr,
        # driven by the same tile visibility as terrain LOD.
        self._scene_mesh_mgr = None  # set via set_scene_zarr()

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

    def set_tile_callbacks(self, on_added=None, on_removed=None):
        """Register callbacks for tile lifecycle events.

        Parameters
        ----------
        on_added : callable(tr, tc, elevation_tile) or None
            Called when a tile is successfully added to the scene.
            *elevation_tile* is the tile's elevation data from the
            pyramid cache or streaming fetch (may be None for HF tiles).
        on_removed : callable(tr, tc) or None
            Called when a tile is evicted from the scene.
        """
        self._on_tile_added = on_added
        self._on_tile_removed = on_removed

    def set_scene_zarr(self, zarr_path):
        """Attach a scene zarr store for placed geometry loading.

        When set, placed geometry (buildings, roads, etc.) from the
        zarr store is loaded and unloaded in sync with terrain tile
        visibility.  The LOD manager's per-tile LOD assignments drive
        mesh simplification.

        Parameters
        ----------
        zarr_path : str or None
            Path to the scene zarr store, or ``None`` to detach.
        """
        if zarr_path is None:
            self._scene_mesh_mgr = None
            return
        from .scene_meshes import SceneMeshManager
        self._scene_mesh_mgr = SceneMeshManager(zarr_path)

    @property
    def scene_mesh_manager(self):
        """The :class:`SceneMeshManager` if a scene zarr is attached."""
        return self._scene_mesh_mgr

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
        self._batch_cache.clear()
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
            if self._chunk_source is not None:
                self._chunk_source.clear_caches()
            self._tile_cache.clear()
            self._tile_data_cache.clear()
            self._lod_tile_meshes.clear()
            self._batch_cache.clear()
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
            self._batch_cache.clear()
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

        if self._streaming:
            # Use floor division so partial edge tiles become out-of-bounds
            # and are handled by streaming (which fetches from zarr without
            # NaN reprojection artifacts).  Ceiling division leaves a gap
            # between the last in-bounds partial tile and the first streaming
            # tile when terrain dims aren't divisible by tile_size.
            ts = self._tile_size
            new_rows = self._H // ts
            new_cols = self._W // ts
            if new_rows != self._n_tile_rows or new_cols != self._n_tile_cols:
                self._n_tile_rows = new_rows
                self._n_tile_cols = new_cols
                self._recompute_tile_centers()
                self._compute_all_roughness()

    def set_crs_transform(self, origin_x, origin_y, dx, dy):
        """Set CRS coordinate origin and signed pixel spacing.

        Required for streaming tiles: converts pixel indices to actual
        CRS coordinates (e.g. UTM easting/northing) before calling
        ``tile_data_fn``.  Without this, the callback receives viewer
        world-space coordinates which are not valid CRS values.

        Parameters
        ----------
        origin_x, origin_y : float
            CRS coordinate of the first pixel (row 0, col 0).
        dx, dy : float
            Signed CRS spacing per pixel.  For UTM rasters, ``dx > 0``
            (easting increases with column) and ``dy < 0`` (northing
            decreases with row, since row 0 = north).
        """
        self._crs_origin = (origin_x, origin_y)
        self._crs_spacing = (dx, dy)

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

    def set_terrain(self, terrain_np=None, offset_x=None, offset_y=None,
                     chunk_source=None):
        """Replace the terrain data (e.g. after dynamic reload).

        Parameters
        ----------
        terrain_np : np.ndarray or None
            New elevation array.  Ignored when *chunk_source* is given.
        offset_x, offset_y : float or None
            World-space offset to apply.
        chunk_source : ChunkDataSource or None
            New chunk source.  When given, replaces the existing source
            and *terrain_np* is ignored.
        """
        if chunk_source is not None:
            self._chunk_source = chunk_source
            self._terrain_np = None
            H, W = chunk_source.shape
            ts = chunk_source.chunk_shape[0]
            self._tile_size = ts
            psx, psy = chunk_source.pixel_spacing
            self._psx = psx
            self._psy = psy
        elif terrain_np is not None:
            self._terrain_np = terrain_np
            if self._chunk_source is not None:
                self._chunk_source.set_terrain(terrain_np)
            H, W = terrain_np.shape
            ts = self._tile_size
        else:
            return  # nothing to do

        self._H = H
        self._W = W
        if self._streaming:
            self._n_tile_rows = H // ts
            self._n_tile_cols = W // ts
        else:
            self._n_tile_rows = (H + ts - 1) // ts
            self._n_tile_cols = (W + ts - 1) // ts
        if offset_x is not None:
            self._offset_x = offset_x
        if offset_y is not None:
            self._offset_y = offset_y
        self._recompute_tile_centers()
        self._compute_all_roughness()
        self._cancel_pending()
        self._pyramid_cache.clear()
        if self._chunk_source is not None:
            self._chunk_source.clear_caches()
        self._tile_cache.clear()
        self._tile_data_cache.clear()
        self._lod_tile_meshes.clear()
        self._batch_cache.clear()
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
        # Use a tighter radius for streaming to keep tile count
        # manageable.  Beyond the last LOD threshold all tiles are at
        # max LOD — adding more doesn't improve quality but each one
        # grows the batch GAS that must be rebuilt on LOD changes.
        if self._streaming:
            max_dist = self._streaming_max_dist
            max_dist_sq = max_dist * max_dist
        else:
            max_dist_sq = self._max_dist_sq
            max_dist = self._max_dist

        # Pass 1: find all tiles within distance range, and flag which
        # are also inside the view frustum (candidates for building).
        # Uses squared distances to avoid sqrt per tile.
        in_frustum = []    # tiles that passed distance + frustum culling
        in_range_ids = set()  # all distance-valid tiles (frustum-independent)
        in_range_rc = set()   # (tr, tc) pairs for all in-range tiles

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
            r_tc = int(max_dist / self._tile_world_x) + 1
            r_tr = int(max_dist / self._tile_world_y) + 1
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
            in_range_rc.add((tr, tc))

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
        self._lod_changed_this_tick = set()

        # --- Phase A: collect completed async builds ---
        changed |= self._collect_completed_builds(rtx, ve)

        # --- Phase B: process tile queue (build/upload/stage) ---
        b_changed, pending = self._process_tile_queue(
            unbuilt + lod_updates, rtx, ve)
        changed |= b_changed

        # --- Phase B2: re-stitch neighbors of tiles that changed LOD ---
        # When a tile changes LOD, its neighbors' boundary stitching
        # becomes stale (they were stitched against the old LOD).
        # Re-stage those neighbors so their stitching uses the updated
        # tile_lods.
        if self._lod_changed_this_tick:
            changed |= self._restitch_neighbors(rtx, ve)

        # --- Phase C: prefetch I/O for streaming tiles ---
        if (self._streaming and self._threaded
                and self._executor is not None):
            self._prefetch_streaming_tiles(unbuilt + lod_updates)

        # --- Phase D: remove stale tiles ---
        changed |= self._remove_stale_tiles(rtx, in_range_ids, in_range_rc)

        # --- Phase E: rebuild heightfield and batch GAS ---
        if self._hf_enabled and self._hf_dirty:
            changed |= self._rebuild_heightfield(rtx, ve)
        if self._batched:
            changed |= self._rebuild_batches(rtx)

        # --- Phase F: sync placed geometry visibility ---
        if self._scene_mesh_mgr is not None:
            # Build chunk distances from tile centers for load ordering
            chunk_dists = {}
            for (tr, tc), _lod in self._tile_lods.items():
                cx, cy = self._tile_center(tr, tc)
                dx_t = cx - cam_x
                dy_t = cy - cam_y
                chunk_dists[(tr, tc)] = math.sqrt(
                    dx_t * dx_t + dy_t * dy_t)
            self._scene_mesh_mgr.update_visibility(
                self._tile_lods, chunk_distances=chunk_dists)

        self._has_in_flight_work = (pending or bool(self._pending_futures)
                             or bool(self._io_futures)
                             or bool(self._dirty_lods))
        return changed

    # ------------------------------------------------------------------
    # update() helper phases
    # ------------------------------------------------------------------

    _MAX_BUILD_RETRIES = 3

    def _collect_completed_builds(self, rtx, ve):
        """Phase A: harvest completed async mesh builds and I/O prefetches.

        Freshly completed meshes are uploaded/staged immediately so they
        don't depend on re-passing the frustum check (the build was
        already approved on a prior tick).
        """
        changed = False
        if self._pending_futures:
            batched = self._batched
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
                    # Upload/stage immediately — don't wait for frustum re-check
                    tr, tc, lod = cache_key[0], cache_key[1], cache_key[2]
                    gid = _tile_gid(tr, tc)
                    if batched:
                        self._stage_tile(gid, tr, tc, lod, result, ve)
                    else:
                        changed |= self._upload_tile(
                            gid, tr, tc, lod, result, ve, rtx)
                else:
                    # Cache empty result so this tile is not rebuilt
                    self._tile_cache[cache_key] = (None, None, None)

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
        return changed

    def _process_tile_queue(self, queue, rtx, ve):
        """Phase B: build, upload, or stage tiles from the priority queue.

        Returns (changed, pending) where *changed* is True if any tile
        GAS was added and *pending* is True if work was deferred.
        """
        # Pre-populate pyramid levels so background threads don't race.
        # Skip when using chunk_source — reads go through the source,
        # not the global pyramid.
        if self._threaded and self._chunk_source is None:
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
                    if prev_lod != 0:
                        self._lod_changed_this_tick.add((tr, tc))
                    was_new = (tr, tc) not in self._tile_lods
                    self._tile_lods[(tr, tc)] = 0
                    self._active_tiles.add(gid)
                    self._hf_dirty = True
                    if was_new and self._on_tile_added is not None:
                        self._fire_tile_added(tr, tc)
                    continue

            cache_key = (tr, tc, lod, self._base_subsample)

            # Already in the cache — upload/stage immediately (free)
            cached = self._tile_cache.get(cache_key)
            if cached is not None:
                if cached[0] is None:
                    continue  # empty tile (no data available)
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

    def _restitch_neighbors(self, rtx, ve):
        """Phase B2: re-stitch neighbors of tiles that changed LOD.

        When a tile transitions between LOD levels, its neighbors'
        boundary stitching becomes stale — they were stitched against the
        old LOD.  This pass re-stages (batched) or re-uploads
        (non-batched) those neighbors so stitching uses the current
        ``_tile_lods``.
        """
        changed = False
        batched = self._batched
        already = self._lod_changed_this_tick  # skip tiles that were just built

        to_restitch = set()
        for (tr, tc) in already:
            for nr, nc in ((tr - 1, tc), (tr + 1, tc),
                           (tr, tc - 1), (tr, tc + 1)):
                if (nr, nc) in already:
                    continue  # already rebuilt this tick
                nlod = self._tile_lods.get((nr, nc), -1)
                if nlod < 0:
                    continue  # not an active tile
                gid = _tile_gid(nr, nc)
                if gid not in self._active_tiles:
                    continue
                to_restitch.add((nr, nc, nlod, gid))

        for nr, nc, nlod, gid in to_restitch:
            cache_key = (nr, nc, nlod, self._base_subsample)
            cached = self._tile_cache.get(cache_key)
            if cached is None or cached[0] is None:
                continue
            if batched:
                self._stage_tile(gid, nr, nc, nlod, cached, ve)
            else:
                changed |= self._upload_tile(
                    gid, nr, nc, nlod, cached, ve, rtx)

        return changed

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

    def _remove_stale_tiles(self, rtx, in_range_ids, in_range_rc):
        """Phase D: remove tiles that left the distance range.

        Never removes tiles merely outside the frustum (avoids flicker
        when the camera rotates).  Evicts mesh cache entries to bound
        memory, including orphaned None-cached entries from streaming
        tiles that were never activated.

        Returns True if any tile GAS was removed.
        """
        stale_ids = self._active_tiles - in_range_ids

        changed = False
        batched = self._batched
        hf_enabled = self._hf_enabled

        if stale_ids:
            # Remove individual GAS entries for stale tiles.  In batched
            # mode, tiles from the pre-batching initial build still have
            # individual GAS entries that must be cleaned up.
            for old_id in stale_ids:
                if rtx.has_geometry(old_id):
                    rtx.remove_geometry(old_id)
                    changed = True

            stale_keys = {k for k in self._tile_lods
                          if _tile_gid(*k) in stale_ids}
            stale_rc = {(k[0], k[1]) for k in stale_keys}
            on_removed = self._on_tile_removed
            for k in stale_keys:
                prev_lod = self._tile_lods.get(k, -1)
                if hf_enabled and prev_lod == 0:
                    self._hf_dirty = True
                elif batched and prev_lod >= 0:
                    self._unstage_tile(k[0], k[1], prev_lod)
                del self._tile_lods[k]
                if on_removed is not None:
                    on_removed(k[0], k[1])
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
            # Evict stale build retry entries
            for brk in [brk for brk in self._build_retries
                        if (brk[0], brk[1]) in stale_rc]:
                del self._build_retries[brk]
            self._active_tiles -= stale_ids

        # Evict orphaned cache entries: tiles cached (including
        # None-cached streaming tiles outside DEM extent) but not in
        # distance range and not pending build.  Without this, None
        # entries from tiles the camera flew past accumulate forever.
        if self._streaming and self._tile_cache:
            pending_rc = {(k[0], k[1]) for k in self._pending_futures}
            orphan_keys = [ck for ck in self._tile_cache
                           if (ck[0], ck[1]) not in in_range_rc
                           and (ck[0], ck[1]) not in pending_rc]
            for ck in orphan_keys:
                del self._tile_cache[ck]

        return changed

    def remove_all(self, rtx):
        """Remove all LOD tile geometries from the scene."""
        self._cancel_pending()
        # Remove batch and heightfield GAS entries
        for gid in list(self._batch_gids):
            rtx.remove_geometry(gid)
        self._batch_gids.clear()
        self._lod_tile_meshes.clear()
        self._batch_cache.clear()
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
            staged = sum(len(t) for t in self._lod_tile_meshes.values())
            total_verts = sum(len(v) // 3
                              for tiles in self._lod_tile_meshes.values()
                              for v, _, _ in tiles.values())
            suffix = f", {n_gas} GAS, {staged}stg/{total_verts//1000}Kv"
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
        cs = self._chunk_source
        raw = {}
        for tr in range(self._n_tile_rows):
            for tc in range(self._n_tile_cols):
                if cs is not None:
                    raw[(tr, tc)] = cs.chunk_roughness(tr, tc)
                else:
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
        if cs is not None:
            stats = cs.elevation_stats()
            if stats is not None:
                elev_range = stats[1] - stats[0]
            else:
                elev_range = r_max  # fallback
        else:
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
        prev_lod = self._tile_lods.get((tr, tc), -1)
        # own=False: rtx.add_geometry copies into GPU buffers internally
        verts, indices, norms = self._prepare_tile(
            mesh_data, tr, tc, lod, ve, own=False)
        rc = rtx.add_geometry(gid, verts, indices, normals=norms)
        if rc == 0 or rc is None:
            self._active_tiles.add(gid)
            if prev_lod != lod:
                self._lod_changed_this_tick.add((tr, tc))
            self._tile_lods[(tr, tc)] = lod
            if self._on_tile_added is not None:
                self._fire_tile_added(tr, tc)
            return True
        return False

    def _fire_tile_added(self, tr, tc):
        """Invoke the on_tile_added callback with this tile's elevation."""
        ts = self._tile_size
        in_bounds = (0 <= tr < self._n_tile_rows
                     and 0 <= tc < self._n_tile_cols)
        cs = self._chunk_source
        if in_bounds and cs is not None:
            elev = cs.read_full_res_chunk(tr, tc)
        elif in_bounds and self._terrain_np is not None:
            r0, c0 = tr * ts, tc * ts
            r1 = min(r0 + ts, self._H)
            c1 = min(c0 + ts, self._W)
            elev = self._terrain_np[r0:r1, c0:c1]
        else:
            # Streaming tile — check prefetch cache
            cache_key = (tr, tc)
            elev = self._tile_data_cache.get(cache_key)
        try:
            self._on_tile_added(tr, tc, elev)
        except Exception:
            logger.debug("on_tile_added callback failed for (%d, %d)",
                         tr, tc, exc_info=True)

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
        needs_stitch = self._needs_stitch(tr, tc, lod)
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
        # Track any tile whose LOD state changed (including first build)
        # so neighbors can be re-stitched.
        if prev_lod != lod:
            self._lod_changed_this_tick.add((tr, tc))
        # Stage into the new LOD's batch
        if lod not in self._lod_tile_meshes:
            self._lod_tile_meshes[lod] = {}
        self._lod_tile_meshes[lod][(tr, tc)] = (verts, indices, normals)
        self._dirty_lods.add(lod)
        self._active_tiles.add(gid)
        self._tile_lods[(tr, tc)] = lod
        if self._on_tile_added is not None:
            self._fire_tile_added(tr, tc)

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
        r_overlap = 1 if r1_full < self._H else 0
        c_overlap = 1 if c1_full < self._W else 0
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub + r_overlap, pH)
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub + c_overlap, pW)

        th = len(range(r0_pyr, r1_pyr, extra))
        tw = len(range(c0_pyr, c1_pyr, extra))
        return th, tw

    def _stitch_tile_boundary(self, verts, tr, tc, lod):
        """Snap boundary vertex Z to match neighbors at different LODs.

        For each edge where the neighbor has a different LOD (or is a
        heightfield tile), the boundary vertices are interpolated from
        the reference pyramid level or the neighbor's cached mesh.
        Works for both in-bounds and streaming tiles.
        """
        in_bounds = (0 <= tr < self._n_tile_rows
                     and 0 <= tc < self._n_tile_cols)
        n_verts = len(verts) // 3
        if n_verts < 4:
            return
        if in_bounds:
            th, tw = self._tile_grid_dims(tr, tc, lod)
            if n_verts != th * tw or th < 2 or tw < 2:
                return
        else:
            # Streaming tile: infer grid dims from vertex count.
            # Streaming meshes are regular grids, so th * tw == n_verts.
            # Count vertices sharing the first row's Y coordinate.
            y_coords = verts[1::3]
            y0 = y_coords[0]
            tol = abs(self._psy) * 0.1
            tw = int(np.sum(np.abs(y_coords - y0) < max(tol, 1e-6)))
            if tw < 2:
                return
            th = n_verts // tw
            if th < 2 or th * tw != n_verts:
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
        """Get Z values from a reference level at a shared tile boundary.

        Returns the 1-D array of elevation values that the reference
        (coarser or heightfield) tile would have along the shared edge.
        Falls back to extracting Z from the neighbor's cached mesh
        when pyramid data is not available (streaming tiles).
        """
        tile_in_bounds = (0 <= tr < self._n_tile_rows
                          and 0 <= tc < self._n_tile_cols)

        # Pyramid path works when the tile itself is in-bounds
        if tile_in_bounds:
            return self._get_boundary_z_ref_pyramid(tr, tc, edge, ref_lod)

        # Tile is streaming (out of bounds) — determine the neighbor
        # and try to get boundary Z from IT
        if edge == 'top':
            nr, nc = tr - 1, tc
        elif edge == 'bottom':
            nr, nc = tr + 1, tc
        elif edge == 'left':
            nr, nc = tr, tc - 1
        else:
            nr, nc = tr, tc + 1

        neighbor_in_bounds = (0 <= nr < self._n_tile_rows
                              and 0 <= nc < self._n_tile_cols)

        if neighbor_in_bounds:
            # Neighbor is in-bounds — use pyramid via neighbor's coords
            opposite = {'top': 'bottom', 'bottom': 'top',
                        'left': 'right', 'right': 'left'}
            return self._get_boundary_z_ref_pyramid(
                nr, nc, opposite[edge], ref_lod)

        # Both tiles are streaming — extract from neighbor's cached mesh
        return self._get_boundary_z_from_cache(nr, nc, edge, ref_lod)

    def _get_boundary_z_from_cache(self, nr, nc, edge, ref_lod):
        """Extract boundary Z values from a neighbor's cached mesh.

        Used for streaming tiles that have no pyramid data.  Reads the
        un-VE'd mesh from ``_tile_cache`` and returns Z at the shared
        edge.

        Parameters
        ----------
        nr, nc : int
            Neighbor tile row/column.
        edge : str
            Edge from the perspective of the tile being stitched
            ('top'/'bottom'/'left'/'right').  The neighbor's shared
            edge is the opposite.
        ref_lod : int
            LOD of the neighbor tile.
        """
        cache_key = (nr, nc, ref_lod, self._base_subsample)
        cached = self._tile_cache.get(cache_key)
        if cached is None or cached[0] is None:
            return None

        n_verts_cached, _, _ = cached
        nverts = len(n_verts_cached) // 3
        if nverts < 4:
            return None

        # Infer grid dims from cached mesh
        y_coords = n_verts_cached[1::3]
        y0 = y_coords[0]
        tol = abs(self._psy) * 0.1
        n_tw = int(np.sum(np.abs(y_coords - y0) < max(tol, 1e-6)))
        if n_tw < 2:
            return None
        n_th = nverts // n_tw
        if n_th < 2 or n_th * n_tw != nverts:
            return None

        z_vals = n_verts_cached[2::3]

        # Map caller's edge to neighbor's opposite edge
        opposite = {'top': 'bottom', 'bottom': 'top',
                    'left': 'right', 'right': 'left'}
        neighbor_edge = opposite[edge]

        if neighbor_edge == 'top':
            return z_vals[:n_tw].copy()
        elif neighbor_edge == 'bottom':
            return z_vals[(n_th - 1) * n_tw:n_th * n_tw].copy()
        elif neighbor_edge == 'left':
            return z_vals[::n_tw].copy()
        elif neighbor_edge == 'right':
            return z_vals[n_tw - 1::n_tw].copy()
        return None

    def _get_boundary_z_ref_pyramid(self, tr, tc, edge, ref_lod):
        """Get Z values from a pyramid level at a shared boundary.

        Original pyramid-based implementation for in-bounds tiles.
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
        r_overlap = 1 if r1_full < self._H else 0
        c_overlap = 1 if c1_full < self._W else 0
        c0_pyr = c0_full // pyr_sub
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub + c_overlap, pW)
        r0_pyr = r0_full // pyr_sub
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub + r_overlap, pH)

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

        # Skip rebuild if the set of LOD 0 tiles and VE haven't changed
        # (avoids re-uploading the full elevation array + AABB rebuild
        # when only higher-LOD tiles transitioned).
        lod0_set = frozenset(lod0_tiles)
        if (lod0_set == self._hf_last_lod0_set
                and ve == getattr(self, '_hf_last_ve', None)
                and gid in self._batch_gids):
            return False
        self._hf_last_lod0_set = lod0_set
        self._hf_last_ve = ve

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

        To avoid frame spikes when multiple LODs are dirty (e.g. on
        camera rotation), at most one non-empty LOD batch is rebuilt
        per tick.  Empty-LOD removals are always processed immediately
        since they're cheap.

        Uses a cached concatenation buffer per LOD level.  When tiles
        are added/removed, the buffer is reallocated; when tile data
        changes (e.g. LOD swap), only the affected slice is overwritten.

        Returns True if any GAS was added or updated.
        """
        if not self._dirty_lods:
            return False
        changed = False
        rebuilt_one = False
        remaining = set()
        for lod in sorted(self._dirty_lods):
            batch_gid = _batch_gid(lod)
            tiles = self._lod_tile_meshes.get(lod, {})
            if not tiles:
                # All tiles left this LOD — remove the batch GAS (cheap)
                if batch_gid in self._batch_gids:
                    rtx.remove_geometry(batch_gid)
                    self._batch_gids.discard(batch_gid)
                    self._batch_cache.pop(lod, None)
                    changed = True
                continue

            # Only rebuild one non-empty batch per tick to cap frame time
            if rebuilt_one:
                remaining.add(lod)
                continue
            rebuilt_one = True

            # Compute total sizes
            total_verts = 0
            total_indices = 0
            for (v, idx, n) in tiles.values():
                total_verts += len(v)
                total_indices += len(idx)

            # Reuse cached buffer if sizes match (common case: tile
            # swapped between LOD levels but total count is stable)
            cached = self._batch_cache.get(lod)
            if (cached is not None
                    and len(cached[0]) == total_verts
                    and len(cached[1]) == total_indices):
                batch_verts, batch_indices, batch_normals = cached
            else:
                batch_verts = np.empty(total_verts, dtype=np.float32)
                batch_indices = np.empty(total_indices, dtype=np.int32)
                batch_normals = np.empty(total_verts, dtype=np.float32)

            # Fill buffers (memcpy per tile)
            v_pos = 0
            i_pos = 0
            vert_offset = 0
            for (v, idx, n) in tiles.values():
                vlen = len(v)
                ilen = len(idx)
                batch_verts[v_pos:v_pos + vlen] = v
                batch_normals[v_pos:v_pos + vlen] = n
                batch_indices[i_pos:i_pos + ilen] = idx
                if vert_offset > 0:
                    batch_indices[i_pos:i_pos + ilen] += vert_offset
                v_pos += vlen
                i_pos += ilen
                vert_offset += vlen // 3

            self._batch_cache[lod] = (batch_verts, batch_indices,
                                      batch_normals)

            rc = rtx.add_geometry(
                batch_gid, batch_verts, batch_indices,
                normals=batch_normals)
            if rc == 0 or rc is None:
                self._batch_gids.add(batch_gid)
                changed = True
        self._dirty_lods = remaining
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
        cs = self._chunk_source

        if in_bounds:
            if cs is not None:
                # Chunk source path — unified for in-bounds tiles
                verts, indices, normals = self._build_tile_mesh_from_source(
                    tr, tc, lod)
                if verts is not None:
                    entry = (verts, indices, normals)
                    self._tile_cache[cache_key] = entry
                    return entry
            else:
                # Legacy path — try requested LOD, fall back for small tiles
                for try_lod in range(lod, -1, -1):
                    verts, indices, normals = self._build_tile_mesh(
                        tr, tc, try_lod)
                    if verts is not None:
                        entry = (verts, indices, normals)
                        self._tile_cache[cache_key] = entry
                        return entry
        elif cs is not None and cs.can_stream(tr, tc):
            # Chunk source streaming path
            verts, indices, normals = self._build_tile_mesh_from_source(
                tr, tc, lod)
            if verts is not None:
                entry = (verts, indices, normals)
                self._tile_cache[cache_key] = entry
                return entry
        elif self._tile_data_fn is not None:
            # Legacy streaming path
            verts, indices, normals = self._build_streaming_tile_mesh(
                tr, tc, lod)
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

        When a chunk_source is active, this method is only used for
        heightfield and boundary stitching — it assembles the full
        pyramid from per-chunk reads.  For tile mesh building, use
        ``_build_tile_mesh_from_source`` which reads individual chunks.
        """
        cached = self._pyramid_cache.get(level)
        if cached is not None:
            return cached

        if self._terrain_np is not None:
            if level == 0:
                bs = self._base_subsample
                if bs > 1:
                    arr = self._terrain_np[::bs, ::bs].copy()
                else:
                    arr = self._terrain_np
            else:
                prev = self._get_pyramid_level(level - 1)
                arr = _box_downsample_2x(prev)
        elif self._chunk_source is not None:
            # Build from chunk source: assemble full array at this LOD
            # by reading each in-bounds chunk and pasting into a canvas.
            cs = self._chunk_source
            bs = self._base_subsample
            canvas_parts = []
            for cr in range(cs.n_chunk_rows):
                row_parts = []
                for cc in range(cs.n_chunk_cols):
                    tile = cs.read_chunk(cr, cc, lod=level,
                                         base_subsample=bs)
                    if tile is None:
                        # Estimate expected size from first available tile
                        ch, cw = cs.chunk_shape
                        sub = bs * (2 ** level)
                        eh = max(1, ch // sub)
                        ew = max(1, cw // sub)
                        tile = np.full((eh, ew), np.nan, dtype=np.float32)
                    row_parts.append(tile)
                if row_parts:
                    canvas_parts.append(np.concatenate(row_parts, axis=1))
            if canvas_parts:
                arr = np.concatenate(canvas_parts, axis=0)
            else:
                arr = np.zeros((1, 1), dtype=np.float32)
        else:
            raise RuntimeError("No terrain data available for pyramid")

        # Fill NaN globally BEFORE caching so every tile that slices from
        # this pyramid level sees the same value at shared boundary pixels.
        # Without this, per-tile nanmean fill produces different Z at
        # boundaries → cracks visible under vertical exaggeration.
        # Also makes stitching reference Z non-NaN.
        bad = ~np.isfinite(arr)
        if bad.any():
            fill_val = np.nanmean(arr)
            if not np.isfinite(fill_val):
                fill_val = 0.0
            arr = arr.copy()
            arr[bad] = fill_val

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
        # Add +1 overlap for shared boundary vertices with the next tile,
        # but NOT for the last row/column of tiles (no neighbor to share
        # with, and the extra row would extend into NaN territory causing
        # elevation spikes from NaN→mean fill).
        r_overlap = 1 if r1_full < self._H else 0
        c_overlap = 1 if c1_full < self._W else 0
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub + r_overlap, pH)
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub + c_overlap, pW)

        # If this LOD needs further subsampling beyond the pyramid level
        # (happens if lod > max precomputed level), stride the remainder.
        extra_sub = subsample // pyr_sub
        tile = pyr_arr[r0_pyr:r1_pyr:extra_sub, c0_pyr:c1_pyr:extra_sub]
        th, tw = tile.shape
        if th < 2 or tw < 2:
            return None, None, None

        # Replace NaN values (common at edges from UTM reprojection)
        # to avoid degenerate triangles and NaN normals.
        bad = ~np.isfinite(tile)
        if bad.all():
            return None, None, None
        if bad.any():
            tile = tile.copy()
            tile[bad] = float(np.nanmean(tile))

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

    def _build_tile_mesh_from_source(self, tr, tc, lod):
        """Build a tile mesh from the chunk data source.

        Unified path that handles both in-bounds and out-of-bounds
        chunks.  The chunk source provides elevation data at the
        requested LOD level; this method triangulates, computes
        normals, and transforms to world coordinates.

        Falls back to lower LOD levels for small edge tiles.
        """
        from .. import mesh as mesh_mod

        cs = self._chunk_source

        # Try requested LOD, fall back for tiles too small to triangulate
        for try_lod in range(lod, -1, -1):
            tile = cs.read_chunk(tr, tc, lod=try_lod,
                                 base_subsample=self._base_subsample)
            if tile is not None and tile.shape[0] >= 2 and tile.shape[1] >= 2:
                lod = try_lod
                break
        else:
            return None, None, None

        th, tw = tile.shape

        # Replace NaN values to avoid degenerate triangles
        bad = ~np.isfinite(tile)
        if bad.all():
            return None, None, None
        if bad.any():
            tile = tile.copy()
            fill = float(np.nanmean(tile))
            tile[bad] = fill

        # Triangulate
        num_verts = th * tw
        num_tris = (th - 1) * (tw - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        indices = np.zeros(num_tris * 3, dtype=np.int32)
        mesh_mod.triangulate_terrain(verts, indices, tile, scale=1.0)

        # World-space bounds of this tile.  Use the actual pixel extent
        # (clipped to terrain shape), not the full tile_size — edge tiles
        # may be smaller than tile_size.
        ts = self._tile_size
        c0 = tc * ts
        r0 = tr * ts
        c1 = min(c0 + ts, self._W)
        r1 = min(r0 + ts, self._H)
        wx0 = c0 * self._psx + self._offset_x
        wy0 = r0 * self._psy + self._offset_y
        wx1 = c1 * self._psx + self._offset_x
        wy1 = r1 * self._psy + self._offset_y

        # Compute smooth normals using effective pixel spacing
        if tw > 1:
            eff_psx = (wx1 - wx0) / (tw - 1)
        else:
            eff_psx = abs(self._psx)
        if th > 1:
            eff_psy = (wy1 - wy0) / (th - 1)
        else:
            eff_psy = abs(self._psy)
        normals = mesh_mod.compute_terrain_normals(
            tile, th, tw, psx=eff_psx, psy=eff_psy)

        # Transform to world coordinates
        if tw > 1:
            verts[0::3] = verts[0::3] / (tw - 1) * (wx1 - wx0) + wx0
        else:
            verts[0::3] = (wx0 + wx1) * 0.5
        if th > 1:
            verts[1::3] = verts[1::3] / (th - 1) * (wy1 - wy0) + wy0
        else:
            verts[1::3] = (wy0 + wy1) * 0.5

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

        # Convert pixel indices to CRS coordinates (e.g. UTM) when
        # available; otherwise fall back to viewer world-space coords.
        if self._crs_origin is not None and self._crs_spacing is not None:
            crs_x0, crs_y0 = self._crs_origin
            crs_dx, crs_dy = self._crs_spacing
            wx0 = crs_x0 + c0 * crs_dx
            wy0 = crs_y0 + r0 * crs_dy
            wx1 = crs_x0 + (c0 + ts) * crs_dx
            wy1 = crs_y0 + (r0 + ts) * crs_dy
        else:
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

        # Replace NaN / extreme values with 0 — NaN creates degenerate
        # triangles and fill-value leaks produce vertices at -21M Z.
        bad = ~np.isfinite(tile)
        if bad.all():
            return None, None, None
        if bad.any():
            tile = tile.copy()
            tile[bad] = 0.0

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


def _tile_gid(tr, tc):
    """Geometry ID for a terrain LOD tile."""
    return f'terrain_lod_r{tr}_c{tc}'



def _batch_gid(lod):
    """Geometry ID for a batched terrain LOD GAS at *lod* level."""
    return f'terrain_lod_batch_L{lod}'


def is_terrain_lod_gid(gid):
    """Return True if *gid* belongs to a terrain LOD tile or batch."""
    return gid.startswith('terrain_lod_')


