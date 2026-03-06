"""Distance-based terrain LOD manager.

Divides a terrain raster into a grid of tiles.  Each tile is assigned
an LOD level based on its distance to the camera, controlling the mesh
resolution (LOD 0 = full detail, LOD *k* = 2^k subsampling).  Tiles
are built as individual GAS entries in the OptiX IAS so the raytracer
traverses only the detail actually needed.

Tile edges get a short vertical skirt to hide T-junction cracks where
adjacent tiles have different LOD levels.
"""

import numpy as np

from ..lod import compute_lod_level, compute_lod_distances


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
        '_active_tiles', '_last_update_pos', '_update_threshold',
        '_base_subsample',
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

        # Pre-compute tile centres in world coordinates
        self._tile_centers = {}
        for tr in range(self._n_tile_rows):
            for tc in range(self._n_tile_cols):
                r0 = tr * tile_size
                c0 = tc * tile_size
                r1 = min(r0 + tile_size, H)
                c1 = min(c0 + tile_size, W)
                cx = (c0 + c1) * 0.5 * pixel_spacing_x
                cy = (r0 + r1) * 0.5 * pixel_spacing_y
                self._tile_centers[(tr, tc)] = (cx, cy)

        # LOD distance thresholds
        tile_diag = np.sqrt(
            (tile_size * pixel_spacing_x) ** 2
            + (tile_size * pixel_spacing_y) ** 2
        )
        self._lod_distances = compute_lod_distances(
            tile_diag, factor=lod_distance_factor, max_lod=max_lod)

        # Per-tile state
        self._tile_lods = {}   # (tr, tc) -> current LOD level
        self._tile_cache = {}  # (tr, tc, lod, base_sub) -> (verts, indices)
        self._active_tiles = set()  # GAS IDs currently in the scene

        # Movement threshold before re-evaluating LOD
        self._last_update_pos = None
        self._update_threshold = tile_diag * 0.25

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            self._tile_cache.clear()
            self._tile_lods.clear()

    def set_terrain(self, terrain_np):
        """Replace the terrain data (e.g. after dynamic reload)."""
        self._terrain_np = terrain_np
        H, W = terrain_np.shape
        self._H = H
        self._W = W
        self._tile_cache.clear()
        self._tile_lods.clear()

    def update(self, camera_pos, rtx, ve=1.0, force=False):
        """Re-evaluate LOD per tile and rebuild changed tiles.

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

        Returns
        -------
        bool
            True if any tile GAS was added or updated.
        """
        cam_x, cam_y = float(camera_pos[0]), float(camera_pos[1])

        # Skip if camera hasn't moved enough
        if not force and self._last_update_pos is not None:
            dx = cam_x - self._last_update_pos[0]
            dy = cam_y - self._last_update_pos[1]
            if dx * dx + dy * dy < self._update_threshold ** 2:
                return False

        self._last_update_pos = (cam_x, cam_y)

        changed = False
        new_tile_ids = set()

        for tr in range(self._n_tile_rows):
            for tc in range(self._n_tile_cols):
                cx, cy = self._tile_centers[(tr, tc)]
                dist = np.sqrt((cam_x - cx) ** 2 + (cam_y - cy) ** 2)

                lod = compute_lod_level(dist, self._lod_distances)
                lod = min(lod, self._max_lod)

                tile_id = _tile_gid(tr, tc)
                new_tile_ids.add(tile_id)

                prev_lod = self._tile_lods.get((tr, tc), -1)
                if lod != prev_lod or force:
                    verts, indices = self._get_tile_mesh(tr, tc, lod)
                    if verts is not None:
                        # Apply VE
                        if ve != 1.0:
                            verts = verts.copy()
                            verts[2::3] *= ve
                        rtx.add_geometry(tile_id, verts, indices)
                        self._tile_lods[(tr, tc)] = lod
                        changed = True

        # Remove stale tiles (shouldn't happen, but be safe)
        for old_id in self._active_tiles - new_tile_ids:
            rtx.remove_geometry(old_id)
            changed = True

        self._active_tiles = new_tile_ids
        return changed

    def remove_all(self, rtx):
        """Remove all LOD tile geometries from the scene."""
        for tile_id in list(self._active_tiles):
            rtx.remove_geometry(tile_id)
        self._active_tiles.clear()
        self._tile_lods.clear()
        self._last_update_pos = None

    def get_stats(self):
        """Return a summary string of LOD state."""
        if not self._tile_lods:
            return "LOD: no tiles"
        from collections import Counter
        counts = Counter(self._tile_lods.values())
        parts = [f"L{lvl}:{cnt}" for lvl, cnt in sorted(counts.items())]
        total = self._n_tile_rows * self._n_tile_cols
        return f"LOD tiles: {total} ({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_tile_mesh(self, tr, tc, lod):
        """Build or retrieve cached tile mesh."""
        cache_key = (tr, tc, lod, self._base_subsample)
        cached = self._tile_cache.get(cache_key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()

        verts, indices = self._build_tile_mesh(tr, tc, lod)
        if verts is not None:
            self._tile_cache[cache_key] = (verts.copy(), indices.copy())
        return verts, indices

    def _build_tile_mesh(self, tr, tc, lod):
        """Triangulate a single tile at the given LOD level."""
        from .. import mesh as mesh_mod

        subsample = self._base_subsample * (2 ** lod)
        r0 = tr * self._tile_size
        c0 = tc * self._tile_size
        # Extend by one pixel so adjacent tiles share boundary vertices,
        # eliminating the one-pixel gap that causes shading seams.
        r1 = min(r0 + self._tile_size + 1, self._H)
        c1 = min(c0 + self._tile_size + 1, self._W)

        # Extract tile data with subsampling
        tile = self._terrain_np[r0:r1:subsample, c0:c1:subsample]
        th, tw = tile.shape
        if th < 2 or tw < 2:
            return None, None

        # Triangulate using the fast numba/CUDA path
        num_verts = th * tw
        num_tris = (th - 1) * (tw - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        indices = np.zeros(num_tris * 3, dtype=np.int32)
        mesh_mod.triangulate_terrain(verts, indices, tile, scale=1.0)

        # Transform from local grid coords to world coords.
        # triangulate_terrain writes x=w, y=h in grid-local pixel indices.
        # We need:  x = (c0 + w*subsample) * psx
        #           y = (r0 + h*subsample) * psy
        verts[0::3] = verts[0::3] * subsample * self._psx + c0 * self._psx
        verts[1::3] = verts[1::3] * subsample * self._psy + r0 * self._psy

        # Only add skirt on exterior edges (terrain boundary).
        # Interior edges shared with adjacent tiles via the +1 overlap
        # don't need skirt — overlapping skirt walls cause artifacts.
        edges = (
            tr == 0,                          # top
            tc == self._n_tile_cols - 1,      # right
            tr == self._n_tile_rows - 1,      # bottom
            tc == 0,                          # left
        )
        verts, indices = _add_tile_skirt(verts, indices, th, tw, edges=edges)

        return verts, indices


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _tile_gid(tr, tc):
    """Geometry ID for a terrain LOD tile."""
    return f'terrain_lod_r{tr}_c{tc}'


def is_terrain_lod_gid(gid):
    """Return True if *gid* belongs to a terrain LOD tile."""
    return gid.startswith('terrain_lod_r')


def _add_tile_skirt(vertices, indices, H, W, skirt_depth=None,
                    edges=(True, True, True, True)):
    """Add a thin skirt around specified tile edges.

    Parameters
    ----------
    edges : tuple of bool
        ``(top, right, bottom, left)`` — which edges get skirt
        geometry.  Interior tile edges shared with adjacent tiles
        should be False to avoid overlapping wall triangles.
    """
    if not any(edges):
        return vertices, indices

    z_vals = vertices[2::3]
    z_min = float(np.nanmin(z_vals))
    z_max = float(np.nanmax(z_vals))

    if skirt_depth is None:
        z_range = z_max - z_min
        skirt_depth = max(0.5, z_range * 0.02)

    skirt_z = z_min - skirt_depth

    # Build clockwise perimeter (same order as mesh.add_terrain_skirt)
    top = np.arange(W, dtype=np.int32)
    right = (np.arange(1, H, dtype=np.int32)) * W + (W - 1)
    bottom = (H - 1) * W + np.arange(W - 2, -1, -1, dtype=np.int32)
    left = np.arange(H - 2, 0, -1, dtype=np.int32) * W
    perim = np.concatenate([top, right, bottom, left])
    n_perim = len(perim)
    n_orig = len(vertices) // 3

    skirt_verts = np.empty(n_perim * 3, dtype=np.float32)
    skirt_verts[0::3] = vertices[perim * 3]
    skirt_verts[1::3] = vertices[perim * 3 + 1]
    skirt_verts[2::3] = skirt_z

    # Mask: only create wall triangles for active edges.
    # Perimeter segments per edge: top W-1, right H-1, bottom W-1, left H-1.
    edge_top, edge_right, edge_bottom, edge_left = edges
    seg_mask = np.zeros(n_perim, dtype=bool)
    off = 0
    for active, count in [(edge_top, W - 1), (edge_right, H - 1),
                           (edge_bottom, W - 1), (edge_left, H - 1)]:
        if active:
            seg_mask[off:off + count] = True
        off += count

    active_segs = np.where(seg_mask)[0].astype(np.int32)
    if len(active_segs) == 0:
        return vertices, indices

    idx_next = (active_segs + 1) % n_perim
    top_a = perim[active_segs]
    top_b = perim[idx_next]
    bot_a = (n_orig + active_segs).astype(np.int32)
    bot_b = (n_orig + idx_next).astype(np.int32)

    n_active = len(active_segs)
    wall_tris = np.empty(n_active * 6, dtype=np.int32)
    wall_tris[0::6] = top_a
    wall_tris[1::6] = bot_b
    wall_tris[2::6] = top_b
    wall_tris[3::6] = top_a
    wall_tris[4::6] = bot_a
    wall_tris[5::6] = bot_b

    new_verts = np.concatenate([vertices, skirt_verts])
    new_indices = np.concatenate([indices, wall_tris])
    return new_verts, new_indices
