"""Placed-geometry mesh management for scene zarr stores.

Loads, caches, and merges per-chunk placed geometry (buildings, roads,
etc.) from a zarr scene store.  Designed to be driven by
:class:`~rtxpy.viewer.terrain_lod.TerrainLODManager`'s per-tile
visibility decisions so a single distance/LOD computation governs both
terrain and placed geometry.

The :class:`SceneMeshManager` tracks which chunks are visible, loads
mesh data lazily from zarr, applies LOD-based simplification, and
produces merged per-geometry-ID buffers ready for GPU upload.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SceneMeshManager:
    """Manages placed geometry from a scene zarr store.

    Chunks are loaded on demand as they become visible and cached in
    memory.  When the visible set or LOD assignments change, merged
    buffers are rebuilt and marked dirty for the engine to upload.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr scene store.
    """

    __slots__ = (
        '_zarr_path', '_gids', '_colors',
        '_chunk_cache',       # {(cr,cc): {gid: data}}
        '_simplify_cache',    # {(cr,cc,gid,lod): (verts, indices)}
        '_visible_set',       # set of (cr,cc) currently visible
        '_visible_lods',      # {(cr,cc): lod} for visible chunks
        '_merged',            # {gid: merged_data}  (last merge result)
        '_dirty',             # True if merged buffers need rebuild
        '_active_gids',       # set of gids currently uploaded to GPU
        'per_tick_load_limit',
        '_simplify_ratios',
    )

    def __init__(self, zarr_path):
        import zarr as _zarr

        self._zarr_path = zarr_path
        store = _zarr.open(str(zarr_path), mode='r', use_consolidated=False)
        mg = store['meshes']

        # Read per-gid colors and list of geometry IDs
        self._colors = {}
        self._gids = []
        for gid in mg:
            gg = mg[gid]
            if hasattr(gg, 'attrs'):
                self._colors[gid] = tuple(
                    gg.attrs.get('color', (0.6, 0.6, 0.6)))
                self._gids.append(gid)

        self._chunk_cache = {}
        self._simplify_cache = {}
        self._visible_set = set()
        self._visible_lods = {}
        self._merged = {}
        self._dirty = False
        self._active_gids = set()

        self.per_tick_load_limit = 2
        self._simplify_ratios = (1.0, 0.5, 0.25, 0.1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def geometry_ids(self):
        """List of geometry IDs in the scene zarr."""
        return list(self._gids)

    @property
    def colors(self):
        """Per-geometry-ID colors: ``{gid: (r, g, b, ...)}``."""
        return dict(self._colors)

    @property
    def active_gids(self):
        """Set of geometry IDs currently in the merged output."""
        return set(self._active_gids)

    @property
    def is_dirty(self):
        """True if merged buffers changed since last ``get_merged()``."""
        return self._dirty

    # ------------------------------------------------------------------
    # Chunk loading
    # ------------------------------------------------------------------

    def _load_chunk(self, cr, cc):
        """Load a single chunk from zarr into cache."""
        if (cr, cc) in self._chunk_cache:
            return
        from ..mesh_store import load_meshes_from_zarr
        meshes, _, _, curves, _spheres = load_meshes_from_zarr(
            self._zarr_path, chunks=[(cr, cc)])
        combined = {}
        for gid, data in meshes.items():
            combined[gid] = data   # (verts, indices)
        for gid, data in curves.items():
            combined[gid] = data   # (verts, widths, indices)
        self._chunk_cache[(cr, cc)] = combined

    def _get_simplified(self, cr, cc, gid, lod, verts, indices):
        """Return (possibly simplified) mesh for a chunk at a given LOD."""
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

    # ------------------------------------------------------------------
    # Visibility update (called by LOD manager)
    # ------------------------------------------------------------------

    def update_visibility(self, visible_lods, chunk_distances=None):
        """Update the visible chunk set and LOD assignments.

        Parameters
        ----------
        visible_lods : dict
            ``{(cr, cc): lod_level}`` for all currently visible chunks
            that may have placed geometry.  Only chunks within the
            scene zarr's grid are considered.
        chunk_distances : dict or None
            ``{(cr, cc): distance}`` for distance-sorted loading.

        Returns
        -------
        bool
            True if the merged output changed (caller should re-upload).
        """
        new_visible = set(visible_lods.keys())

        # Check if anything changed
        if (new_visible == self._visible_set
                and all(visible_lods.get(k) == self._visible_lods.get(k)
                        for k in new_visible)):
            # Check for deferred loads
            has_deferred = any(k not in self._chunk_cache
                               for k in new_visible)
            if not has_deferred:
                return False

        # Evict simplification cache for departed chunks
        departed = self._visible_set - new_visible
        if departed and self._simplify_cache:
            for k in [k for k in self._simplify_cache
                      if (k[0], k[1]) in departed]:
                del self._simplify_cache[k]

        self._visible_set = new_visible
        self._visible_lods = dict(visible_lods)

        # Load uncached chunks (rate-limited, closest first)
        uncached = [k for k in new_visible if k not in self._chunk_cache]
        if uncached and chunk_distances:
            uncached.sort(key=lambda c: chunk_distances.get(c, float('inf')))
        loads = 0
        for cr, cc in uncached:
            if loads >= self.per_tick_load_limit:
                break
            try:
                self._load_chunk(cr, cc)
                loads += 1
            except Exception:
                logger.debug("Failed to load mesh chunk (%d, %d)",
                             cr, cc, exc_info=True)

        # Rebuild merged buffers
        self._rebuild_merged()
        return self._dirty

    def _rebuild_merged(self):
        """Merge visible chunks per geometry ID."""
        merge_acc = {}
        for cr, cc in sorted(self._visible_set):
            chunk_data = self._chunk_cache.get((cr, cc))
            if not chunk_data:
                continue
            clod = self._visible_lods.get((cr, cc), 0)
            for gid, data in chunk_data.items():
                if len(data) == 3:
                    # Curve: (verts, widths, indices)
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
                    # Triangle mesh: (verts, indices)
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

        new_merged = {}
        for gid, (all_v, all_w, all_i, _, is_curve) in merge_acc.items():
            if all_v:
                if is_curve:
                    new_merged[gid] = (np.concatenate(all_v),
                                       np.concatenate(all_w),
                                       np.concatenate(all_i))
                else:
                    new_merged[gid] = (np.concatenate(all_v),
                                       np.concatenate(all_i))

        # Detect changes
        old_gids = set(self._merged.keys())
        new_gids = set(new_merged.keys())
        self._dirty = (old_gids != new_gids
                       or any(not _arrays_equal(self._merged.get(g),
                                               new_merged.get(g))
                              for g in new_gids))

        self._merged = new_merged
        self._active_gids = new_gids

    # ------------------------------------------------------------------
    # Output interface
    # ------------------------------------------------------------------

    def get_merged(self):
        """Return the current merged mesh buffers.

        Returns
        -------
        dict
            ``{gid: (verts, indices)}`` for triangle meshes,
            ``{gid: (verts, widths, indices)}`` for curves.
        """
        self._dirty = False
        return dict(self._merged)

    def get_removed_gids(self, prev_active):
        """Return gids that were in *prev_active* but are no longer active.

        Parameters
        ----------
        prev_active : set
            Set of gids that were active last time.

        Returns
        -------
        set
            Gids to remove from the GPU.
        """
        return prev_active - self._active_gids

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self):
        """Clear mesh and simplification caches (keeps zarr metadata)."""
        self._chunk_cache.clear()
        self._simplify_cache.clear()
        self._visible_set.clear()
        self._visible_lods.clear()
        self._merged.clear()
        self._active_gids.clear()
        self._dirty = True

    def clear_active(self):
        """Clear active state without clearing cache.

        Used on resolution change: zarr mesh data is resolution-independent,
        so the cache stays valid.  The next ``update_visibility`` call
        re-merges from cache.
        """
        self._visible_set.clear()
        self._visible_lods.clear()
        self._merged.clear()
        self._active_gids.clear()
        self._dirty = True

    def get_stats(self):
        """Return a summary string."""
        n_cached = len(self._chunk_cache)
        n_visible = len(self._visible_set)
        n_gids = len(self._active_gids)
        n_tris = 0
        n_segs = 0
        for data in self._merged.values():
            if len(data) == 3:
                n_segs += len(data[2])
            else:
                n_tris += len(data[1]) // 3
        parts = []
        if n_tris:
            parts.append(f"{n_tris:,} tris")
        if n_segs:
            parts.append(f"{n_segs:,} segs")
        detail = f" ({', '.join(parts)})" if parts else ""
        return (f"Meshes: {n_visible} chunks, "
                f"{n_gids} geoms{detail}, {n_cached} cached")


def _arrays_equal(a, b):
    """Check if two mesh data tuples are identical (by reference)."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    return all(x is y for x, y in zip(a, b))
