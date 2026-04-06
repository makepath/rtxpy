"""ParticleSystem — pure data container with lifecycle management.

No rendering, no advection, no viewer references.  Just arrays and
the spawn/age/respawn/trail mechanics shared by all particle types.
"""

import numpy as np


class ParticleSystem:
    """Reusable particle state: positions, ages, lifetimes, optional trails.

    Parameters
    ----------
    n : int
        Number of particles.
    grid_shape : tuple of int
        (H, W) of the domain grid.  Particles live in row/col pixel coords.
    max_age : int
        Maximum lifetime in ticks.  Actual per-particle lifetimes are
        jittered to ``[max_age // 2, max_age)`` to prevent sync'd deaths.
    trail_len : int, optional
        If > 0, a ``(n, trail_len, 2)`` ring buffer is allocated.
    weight_grid : ndarray, optional
        (H, W) spawn-probability weights.  If *None*, particles spawn
        uniformly over the grid.
    """

    def __init__(self, n, grid_shape, max_age, trail_len=0,
                 weight_grid=None):
        H, W = grid_shape
        self.n = n
        self.grid_shape = grid_shape
        self.max_age = max_age
        self.trail_len = trail_len

        # Core arrays
        self.positions = np.column_stack([
            np.random.uniform(0, H, n),
            np.random.uniform(0, W, n),
        ]).astype(np.float32)

        self.ages = np.random.randint(0, max_age, n).astype(np.int32)
        self.lifetimes = np.random.randint(
            max(1, max_age // 2), max(2, max_age), n,
        ).astype(np.int32)

        # Optional trail ring buffer
        if trail_len > 0:
            self.trails = np.zeros((n, trail_len, 2), dtype=np.float32)
            # Initialize all trail slots to current position
            for t in range(trail_len):
                self.trails[:, t, :] = self.positions
        else:
            self.trails = None

        # Spawn distribution (cached for fast respawn)
        self._build_spawn_distribution(weight_grid)

    # ------------------------------------------------------------------
    # Spawn distribution
    # ------------------------------------------------------------------

    def _build_spawn_distribution(self, weight_grid=None):
        """Precompute spawn indices and probabilities from *weight_grid*.

        If *weight_grid* is None or all-zero, falls back to uniform
        sampling over the full grid.
        """
        H, W = self.grid_shape
        if weight_grid is not None:
            flat = weight_grid.ravel().astype(np.float64)
            total = flat.sum()
            if total > 0:
                valid = flat > 0
                self.spawn_indices = np.where(valid)[0]
                probs = flat[valid]
                self.spawn_probs = probs / probs.sum()
                return
        # Uniform fallback
        self.spawn_indices = None
        self.spawn_probs = None

    def set_spawn_weights(self, weight_grid):
        """Update spawn distribution from a new weight grid."""
        self._build_spawn_distribution(weight_grid)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def tick_age(self):
        """Increment ages by 1 and return a boolean respawn mask.

        Particles are flagged for respawn when:
        - ``age >= lifetime``
        - position is out of bounds
        - position contains NaN
        """
        self.ages += 1

        H, W = self.grid_shape
        pts = self.positions
        nan_pos = np.isnan(pts[:, 0]) | np.isnan(pts[:, 1])
        oob = (nan_pos
               | (pts[:, 0] < 0) | (pts[:, 0] >= H)
               | (pts[:, 1] < 0) | (pts[:, 1] >= W))
        old = self.ages >= self.lifetimes
        return oob | old

    def respawn(self, mask):
        """Re-place particles where *mask* is True.

        New positions are drawn from the cached spawn distribution.
        Ages are reset to 0 and lifetimes are re-jittered.
        Trails (if present) are reset to the new position.
        """
        n_respawn = int(mask.sum())
        if n_respawn == 0:
            return

        H, W = self.grid_shape

        if self.spawn_indices is not None:
            chosen = np.random.choice(
                self.spawn_indices, size=n_respawn, p=self.spawn_probs)
            self.positions[mask, 0] = (chosen // W).astype(np.float32) + \
                np.random.uniform(-0.5, 0.5, n_respawn).astype(np.float32)
            self.positions[mask, 1] = (chosen % W).astype(np.float32) + \
                np.random.uniform(-0.5, 0.5, n_respawn).astype(np.float32)
        else:
            self.positions[mask, 0] = np.random.uniform(0, H, n_respawn)
            self.positions[mask, 1] = np.random.uniform(0, W, n_respawn)

        self.ages[mask] = 0
        self.lifetimes[mask] = np.random.randint(
            max(1, self.max_age // 2), max(2, self.max_age), n_respawn,
        ).astype(np.int32)

        # Reset trails for respawned particles
        if self.trails is not None:
            for t in range(self.trail_len):
                self.trails[mask, t, :] = self.positions[mask]

    def push_trail(self):
        """Shift the trail buffer and prepend current positions.

        Call this *before* advection so the trail records the pre-move
        position (the standard ring-buffer pattern used by wind and hydro).
        """
        if self.trails is None:
            return
        self.trails[:, 1:, :] = self.trails[:, :-1, :]
        self.trails[:, 0, :] = self.positions
