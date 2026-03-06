"""Terrain LOD demo — distance-based level of detail for large terrains.

Generates a synthetic 2048x2048 terrain and launches the interactive
viewer.  Press Shift+A to toggle terrain LOD, which splits the terrain
into tiles and assigns each tile a resolution based on camera distance.

Controls:
    Shift+A        Toggle terrain LOD on/off
    R / Shift+R    Manual resolution down / up (applies globally)
    Z / Shift+Z    Vertical exaggeration
    WASD / arrows  Move camera

The LOD system is most useful with large terrains where full-resolution
rendering everywhere would exceed GPU memory or hurt frame rate.  Nearby
tiles render at full detail while distant tiles are progressively
subsampled (2x, 4x, 8x).

Usage:
    python terrain_lod_demo.py
    python terrain_lod_demo.py --size 4096
"""

import argparse

import numpy as np
import xarray as xr

import rtxpy  # noqa: F401 — registers .rtx accessor


def make_terrain(size, seed=42):
    """Generate a synthetic island terrain using multi-octave Perlin noise.

    Falls back to simple sine-based terrain if xarray-spatial is not
    installed.
    """
    try:
        import cupy as cp
        from xrspatial import generate_terrain

        template = xr.DataArray(
            cp.zeros((size, size), dtype=cp.float32), dims=['y', 'x'],
        )
        terrain = generate_terrain(
            template,
            x_range=(-5000, 5000),
            y_range=(-5000, 5000),
            seed=seed,
            zfactor=3000,
            full_extent=(-5000, -5000, 5000, 5000),
            noise_mode='ridged',
            warp_strength=0.3,
            octaves=5,
        )
        # Make it an island
        sea_level = float(cp.nanmedian(terrain.data)) * 0.8
        terrain.data[:] = cp.maximum(terrain.data - sea_level, 0)
        return terrain
    except ImportError:
        pass

    # Fallback: sin/cos-based terrain (no xarray-spatial needed)
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 4 * np.pi, size, dtype=np.float32)
    x = np.linspace(0, 4 * np.pi, size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    z = (np.sin(xx) * np.cos(yy * 0.7) * 500
         + np.sin(xx * 2.3 + 1) * np.cos(yy * 1.7 + 0.5) * 200
         + rng.normal(0, 10, (size, size)).astype(np.float32))
    z = np.maximum(z, 0)
    return xr.DataArray(z, dims=['y', 'x'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Terrain LOD demo — press Shift+A in viewer to toggle",
    )
    parser.add_argument("--size", type=int, default=2048,
                        help="Grid size in pixels (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Generating {args.size}x{args.size} terrain...")
    terrain = make_terrain(args.size, seed=args.seed)

    elev = terrain.data
    if hasattr(elev, 'get'):
        elev = elev.get()
    print(f"Elevation range: {np.nanmin(elev):.0f} - {np.nanmax(elev):.0f} m")
    print(f"\nPress Shift+A to toggle terrain LOD")
    print(f"Press H for help overlay\n")

    terrain.rtx.explore(
        width=1920,
        height=1080,
        render_scale=0.5,
        subsample=1,
        title="Terrain LOD Demo",
    )
