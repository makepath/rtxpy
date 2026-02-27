"""
Point cloud loading and processing for rtxpy.

Supports LAS/LAZ files via laspy, numpy arrays, and callable data sources.
Provides filtering by classification, return number, spatial bounds,
subsampling, and color mode generation.
"""

import numpy as np

# ASPRS LiDAR classification codes → default RGBA colors
CLASSIFICATION_COLORS = {
    0: (0.6, 0.6, 0.6, 1.0),    # Never classified (gray)
    1: (0.5, 0.5, 0.5, 1.0),    # Unassigned (gray)
    2: (0.6, 0.4, 0.2, 1.0),    # Ground (brown)
    3: (0.1, 0.6, 0.1, 1.0),    # Low vegetation (light green)
    4: (0.0, 0.5, 0.0, 1.0),    # Medium vegetation (green)
    5: (0.0, 0.35, 0.0, 1.0),   # High vegetation (dark green)
    6: (0.8, 0.2, 0.2, 1.0),    # Building (red)
    7: (0.5, 0.5, 0.5, 1.0),    # Low point / noise (gray)
    8: (0.5, 0.5, 0.5, 1.0),    # Reserved
    9: (0.2, 0.4, 0.8, 1.0),    # Water (blue)
    10: (0.7, 0.7, 0.3, 1.0),   # Rail (yellow)
    11: (0.3, 0.3, 0.3, 1.0),   # Road surface (dark gray)
    12: (0.5, 0.5, 0.5, 1.0),   # Reserved
    13: (0.9, 0.9, 0.0, 1.0),   # Wire - guard (yellow)
    14: (0.9, 0.6, 0.0, 1.0),   # Wire - conductor (orange)
    15: (0.7, 0.7, 0.7, 1.0),   # Transmission tower (light gray)
    16: (0.9, 0.9, 0.0, 1.0),   # Wire - connector (yellow)
    17: (0.4, 0.4, 0.8, 1.0),   # Bridge deck (blue-gray)
    18: (0.5, 0.5, 0.5, 1.0),   # High noise
}


def load_pointcloud(source, classification=None, returns=None,
                    bounds=None, subsample=1, max_points=None,
                    thin=None):
    """
    Load a point cloud from a file path or numpy array.

    Args:
        source: One of:
            - str: Path to LAS/LAZ/COPC file (requires laspy)
            - ndarray: (N, 3) float32 array of XYZ coordinates
            - callable: Function returning (centers_N3, attributes_dict)
        classification: Optional int or list of ints to filter by ASPRS class.
        returns: Optional 'first', 'last', or 'all' to filter by return number.
        bounds: Optional (xmin, ymin, xmax, ymax) spatial crop.
        subsample: Keep every Nth point (default 1 = all).
        thin: Grid cell size for spatial thinning (in source CRS units,
            typically metres). Keeps one point per XY grid cell, removing
            density striations from overlapping flight lines. None disables.
        max_points: Maximum number of points to keep (random sample if exceeded).

    Returns:
        Tuple of (centers, attributes) where:
            - centers: (N, 3) float32 array of XYZ coordinates
            - attributes: dict with optional keys:
                'intensity': (N,) float32 [0, 1]
                'classification': (N,) int32
                'rgb': (N, 3) float32 [0, 1]
                'return_number': (N,) int32
                'number_of_returns': (N,) int32
    """
    if callable(source) and not isinstance(source, (str, np.ndarray)):
        centers, attributes = source()
        centers = np.asarray(centers, dtype=np.float32)
    elif isinstance(source, np.ndarray):
        centers = np.asarray(source, dtype=np.float32)
        if centers.ndim == 1:
            centers = centers.reshape(-1, 3)
        attributes = {}
    elif isinstance(source, str):
        centers, attributes = _load_las(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Apply filters
    mask = np.ones(len(centers), dtype=bool)

    if classification is not None:
        if 'classification' in attributes:
            if isinstance(classification, int):
                classification = [classification]
            cls = attributes['classification']
            mask &= np.isin(cls, classification)

    if returns is not None and 'return_number' in attributes:
        rn = attributes['return_number']
        nr = attributes.get('number_of_returns', rn)
        if returns == 'first':
            mask &= rn == 1
        elif returns == 'last':
            mask &= rn == nr

    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        mask &= ((centers[:, 0] >= xmin) & (centers[:, 0] <= xmax) &
                 (centers[:, 1] >= ymin) & (centers[:, 1] <= ymax))

    # Apply mask
    if not mask.all():
        centers = centers[mask]
        attributes = {k: v[mask] for k, v in attributes.items()}

    # Subsample
    if subsample > 1:
        centers = centers[::subsample]
        attributes = {k: v[::subsample] for k, v in attributes.items()}

    # Spatial grid thinning — one point per XY cell
    if thin is not None and thin > 0 and len(centers) > 0:
        cell_x = (centers[:, 0] / thin).astype(np.int64)
        cell_y = (centers[:, 1] / thin).astype(np.int64)
        cell_keys = cell_x + cell_y * 2_000_000_003  # large prime avoids collisions
        _, keep = np.unique(cell_keys, return_index=True)
        keep.sort()
        centers = centers[keep]
        attributes = {k: v[keep] for k, v in attributes.items()}

    # Cap total points
    if max_points is not None and len(centers) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(centers), max_points, replace=False)
        idx.sort()
        centers = centers[idx]
        attributes = {k: v[idx] for k, v in attributes.items()}

    return centers, attributes


def _load_las(filepath):
    """Load a LAS/LAZ file using laspy."""
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy is required for LAS/LAZ file loading. "
            "Install with: pip install laspy[lazrs]"
        )

    las = laspy.read(filepath)

    centers = np.column_stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ]).astype(np.float32)

    attributes = {}

    # Intensity (normalize to [0, 1])
    if hasattr(las, 'intensity'):
        intensity = np.asarray(las.intensity, dtype=np.float32)
        max_val = intensity.max()
        if max_val > 0:
            intensity = intensity / max_val
        attributes['intensity'] = intensity

    # Classification
    if hasattr(las, 'classification'):
        attributes['classification'] = np.asarray(
            las.classification, dtype=np.int32)

    # Return number
    if hasattr(las, 'return_number'):
        attributes['return_number'] = np.asarray(
            las.return_number, dtype=np.int32)

    if hasattr(las, 'number_of_returns'):
        attributes['number_of_returns'] = np.asarray(
            las.number_of_returns, dtype=np.int32)

    # RGB color (LAS point formats 2, 3, 5, 7, 8, 10)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = np.asarray(las.red, dtype=np.float32)
        g = np.asarray(las.green, dtype=np.float32)
        b = np.asarray(las.blue, dtype=np.float32)
        # LAS stores 16-bit RGB (0-65535)
        max_rgb = max(r.max(), g.max(), b.max(), 1.0)
        if max_rgb > 255:
            r /= 65535.0
            g /= 65535.0
            b /= 65535.0
        elif max_rgb > 1:
            r /= 255.0
            g /= 255.0
            b /= 255.0
        attributes['rgb'] = np.column_stack([r, g, b])

    return centers, attributes


def build_colors(centers, attributes, color_mode='elevation',
                 colormap=None):
    """
    Build per-point RGBA color array from point cloud attributes.

    Args:
        centers: (N, 3) float32 XYZ coordinates.
        attributes: Dict of per-point attributes from load_pointcloud().
        color_mode: One of:
            - 'elevation': Color by Z value using colormap
            - 'intensity': Grayscale from intensity attribute
            - 'classification': ASPRS standard colors
            - 'rgb': Direct RGB from LAS file
            - tuple (r, g, b): Uniform solid color
        colormap: Optional (256, 3) float32 lookup table for elevation mode.
                  If None, uses a default terrain colormap.

    Returns:
        (N, 4) float32 RGBA array with values in [0, 1].
    """
    n = len(centers)
    colors = np.ones((n, 4), dtype=np.float32)  # default alpha = 1.0

    if isinstance(color_mode, (tuple, list)):
        # Uniform solid color
        colors[:, 0] = color_mode[0]
        colors[:, 1] = color_mode[1]
        colors[:, 2] = color_mode[2]

    elif color_mode == 'elevation':
        z = centers[:, 2]
        z_min, z_max = np.nanmin(z), np.nanmax(z)
        z_range = z_max - z_min
        if z_range < 1e-6:
            z_range = 1.0
        norm = np.clip((z - z_min) / z_range, 0.0, 1.0)

        if colormap is not None:
            idx = (norm * 255).astype(np.int32)
            idx = np.clip(idx, 0, 255)
            colors[:, :3] = colormap[idx]
        else:
            # Default terrain ramp: blue → green → yellow → red → white
            colors[:, 0] = np.clip(norm * 3.0 - 1.0, 0, 1)
            colors[:, 1] = np.clip(1.0 - abs(norm * 3.0 - 1.5) * 1.5, 0, 1)
            colors[:, 2] = np.clip(1.0 - norm * 2.0, 0, 1)

    elif color_mode == 'intensity':
        if 'intensity' in attributes:
            intensity = attributes['intensity']
            colors[:, 0] = intensity
            colors[:, 1] = intensity
            colors[:, 2] = intensity
        else:
            colors[:, :3] = 0.5  # fallback gray

    elif color_mode == 'classification':
        if 'classification' in attributes:
            cls = attributes['classification']
            for code, rgba in CLASSIFICATION_COLORS.items():
                mask = cls == code
                if mask.any():
                    colors[mask, 0] = rgba[0]
                    colors[mask, 1] = rgba[1]
                    colors[mask, 2] = rgba[2]
                    colors[mask, 3] = rgba[3]
            # Unknown classes → default gray
            unknown = ~np.isin(cls, list(CLASSIFICATION_COLORS.keys()))
            if unknown.any():
                colors[unknown, :3] = 0.5
        else:
            colors[:, :3] = 0.5

    elif color_mode == 'rgb':
        if 'rgb' in attributes:
            colors[:, :3] = attributes['rgb']
        else:
            colors[:, :3] = 0.5

    return colors
