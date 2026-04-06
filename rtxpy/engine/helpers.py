"""Small helper functions used by the interactive viewer."""

import numpy as np

from ..rtx import has_cupy

if has_cupy:
    import cupy as cp


def _glfw_to_key(glfw_key, mods):
    """Translate a GLFW key code + modifiers to the string format used by
    _handle_key_press / _handle_key_release.

    Returns (raw_key, key_lower) matching the old matplotlib convention:
    - raw_key preserves case (uppercase if SHIFT held for letters)
    - key_lower is always lowercase
    """
    import glfw

    _SPECIAL = {
        glfw.KEY_UP: 'up', glfw.KEY_DOWN: 'down',
        glfw.KEY_LEFT: 'left', glfw.KEY_RIGHT: 'right',
        glfw.KEY_PAGE_UP: 'pageup', glfw.KEY_PAGE_DOWN: 'pagedown',
        glfw.KEY_ESCAPE: 'escape',
        glfw.KEY_EQUAL: '=', glfw.KEY_MINUS: '-',
        glfw.KEY_COMMA: ',', glfw.KEY_PERIOD: '.',
        glfw.KEY_LEFT_BRACKET: '[', glfw.KEY_RIGHT_BRACKET: ']',
        glfw.KEY_SEMICOLON: ';', glfw.KEY_APOSTROPHE: "'",
    }

    if glfw_key in _SPECIAL:
        raw = _SPECIAL[glfw_key]
        # SHIFT variants for special keys
        if mods & glfw.MOD_SHIFT:
            if raw == '=':
                raw = '+'
            elif raw == '-':
                raw = '_'  # unlikely to be used, keep '-' behaviour
            elif raw == ';':
                raw = ':'
            elif raw == "'":
                raw = '"'
        return raw, raw.lower()

    # Letter keys A-Z
    if glfw.KEY_A <= glfw_key <= glfw.KEY_Z:
        lower = chr(glfw_key - glfw.KEY_A + ord('a'))
        if mods & glfw.MOD_SHIFT:
            return lower.upper(), lower
        return lower, lower

    # Digit keys 0-9
    if glfw.KEY_0 <= glfw_key <= glfw.KEY_9:
        digit = chr(glfw_key - glfw.KEY_0 + ord('0'))
        return digit, digit

    return '', ''


def _bilinear_terrain_z(terrain, vx, vy, psx, psy):
    """Sample terrain Z at world positions using bilinear interpolation.

    This matches the interpolation used by the triangle mesh surface,
    preventing Z mismatches between placed meshes and the rendered terrain.

    Supports both numpy and cupy arrays -- the array module is chosen
    automatically based on the type of ``terrain``.
    """
    if has_cupy and isinstance(terrain, cp.ndarray):
        xp = cp
    else:
        xp = np
    H, W = terrain.shape
    cols = vx / psx
    rows = vy / psy
    cols = xp.clip(cols, 0, W - 1)
    rows = xp.clip(rows, 0, H - 1)
    x0 = xp.clip(xp.floor(cols).astype(xp.int32), 0, max(W - 2, 0))
    y0 = xp.clip(xp.floor(rows).astype(xp.int32), 0, max(H - 2, 0))
    fx = cols - x0
    fy = rows - y0
    z00 = terrain[y0, x0].astype(xp.float32)
    z10 = terrain[y0, xp.minimum(x0 + 1, W - 1)].astype(xp.float32)
    z01 = terrain[xp.minimum(y0 + 1, H - 1), x0].astype(xp.float32)
    z11 = terrain[xp.minimum(y0 + 1, H - 1),
                  xp.minimum(x0 + 1, W - 1)].astype(xp.float32)
    z00 = xp.where(xp.isnan(z00), 0.0, z00)
    z10 = xp.where(xp.isnan(z10), 0.0, z10)
    z01 = xp.where(xp.isnan(z01), 0.0, z01)
    z11 = xp.where(xp.isnan(z11), 0.0, z11)
    return (z00 * (1 - fx) * (1 - fy) +
            z10 * fx * (1 - fy) +
            z01 * (1 - fx) * fy +
            z11 * fx * fy)


def _add_overlay(viewer, name, data, color_lut=None):
    """Add or replace an overlay layer on *viewer* and switch to it.

    Must be called on the main (render) thread.

    Parameters
    ----------
    color_lut : np.ndarray, optional
        (256, 3) float32 categorical palette LUT.  When provided, the
        overlay bypasses the terrain colormap and color stretch.
    """
    viewer._overlay_layers[name] = data
    viewer._base_overlay_layers[name] = data
    if color_lut is not None:
        viewer._overlay_color_luts[name] = color_lut
    viewer._overlay_names = list(viewer._overlay_layers.keys())
    viewer._terrain_layer_order = (
        ['elevation'] + list(viewer._overlay_names))
    idx = viewer._terrain_layer_order.index(name)
    viewer._terrain_layer_idx = idx
    viewer._active_color_data = None
    viewer._active_overlay_data = data
    viewer._overlay_as_water = name.startswith('flood_')
    viewer._active_overlay_color_lut = color_lut
    # Skip render if camera not yet initialized (overlays added before run())
    if viewer.position is not None:
        viewer._update_frame()
    print(f"Terrain: {name}")
