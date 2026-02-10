_MAJOR_WATER = {'river', 'canal'}
_MINOR_WATER = {'stream', 'drain', 'ditch'}


def print_controls():
    print("\nControls:")
    print("  W/S/A/D or Arrow keys: Move camera")
    print("  Q/E or Page Up/Down: Move up/down")
    print("  I/J/K/L: Look around")
    print("  +/-: Adjust movement speed")
    print("  G: Cycle overlay layers")
    print("  O: Place observer (for viewshed)")
    print("  V: Toggle viewshed (teal glow)")
    print("  [/]: Adjust observer height")
    print("  T: Toggle shadows")
    print("  C: Cycle colormap")
    print("  U: Toggle tile overlay")
    print("  F: Screenshot")
    print("  H: Toggle help overlay")
    print("  X: Exit\n")


def classify_water_features(water_data):
    """Split water GeoJSON features into (major, minor, body) lists."""
    major = []
    minor = []
    body = []
    for f in water_data.get('features', []):
        ww = (f.get('properties') or {}).get('waterway', '')
        nat = (f.get('properties') or {}).get('natural', '')
        if ww in _MAJOR_WATER:
            major.append(f)
        elif ww in _MINOR_WATER:
            minor.append(f)
        elif nat == 'water':
            body.append(f)
        else:
            minor.append(f)
    return major, minor, body


def scale_building_heights(bldg_data, elev_scale=0.025, default_height_m=8.0):
    """Scale MS building heights in-place to match terrain elevation scale."""
    for feat in bldg_data.get("features", []):
        props = feat.get("properties", {})
        h = props.get("height", -1)
        if not isinstance(h, (int, float)) or h <= 0:
            h = default_height_m
        props["height"] = h * elev_scale
