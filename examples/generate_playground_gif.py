"""Generate a GIF from the playground viewshed/hillshade animation.

This script creates frames from the Crater Lake hiking animation and
combines them into a GIF suitable for the README.
"""

import numpy as np
import cupy
import xarray as xr
from pathlib import Path
from PIL import Image
import io

from rtxpy import RTX, viewshed, hillshade


def load_terrain():
    """Load Crater Lake terrain data."""
    import rioxarray as rxr

    dem_path = Path(__file__).parent / "crater_lake_national_park.tif"

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found at {dem_path}. Run playground.py first to download it.")

    print(f"Loading DEM: {dem_path}")
    terrain = rxr.open_rasterio(str(dem_path), masked=True).squeeze()

    # Subsample aggressively for smaller GIF file size
    terrain = terrain[::10, ::10]

    # Crop edges to remove invalid border values
    crop = 20
    terrain = terrain[crop:-crop, crop:-crop]

    # Scale down elevation for visualization
    terrain.data = terrain.data * 0.2

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Convert to cupy for GPU processing
    terrain.data = cupy.asarray(terrain.data)

    print(f"Terrain loaded: {terrain.shape}")
    return terrain


def generate_hiking_path(x_coords, y_coords, num_points=360):
    """Generate a hiking path around Crater Lake (roughly circular)."""
    cx = (x_coords.min() + x_coords.max()) / 2
    cy = (y_coords.min() + y_coords.max()) / 2

    rx = (x_coords.max() - x_coords.min()) * 0.25
    ry = (y_coords.max() - y_coords.min()) * 0.25

    angles = np.linspace(0, 2 * np.pi, num_points)
    wobble = np.sin(angles * 8) * 0.1

    path_x = cx + (rx + rx * wobble) * np.cos(angles)
    path_y = cy + (ry + ry * wobble) * np.sin(angles)

    return path_x, path_y


def coords_to_pixel(x, y, x_coords, y_coords):
    """Convert data coordinates to pixel coordinates."""
    px = np.searchsorted(x_coords, x)
    py = np.searchsorted(-y_coords, -y)
    return int(np.clip(px, 0, len(x_coords) - 1)), int(np.clip(py, 0, len(y_coords) - 1))


def draw_legend(colors, x=10, y=10):
    """Draw a legend in the corner of the frame."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    H, W = colors.shape[:2]

    # Create a small PIL image for drawing text
    legend_w, legend_h = 90, 52
    legend = PILImage.new('RGBA', (legend_w, legend_h), (0, 0, 0, 180))
    draw = ImageDraw.Draw(legend)

    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Legend entries: color swatch + label
    entries = [
        ((50, 220, 50), "Visible"),
        ((80, 80, 85), "Not Visible"),
        ((0, 255, 255), "Observer"),
    ]

    for i, (color, label) in enumerate(entries):
        row_y = 5 + i * 15
        # Draw color swatch
        draw.rectangle([5, row_y, 15, row_y + 10], fill=color)
        # Draw label
        draw.text((20, row_y - 1), label, fill=(255, 255, 255), font=font)

    # Convert legend to numpy and overlay on frame
    legend_arr = np.array(legend)

    # Blend legend onto frame
    for ly in range(legend_h):
        for lx in range(legend_w):
            fy, fx = y + ly, x + lx
            if 0 <= fy < H and 0 <= fx < W:
                alpha = legend_arr[ly, lx, 3] / 255.0
                colors[fy, fx, :3] = (
                    colors[fy, fx, :3] * (1 - alpha) + legend_arr[ly, lx, :3] * alpha
                ).astype(np.uint8)


def draw_observer_marker(colors, px, py, radius=6, glow_radius=18):
    """Draw a glowing teal marker with dark outline at the observer's position."""
    H, W = colors.shape[:2]
    outline_width = 2

    # Draw outer glow, dark outline, then bright center
    for dy in range(-glow_radius, glow_radius + 1):
        for dx in range(-glow_radius, glow_radius + 1):
            dist_sq = dx*dx + dy*dy
            if dist_sq <= glow_radius * glow_radius:
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    dist = np.sqrt(dist_sq)
                    if dist <= radius:
                        # Bright cyan/teal center
                        colors[ny, nx] = [0, 255, 255, 255]
                    elif dist <= radius + outline_width:
                        # Dark outline for contrast
                        colors[ny, nx] = [0, 40, 40, 255]
                    elif dist <= glow_radius:
                        # Glow falloff - blend cyan with existing color
                        t = (dist - radius - outline_width) / (glow_radius - radius - outline_width)
                        glow_strength = (1 - t) ** 1.5  # Slightly softer falloff
                        existing = colors[ny, nx, :3].astype(np.float32)
                        cyan = np.array([0, 255, 255], dtype=np.float32)
                        blended = existing + (cyan - existing) * glow_strength * 0.7
                        colors[ny, nx, :3] = np.clip(blended, 0, 255).astype(np.uint8)


def generate_frames(terrain, num_frames=72):
    """Generate animation frames.

    Parameters
    ----------
    terrain : xarray.DataArray
        The terrain data.
    num_frames : int
        Number of frames to generate. Both the hillshade and hiker will
        complete exactly one full 360° loop in this many frames.
    """
    H, W = terrain.data.shape
    rtx = RTX()

    x_coords = terrain.indexes.get('x').values
    y_coords = terrain.indexes.get('y').values

    path_x, path_y = generate_hiking_path(x_coords, y_coords, num_points=360)

    frames = []
    azimuth = 225

    print(f"Generating {num_frames} frames...")

    # Calculate rotation per frame for full 360° loop
    azimuth_step = 360 / num_frames
    hiker_step = 360 / num_frames

    for frame_idx in range(num_frames):
        path_idx = int((frame_idx * hiker_step) % 360)

        vsw = path_x[path_idx]
        vsh = path_y[path_idx]
        azimuth = (225 + frame_idx * azimuth_step) % 360

        # Compute hillshade and viewshed
        hs = hillshade(terrain,
                       shadows=True,
                       azimuth=azimuth,
                       angle_altitude=25,
                       rtx=rtx)
        vs = viewshed(terrain,
                      x=vsw,
                      y=vsh,
                      observer_elev=100.0,
                      rtx=rtx)

        # Convert to numpy arrays
        hs_data = hs.data.get() if hasattr(hs.data, 'get') else hs.data
        vs_data = vs.data.get() if hasattr(vs.data, 'get') else vs.data

        # Track NaN and zero pixels before converting - these will be transparent
        transparent_mask = np.isnan(hs_data) | np.isnan(vs_data) | (hs_data == 0)

        hs_data = np.nan_to_num(hs_data, nan=0.5)
        gray = np.uint8(np.clip(hs_data * 200, 0, 255))

        # Viewshed returns -1 for invisible, 0-180 for visible (angle)
        visible_mask = vs_data >= 0
        not_visible_mask = (vs_data < 0) & ~transparent_mask

        # Compose the final image with alpha channel (RGBA)
        colors = np.zeros((H, W, 4), dtype=np.uint8)
        colors[:, :, 0] = gray
        colors[:, :, 1] = gray
        colors[:, :, 2] = gray
        colors[:, :, 3] = 255  # Fully opaque by default

        # Tint visible areas bright lime green - make it really pop!
        colors[visible_mask, 0] = 50   # Low red
        colors[visible_mask, 1] = np.minimum(255, gray[visible_mask].astype(np.int16) + 120).astype(np.uint8)  # Bright green
        colors[visible_mask, 2] = 50   # Low blue

        # Tint non-visible areas darker gray
        colors[not_visible_mask, 0] = (colors[not_visible_mask, 0] * 0.5).astype(np.uint8)
        colors[not_visible_mask, 1] = (colors[not_visible_mask, 1] * 0.5).astype(np.uint8)
        colors[not_visible_mask, 2] = (colors[not_visible_mask, 2] * 0.55).astype(np.uint8)

        # Make NaN and zero pixels transparent
        colors[transparent_mask, 3] = 0

        # Draw observer marker
        px, py = coords_to_pixel(vsw, vsh, x_coords, y_coords)
        draw_observer_marker(colors, px, py, radius=4)

        # Draw legend
        draw_legend(colors, x=10, y=10)

        frames.append(Image.fromarray(colors, mode='RGBA'))

        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx + 1}/{num_frames}")

    return frames


def create_gif(frames, output_path, fps=12, max_colors=64):
    """Create a GIF from frames.

    Parameters
    ----------
    frames : list of PIL.Image
        The frames to combine.
    output_path : Path
        Output path for the GIF.
    fps : int
        Frames per second.
    max_colors : int
        Maximum colors in palette for smaller file size.
    """
    duration = int(1000 / fps)  # Duration in milliseconds

    # Use a magenta color as the transparency key (unlikely to appear in terrain)
    transparent_color = (255, 0, 255)

    # Convert RGBA frames to RGB, replacing transparent pixels with the key color
    print("Converting frames for GIF transparency...")
    rgb_frames = []
    for frame in frames:
        arr = np.array(frame)
        rgb = arr[:, :, :3].copy()
        alpha = arr[:, :, 3]
        # Set transparent pixels to the key color
        rgb[alpha == 0] = transparent_color
        rgb_frames.append(Image.fromarray(rgb, mode='RGB'))

    # Create global palette from sampled frames to avoid flickering
    print(f"Building global palette from {len(rgb_frames)} frames...")
    sample_step = max(1, len(rgb_frames) // 10)
    sampled = [np.array(rgb_frames[i]) for i in range(0, len(rgb_frames), sample_step)]
    combined = np.concatenate([p.reshape(-1, 3) for p in sampled], axis=0)

    h, w = np.array(rgb_frames[0]).shape[:2]
    sample_h = int(np.ceil(len(combined) / w))
    padded = np.zeros((sample_h * w, 3), dtype=np.uint8)
    padded[:len(combined)] = combined
    palette_img = Image.fromarray(padded.reshape(sample_h, w, 3), mode='RGB')
    global_palette = palette_img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)

    # Modify palette to reserve index 0 for transparency
    palette_data = list(global_palette.getpalette())
    palette_data[0:3] = transparent_color  # Force index 0 to be transparent color
    global_palette.putpalette(palette_data)
    transparency_index = 0

    # Quantize all frames using the global palette
    print(f"Quantizing frames to {max_colors} colors...")
    quantized_frames = []
    for frame in rgb_frames:
        q_frame = frame.quantize(palette=global_palette, dither=Image.Dither.FLOYDSTEINBERG)
        quantized_frames.append(q_frame)

    print(f"Creating GIF at {output_path}...")
    save_kwargs = {
        'save_all': True,
        'append_images': quantized_frames[1:],
        'duration': duration,
        'loop': 0,  # Loop forever
        'optimize': True
    }
    if transparency_index is not None:
        save_kwargs['transparency'] = transparency_index
        save_kwargs['disposal'] = 2  # Restore to background
        print(f"  Using transparency index: {transparency_index}")

    quantized_frames[0].save(output_path, **save_kwargs)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"GIF created: {output_path} ({file_size:.1f} MB)")


def main():
    output_path = Path(__file__).parent / "images" / "playground_demo.gif"
    output_path.parent.mkdir(exist_ok=True)

    terrain = load_terrain()

    # Generate 120 frames - both hillshade and hiker complete exactly one 360° loop
    # At 15fps this gives an 8 second loop that repeats seamlessly
    frames = generate_frames(terrain, num_frames=120)

    create_gif(frames, output_path, fps=6, max_colors=128)

    print(f"\nDone! GIF saved to: {output_path}")


if __name__ == "__main__":
    main()
