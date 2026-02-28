"""Camera state and projection helpers for the interactive viewer."""

import numpy as np


class CameraState:
    """Camera position, orientation, and projection parameters.

    Encapsulates all camera-related state: position, yaw/pitch, FOV,
    movement/look speeds, and time-of-day presets for sun positioning.
    """

    __slots__ = (
        'position', 'yaw', 'pitch', 'fov',
        'move_speed', 'look_speed',
        '_time_presets', '_time_preset_idx',
    )

    def __init__(self):
        self.position = None
        self.yaw = 90.0       # Degrees, 0 = +X, 90 = +Y
        self.pitch = -15.0    # Degrees, negative = looking down
        self.fov = 60.0
        self.move_speed = None  # Set in run() based on terrain extent
        self.look_speed = 5.0
        self._time_presets = [
            ('Morning',     135.0, 25.0),
            ('Midday',      180.0, 65.0),
            ('Afternoon',   225.0, 35.0),
            ('Golden Hour', 270.0, 12.0),
            ('Sunset',      280.0,  3.0),
        ]
        self._time_preset_idx = 2  # Afternoon (default)

    def get_front(self):
        """Get the forward direction vector."""
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        return np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)

    def get_right(self):
        """Get the right direction vector."""
        front = self.get_front()
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, front)
        return right / (np.linalg.norm(right) + 1e-8)

    def get_look_at(self):
        """Get the current look-at point."""
        return self.position + self.get_front() * 1000.0

    def screen_to_ray(self, screen_x, screen_y, render_width, render_height,
                      display_width, display_height):
        """Convert screen pixel coordinates to a world-space ray.

        Returns (origin, direction) as numpy float32 arrays of shape (3,).
        """
        front = self.get_front()
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, front)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right /= rn
        else:
            right = np.array([1, 0, 0], dtype=np.float32)
        cam_up = np.cross(front, right)

        fov_scale = np.tan(np.radians(self.fov) / 2.0)
        aspect = render_width / max(1, render_height)

        # Window coords -> NDC  (-1..1)
        nx = 2.0 * screen_x / max(1, display_width) - 1.0
        ny = 1.0 - 2.0 * screen_y / max(1, display_height)

        direction = front + nx * fov_scale * aspect * right + ny * fov_scale * cam_up
        direction = direction / (np.linalg.norm(direction) + 1e-30)
        return self.position.copy(), direction.astype(np.float32)
