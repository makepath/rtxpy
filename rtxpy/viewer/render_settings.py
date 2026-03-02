"""Render settings for the interactive viewer."""


class RenderSettings:
    """Rendering parameters: lighting, colormap, AO, denoiser, DOF.

    Groups all render-quality knobs into a single object so they can
    be adjusted, serialised, or reset independently of the viewer.
    """

    __slots__ = (
        'shadows', 'ambient',
        'sun_azimuth', 'sun_altitude',
        'fog_density', 'fog_color',
        'colormap', 'colormaps', 'colormap_idx',
        'color_stretch', '_color_stretches', '_color_stretch_idx',
        'ao_enabled', 'ao_radius', 'gi_intensity', 'gi_bounces',
        '_ao_samples_per_frame', '_ao_max_frames', '_ao_frame_count',
        '_d_ao_accum', '_prev_cam_state',
        'edl_enabled',
        'denoise_enabled', '_prev_cam_for_flow', '_d_flow',
        'dof_enabled', '_dof_aperture', '_dof_focal_distance',
    )

    def __init__(self):
        self.shadows = True
        self.ambient = 0.2
        self.sun_azimuth = 225.0
        self.sun_altitude = 35.0
        self.fog_density = 0.0
        self.fog_color = (0.7, 0.8, 0.9)
        self.colormap = 'gray'
        self.colormaps = ['gray', 'terrain', 'viridis', 'plasma', 'cividis']
        self.colormap_idx = 0
        self.color_stretch = 'linear'
        self._color_stretches = ['linear', 'sqrt', 'cbrt', 'log']
        self._color_stretch_idx = 0

        # Ambient occlusion
        self.ao_enabled = True
        self.ao_radius = None  # auto-computed from scene extent
        self.gi_intensity = 2.0
        self.gi_bounces = 1
        self._ao_samples_per_frame = 4
        self._ao_max_frames = 32
        self._ao_frame_count = 0
        self._d_ao_accum = None
        self._prev_cam_state = None

        # Eye Dome Lighting
        self.edl_enabled = True

        # Denoiser
        self.denoise_enabled = True
        self._prev_cam_for_flow = None
        self._d_flow = None

        # Depth of field
        self.dof_enabled = False
        self._dof_aperture = 20.0
        self._dof_focal_distance = 1000.0

    def reset_accumulation(self):
        """Reset AO/DOF accumulation state.

        Call this whenever the camera moves, lighting changes, or any
        render parameter changes that invalidates accumulated samples.
        """
        self._d_ao_accum = None
        self._ao_frame_count = 0
        self._prev_cam_state = None
