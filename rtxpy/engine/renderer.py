"""Rendering pipeline methods for the interactive viewer."""

import time

import numpy as np

from ..rtx import has_cupy

if has_cupy:
    import cupy as cp


class FrameRenderer:
    """Methods for rendering frames, compositing overlays, and display.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    def _save_screenshot(self):
        """Save current view as PNG image.

        When AO is enabled, renders multiple accumulated frames for
        high-quality output with smooth AA, soft shadows, AO, and DOF.
        """
        v = self.v
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rtxpy_screenshot_{timestamp}.png"

        # Pass viewshed data directly to render if enabled
        viewshed_data = None
        observer_pos = None
        active_obs = (v._observers.get(v._active_observer)
                      if v._active_observer else None)
        if active_obs is not None and active_obs.position is not None:
            observer_pos = active_obs.position
        if v.viewshed_enabled and v._viewshed_cache is not None:
            viewshed_data = v._viewshed_cache

        # Basemap texture for screenshot
        rgb_texture = None
        _tex_off_y = 0
        _tex_off_x = 0
        if (v._texture_tile_mgr is not None
                and v.lod_enabled
                and v._terrain_lod_manager is not None):
            visible = set(v._terrain_lod_manager._tile_lods.keys())
            d_tex, tex_r, tex_c = v._texture_tile_mgr.get_composite(
                visible)
            if d_tex is not None:
                rgb_texture = d_tex
                _tex_off_y = tex_r
                _tex_off_x = tex_c
        elif v._tiles_enabled and v._tile_service is not None:
            rgb_texture = v._tile_service.get_gpu_texture()
            if rgb_texture is not None and v.subsample_factor > 1:
                f = v.subsample_factor
                rgb_texture = rgb_texture[::f, ::f, :]

        # Build geometry colors GPU LUT if a builder is available
        geometry_colors = None
        builder = getattr(v, '_geometry_colors_builder', None)
        if builder is not None:
            geometry_colors = builder()

        from ..analysis import render as render_func

        # Common render kwargs
        # fog_density is scene-relative; convert to absolute for the kernel
        _fog = v.fog_density / v._scene_diagonal if v.fog_density > 0 else 0.0

        # Cloud fog map for screenshot — reuse cached GPU array
        _cloud_fog_map = None
        _cloud_fog_density = 0.0
        if v._clouds_enabled and v._cloud_cover_grid is not None:
            _d = getattr(v, '_d_cloud_fog_map', None)
            f = v.subsample_factor
            if _d is None or getattr(v, '_cloud_fog_subsample', 0) != f:
                src = v._cloud_cover_grid
                if f > 1:
                    src = src[::f, ::f]
                v._d_cloud_fog_map = cp.asarray(src, dtype=cp.float32)
                v._cloud_fog_subsample = f
            _cloud_fog_map = v._d_cloud_fog_map
            _cloud_fog_density = 12.0 / v._scene_diagonal

        # Resolve overlay: per-tile composite when LOD active,
        # else monolithic array.  Used by both screenshot and live paths.
        _ov_data = v._active_overlay_data
        _ov_lut = v._active_overlay_color_lut
        _ov_off_y = 0
        _ov_off_x = 0
        if (v._overlay_tile_mgr is not None
                and v._active_overlay_data is not None
                and v.lod_enabled
                and v._terrain_lod_manager is not None):
            visible = set(v._terrain_lod_manager._tile_lods.keys())
            d_comp, off_r, off_c = v._overlay_tile_mgr.get_composite(
                visible)
            if d_comp is not None:
                _ov_data = d_comp
                _ov_off_y = off_r
                _ov_off_x = off_c
                lut = v._overlay_tile_mgr.color_lut
                if lut is not None:
                    _ov_lut = lut

        render_kwargs = dict(
            camera_position=tuple(v.position),
            look_at=tuple(v._get_look_at()),
            fov=v.fov,
            width=v.width,
            height=v.height,
            sun_azimuth=v.sun_azimuth,
            sun_altitude=v.sun_altitude,
            shadows=v.shadows,
            ambient=v.ambient,
            fog_density=_fog,
            fog_color=v.fog_color,
            colormap=v.colormap,
            rtx=v.rtx,
            viewshed_data=viewshed_data,
            viewshed_opacity=v.viewshed_opacity,
            observer_position=observer_pos,
            pixel_spacing_x=v.pixel_spacing_x,
            pixel_spacing_y=v.pixel_spacing_y,
            color_stretch=v.color_stretch,
            color_range=v._land_color_range,
            rgb_texture=rgb_texture,
            rgb_texture_offset_y=_tex_off_y,
            rgb_texture_offset_x=_tex_off_x,
            overlay_data=_ov_data,
            overlay_alpha=v._overlay_alpha,
            overlay_as_water=v._overlay_as_water,
            overlay_color_lut=_ov_lut,
            overlay_offset_y=_ov_off_y,
            overlay_offset_x=_ov_off_x,
            geometry_colors=geometry_colors,
            cloud_fog_map=_cloud_fog_map,
            cloud_fog_density=_cloud_fog_density,
            volumetric_clouds=v._volumetric_clouds_enabled and v._clouds_enabled,
            cloud_base_z=v._cloud_altitude,
            cloud_top_z=v._cloud_altitude + v._cloud_thickness,
            cloud_time=v._cloud_time,
        )

        # Accumulated multi-frame screenshot when AO or DOF is enabled
        num_frames = 64 if (v.ao_enabled or v.dof_enabled) else 1

        if num_frames > 1:
            import cupy
            from ..analysis.render import _bloom, _tone_map_aces, _render_buffers
            print(f"Rendering {num_frames} accumulated frames...", end='', flush=True)

            # DOF params
            if v.dof_enabled:
                dof_aperture = v._dof_aperture
                dof_focal = v._dof_focal_distance
            else:
                dof_aperture = 0.0
                dof_focal = 0.0

            d_accum = None
            for i in range(num_frames):
                frame_seed = i + 1
                d_frame = render_func(
                    v.raster,
                    **render_kwargs,
                    ao_samples=v._ao_samples_per_frame,
                    ao_radius=v.ao_radius,
                    ao_seed=i,
                    gi_intensity=v.gi_intensity,
                    gi_bounces=v.gi_bounces,
                    frame_seed=frame_seed,
                    sun_angle=1.5,
                    aperture=dof_aperture,
                    focal_distance=dof_focal,
                    bloom=False,
                    tone_map=False,
                    _return_gpu=True,
                )
                if d_accum is None:
                    d_accum = d_frame.astype(cupy.float32)
                else:
                    d_accum += d_frame
            d_accum /= num_frames

            # Apply bloom and tone mapping once to the averaged result
            bufs = _render_buffers
            if bufs.bloom_temp is not None:
                _bloom(d_accum, bufs.bloom_temp, bufs.bloom_scratch)
            _tone_map_aces(d_accum)

            img = cupy.asnumpy(d_accum)
            print(" done")
        else:
            img = render_func(v.raster, **render_kwargs)

        # Convert from float [0-1] to uint8 [0-255]
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        from PIL import Image
        Image.fromarray(img_uint8).save(filename)

        print(f"Screenshot saved: {filename}")

    def _render_frame(self):
        """Render a frame using rtxpy."""
        v = self.v
        from ..analysis import render

        # Always show observer orb when placed; viewshed overlay only when enabled
        viewshed_data = None
        observer_pos = None
        active_obs = (v._observers.get(v._active_observer)
                      if v._active_observer else None)
        if active_obs is not None and active_obs.position is not None:
            observer_pos = active_obs.position
        if v.viewshed_enabled:
            if v._viewshed_cache is not None:
                viewshed_data = v._viewshed_cache
            else:
                # Debug: viewshed enabled but no cache
                if v.frame_count % 100 == 0:  # Only print occasionally
                    print(f"[DEBUG] Viewshed enabled but cache is None")

        # Basemap texture — per-tile composite when LOD active,
        # else monolithic GPU texture from tile service.
        rgb_texture = None
        _tex_off_y = 0
        _tex_off_x = 0
        if (v._texture_tile_mgr is not None
                and v.lod_enabled
                and v._terrain_lod_manager is not None):
            visible = set(v._terrain_lod_manager._tile_lods.keys())
            d_tex, tex_r, tex_c = v._texture_tile_mgr.get_composite(
                visible)
            if d_tex is not None:
                rgb_texture = d_tex
                _tex_off_y = tex_r
                _tex_off_x = tex_c
        elif v._tiles_enabled and v._tile_service is not None:
            rgb_texture = v._tile_service.get_gpu_texture()
            if rgb_texture is not None and v.subsample_factor > 1:
                f = v.subsample_factor
                rgb_texture = rgb_texture[::f, ::f, :]

        # Build geometry colors GPU LUT if a builder is available
        geometry_colors = None
        builder = getattr(v, '_geometry_colors_builder', None)
        if builder is not None:
            geometry_colors = builder()

        # Progressive accumulation is needed for AO convergence and/or
        # DOF (thin-lens jitter needs multi-frame averaging to converge).
        needs_accum = v.ao_enabled or v.dof_enabled

        # AO parameters: multiple samples per frame for smooth early results,
        # with progressive accumulation across frames for further refinement
        ao_samples = v._ao_samples_per_frame if v.ao_enabled else 0
        ao_seed = v._ao_frame_count if v.ao_enabled else 0

        # When progressive accumulation is active, pass frame seed for AA + soft shadows + DOF
        frame_seed = v._ao_frame_count + 1 if needs_accum else 0

        # Depth of field
        if v.dof_enabled:
            dof_aperture = v._dof_aperture
            dof_focal = v._dof_focal_distance
        else:
            dof_aperture = 0.0
            dof_focal = 0.0

        # When progressive AO accumulation or denoising is active, defer
        # bloom and tone mapping until after averaging / denoising.  Both
        # are non-linear operations that must act on the clean signal.
        defer_post = v.ao_enabled or v.denoise_enabled

        # fog_density is scene-relative; convert to absolute for the kernel
        _fog = v.fog_density / v._scene_diagonal if v.fog_density > 0 else 0.0

        # Cloud fog map: pass cloud_cover_grid for cloud shadow modulation
        _cloud_fog_map = None
        _cloud_fog_density = 0.0
        if v._clouds_enabled and v._cloud_cover_grid is not None:
            # Cache GPU array to avoid per-frame upload
            _d = getattr(v, '_d_cloud_fog_map', None)
            f = v.subsample_factor
            if _d is None or getattr(v, '_cloud_fog_subsample', 0) != f:
                src = v._cloud_cover_grid
                if f > 1:
                    src = src[::f, ::f]
                v._d_cloud_fog_map = cp.asarray(src, dtype=cp.float32)
                v._cloud_fog_subsample = f
            _cloud_fog_map = v._d_cloud_fog_map
            # Spatial frequency: ~12 cloud cells across scene
            _cloud_fog_density = 12.0 / v._scene_diagonal

        # Resolve overlay: per-tile composite when LOD active,
        # else monolithic array.
        _ov_data = v._active_overlay_data
        _ov_lut = v._active_overlay_color_lut
        _ov_off_y = 0
        _ov_off_x = 0
        if (v._overlay_tile_mgr is not None
                and v._active_overlay_data is not None
                and v.lod_enabled
                and v._terrain_lod_manager is not None):
            visible = set(v._terrain_lod_manager._tile_lods.keys())
            d_comp, off_r, off_c = v._overlay_tile_mgr.get_composite(
                visible)
            if d_comp is not None:
                _ov_data = d_comp
                _ov_off_y = off_r
                _ov_off_x = off_c
                lut = v._overlay_tile_mgr.color_lut
                if lut is not None:
                    _ov_lut = lut

        d_output = render(
            v.raster,
            camera_position=tuple(v.position),
            look_at=tuple(v._get_look_at()),
            fov=v.fov,
            width=v.render_width,
            height=v.render_height,
            sun_azimuth=v.sun_azimuth,
            sun_altitude=v.sun_altitude,
            shadows=v.shadows,
            ambient=v.ambient,
            fog_density=_fog,
            fog_color=v.fog_color,
            colormap=v.colormap,
            rtx=v.rtx,
            viewshed_data=viewshed_data,
            viewshed_opacity=v.viewshed_opacity,
            observer_position=observer_pos,
            pixel_spacing_x=v.pixel_spacing_x,
            pixel_spacing_y=v.pixel_spacing_y,
            mesh_type='heightfield',
            color_data=v._active_color_data,
            color_stretch=v.color_stretch,
            color_range=v._land_color_range,
            rgb_texture=rgb_texture,
            rgb_texture_offset_y=_tex_off_y,
            rgb_texture_offset_x=_tex_off_x,
            overlay_data=_ov_data,
            overlay_alpha=v._overlay_alpha,
            overlay_as_water=v._overlay_as_water,
            overlay_color_lut=_ov_lut,
            overlay_offset_y=_ov_off_y,
            overlay_offset_x=_ov_off_x,
            geometry_colors=geometry_colors,
            ao_samples=ao_samples,
            ao_radius=v.ao_radius,
            ao_seed=ao_seed,
            gi_intensity=v.gi_intensity,
            gi_bounces=v.gi_bounces,
            frame_seed=frame_seed,
            sun_angle=1.5,
            aperture=dof_aperture,
            focal_distance=dof_focal,
            edge_strength=0.2,
            edge_color=(0.15, 0.13, 0.10),
            edl=v.edl_enabled,
            cloud_fog_map=_cloud_fog_map,
            cloud_fog_density=_cloud_fog_density,
            volumetric_clouds=v._volumetric_clouds_enabled and v._clouds_enabled,
            cloud_base_z=v._cloud_altitude,
            cloud_top_z=v._cloud_altitude + v._cloud_thickness,
            cloud_time=v._cloud_time,
            bloom=not defer_post,
            tone_map=not defer_post,
            _return_gpu=True,
        )

        return d_output

    def _update_frame(self):
        """Full render: GPU ray trace -> D2H copy -> overlays -> display."""
        v = self.v
        # Sync previous frame's async D2H copy (no-op on first frame)
        v._readback_stream.synchronize()

        # GPU render — returns cupy array (no D2H copy)
        d_output = self._render_frame()
        v.frame_count += 1

        # Extract depth buffer from primary hits for hydro occlusion culling.
        # primary_hits[:, 0] holds ray t-values; persists until next render.
        if v._hydro_enabled:
            from ..analysis.render import _render_buffers as _depth_bufs
            if _depth_bufs.primary_hits is not None:
                h, w = v.render_height, v.render_width
                _ddt = getattr(v, '_d_depth_t', None)
                if _ddt is None or _ddt.shape != (h, w):
                    v._d_depth_t = cp.empty((h, w), dtype=cp.float32)
                t_src = _depth_bufs.primary_hits.reshape(h * w, 4)[:, 0]
                v._d_depth_t[:] = t_src.reshape(h, w)

        # Progressive accumulation (needed for AO convergence and/or DOF)
        needs_accum = v.ao_enabled or v.dof_enabled
        if needs_accum:
            from ..analysis.render import _bloom, _tone_map_aces, _render_buffers

            # Check if camera moved — compare current state to previous
            cam_state = (tuple(v.position), v.yaw, v.pitch, v.fov)
            if v._prev_cam_state != cam_state:
                # Camera moved: reset accumulation
                v._d_ao_accum = None
                v._ao_frame_count = 0
                v._prev_cam_state = cam_state

            # Accumulate
            if v._d_ao_accum is None or v._d_ao_accum.shape != d_output.shape:
                v._d_ao_accum = d_output.copy()
            else:
                v._d_ao_accum += d_output
            v._ao_frame_count += 1

            # Average the accumulated frames
            d_display = v._d_ao_accum / v._ao_frame_count
        else:
            d_display = d_output

        # Deferred post-processing: denoise -> bloom -> tone map.
        # These are deferred when AO, DOF, or denoiser is active so they
        # operate on the clean / averaged signal.
        defer_post = needs_accum or v.denoise_enabled
        if defer_post:
            if not needs_accum:
                from ..analysis.render import _bloom, _tone_map_aces, _render_buffers

            if v.denoise_enabled:
                from ..rtx import denoise as _denoise
                from ..analysis.render import (
                    _compute_camera_basis, _render_buffers as _bufs,
                    compute_flow,
                )
                h, w = v.render_height, v.render_width
                d_normals = _bufs.primary_hits.reshape(h, w, 4)[:, :, 1:4].copy()
                ve = v.vertical_exaggeration
                pos = v.position
                look = v._get_look_at()
                scaled_pos = (pos[0], pos[1], pos[2] * ve)
                scaled_look = (look[0], look[1], look[2] * ve)
                forward, right, cam_up = _compute_camera_basis(
                    scaled_pos, scaled_look, (0, 0, 1))

                # Compute flow vectors for temporal denoising
                d_flow = None
                aspect = w / h
                fov_scale = np.tan(np.radians(v.fov) / 2.0)
                if v._prev_cam_for_flow is not None:
                    prev_pos, prev_fwd, prev_right, prev_up, prev_aspect, prev_fov_scale = v._prev_cam_for_flow
                    # Allocate / resize flow buffer
                    if v._d_flow is None or v._d_flow.shape != (h, w, 2):
                        v._d_flow = cp.zeros((h, w, 2), dtype=cp.float32)
                    d_prev_pos = cp.asarray(np.array(prev_pos, dtype=np.float32))
                    d_prev_fwd = cp.asarray(np.array(prev_fwd, dtype=np.float32))
                    d_prev_right = cp.asarray(np.array(prev_right, dtype=np.float32))
                    d_prev_up = cp.asarray(np.array(prev_up, dtype=np.float32))
                    compute_flow(
                        v._d_flow, _bufs.primary_rays, _bufs.primary_hits,
                        w, h,
                        d_prev_pos, d_prev_fwd, d_prev_right, d_prev_up,
                        prev_aspect, prev_fov_scale,
                    )
                    d_flow = v._d_flow

                v._prev_cam_for_flow = (
                    scaled_pos, tuple(forward), tuple(right), tuple(cam_up),
                    aspect, fov_scale,
                )

                _denoise(d_display, d_normals, w, h, right, cam_up, forward,
                         albedo=_bufs.albedo, flow=d_flow)

            bufs = _render_buffers
            if bufs.bloom_temp is not None:
                _bloom(d_display, bufs.bloom_temp, bufs.bloom_scratch)
            _tone_map_aces(d_display)

        # Save clean post-processed frame for idle wind/hydro/rain replay
        _any_particles = (v._wind_enabled or v._hydro_enabled
                          or (v._clouds_enabled and v._rain_particles is not None))
        if _any_particles:
            if v._d_base_frame is None or v._d_base_frame.shape != d_display.shape:
                v._d_base_frame = cp.empty_like(d_display)
                v._d_wind_scratch = cp.empty_like(d_display)
            cp.copyto(v._d_base_frame, d_display)

        # Advance volumetric cloud animation time
        if v._clouds_enabled and v._volumetric_clouds_enabled:
            v._cloud_time += 0.05

        if v._interop_enabled:
            # --- CUDA-GL interop path: zero-copy GPU->GL ---
            try:
                d_pbo = v._cuda_gl_buf.map()
                try:
                    # Copy post-processed frame into PBO
                    cp.copyto(d_pbo, d_display)

                    # Splat particles directly into PBO memory
                    if v._wind_enabled and v._wind_particles is not None:
                        v._update_wind_particles()
                        v._splat_wind_gpu(d_pbo)
                    if v._hydro_enabled and v._hydro_particles is not None:
                        v.hydro_mgr.check_streaming_result()
                        v._transfer_streaming_overlay()
                        cam_r = v.position[1] / v._base_pixel_spacing_y
                        cam_c = v.position[0] / v._base_pixel_spacing_x
                        v.hydro_mgr.update_streaming_window(cam_r, cam_c)
                        v._update_hydro_particles()
                        v._splat_hydro_gpu(d_pbo)
                    if v._clouds_enabled and v._rain_particles is not None:
                        v._update_rain_particles()
                        v._splat_rain_gpu(d_pbo)

                    # No synchronize needed — unmap is stream-ordered
                finally:
                    v._cuda_gl_buf.unmap()

                # GPU-internal PBO->texture upload (<0.5ms)
                v._cuda_gl_buf.upload_to_texture(v._interop_frame_tex)

                # Build overlay RGBA (CPU) — only re-uploaded when content changes
                self._build_overlay_rgba_cached()
                self._update_window_title()
                v._frame_dirty = True
            except Exception:
                # Interop failed mid-frame — fall back to CPU path for this
                # frame and disable interop for subsequent frames.
                import traceback
                traceback.print_exc()
                print("CUDA-GL interop error - falling back to CPU path")
                v._interop_enabled = False
                self._update_frame_cpu(d_display, _any_particles)
        else:
            self._update_frame_cpu(d_display, _any_particles)

    def _update_frame_cpu(self, d_display, _any_particles):
        """CPU fallback display path: GPU->CPU readback + overlay compositing."""
        v = self.v
        # Allocate pinned host buffer lazily (or on shape change)
        if v._pinned_frame is None or v._pinned_frame.shape != d_display.shape:
            v._pinned_mem = cp.cuda.alloc_pinned_memory(d_display.nbytes)
            v._pinned_frame = np.frombuffer(
                v._pinned_mem, dtype=np.float32, count=d_display.size
            ).reshape(d_display.shape)

        # GPU wind: advect on CPU, splat on GPU, then readback
        if v._wind_enabled and v._wind_particles is not None:
            v._update_wind_particles()
            v._splat_wind_gpu(d_display)

        # GPU hydro: advect + splat on GPU
        if v._hydro_enabled and v._hydro_particles is not None:
            v.hydro_mgr.check_streaming_result()
            v._transfer_streaming_overlay()
            cam_r = v.position[1] / v._base_pixel_spacing_y
            cam_c = v.position[0] / v._base_pixel_spacing_x
            v.hydro_mgr.update_streaming_window(cam_r, cam_c)
            v._update_hydro_particles()
            v._splat_hydro_gpu(d_display)

        # GPU rain: advect on CPU, splat on GPU (cloud fog is baked into ray trace)
        if v._clouds_enabled and v._rain_particles is not None:
            v._update_rain_particles()
            v._splat_rain_gpu(d_display)

        # Sync: splat kernels run on stream 0, readback on non-blocking stream
        if _any_particles:
            sync_event = v._wind_done_event or v._hydro_done_event
            if sync_event is None:
                sync_event = cp.cuda.Event()
                if v._wind_enabled:
                    v._wind_done_event = sync_event
                else:
                    v._hydro_done_event = sync_event
            sync_event.record()
            v._readback_stream.wait_event(sync_event)

        # Async D2H copy on non-blocking stream
        d_display.get(out=v._pinned_frame, stream=v._readback_stream)

        # Wait for DMA to complete
        v._readback_stream.synchronize()

        # Composite overlays on top of the ray-traced base frame
        self._composite_overlays()

    def _composite_overlays(self):
        """Composite CPU overlays (wind, minimap, help) onto the base frame.

        Can be called without re-ray-tracing to animate wind cheaply.
        """
        v = self.v
        # FPS tracking
        v._fps_counter += 1
        now = time.monotonic()
        elapsed = now - v._fps_last_time
        if elapsed >= 1.0:
            v._fps_display = v._fps_counter / elapsed
            v._fps_counter = 0
            v._fps_last_time = now

        # Build window title
        title = v._build_title()
        pos = v.position
        fps = v._fps_display
        sub = f"{fps:.0f} FPS  Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})  Speed: {v.move_speed:.0f}"
        if v._observers:
            obs_parts = []
            for slot in sorted(v._observers):
                obs = v._observers[slot]
                marker = '*' if slot == v._active_observer else ''
                mode = ''
                if obs.drone_mode != 'off':
                    mode = f' {obs.drone_mode.upper()}'
                if obs.is_touring():
                    mode += ' TOUR'
                obs_parts.append(f"{slot}{marker}{mode}")
            sub += f"  \u2502  Obs: [{' '.join(obs_parts)}]"
            active_obs = (v._observers.get(v._active_observer)
                          if v._active_observer else None)
            if active_obs is not None:
                sub += f"  h={active_obs.observer_elev:.3f}"
            if v.viewshed_enabled:
                sub += f"  Coverage: {v._viewshed_coverage:.1f}%"

        combined = f"{title}  |  {sub}"
        if combined != v._last_title:
            v._last_title = combined
            if v._glfw_window is not None:
                import glfw
                glfw.set_window_title(v._glfw_window, combined)

        # Build display frame (copy if we need overlays, else use pinned directly)
        _help_visible = (v._help_page_idx >= 0 and v._help_pages)
        needs_overlay = (
            (v._gtfs_rt_enabled and v._gtfs_rt_vehicles is not None)
            or v.show_minimap
            or v._title_overlay_rgba is not None
            or v._legend_rgba is not None
            or _help_visible
        )
        if needs_overlay:
            img = v._pinned_frame.copy()
        else:
            img = v._pinned_frame

        # GTFS-RT vehicle overlay
        if v._gtfs_rt_enabled and v._gtfs_rt_vehicles is not None:
            v._draw_gtfs_rt_on_frame(img)

        # Minimap overlay
        v._blit_minimap_on_frame(img)

        # Title overlay
        v._blit_title_on_frame(img)

        # Legend overlay (always visible)
        v._blit_legend_on_frame(img)

        # Help page overlay
        if _help_visible:
            v._blit_help_on_frame(img)

        v._display_frame = img
        v._frame_dirty = True

    def _update_window_title(self):
        """Update FPS counter and GLFW window title (shared by both paths)."""
        v = self.v
        v._fps_counter += 1
        now = time.monotonic()
        elapsed = now - v._fps_last_time
        if elapsed >= 1.0:
            v._fps_display = v._fps_counter / elapsed
            v._fps_counter = 0
            v._fps_last_time = now

        title = v._build_title()
        pos = v.position
        fps = v._fps_display
        sub = f"{fps:.0f} FPS  Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})  Speed: {v.move_speed:.0f}"
        if v._observers:
            obs_parts = []
            for slot in sorted(v._observers):
                obs = v._observers[slot]
                marker = '*' if slot == v._active_observer else ''
                mode = ''
                if obs.drone_mode != 'off':
                    mode = f' {obs.drone_mode.upper()}'
                if obs.is_touring():
                    mode += ' TOUR'
                obs_parts.append(f"{slot}{marker}{mode}")
            sub += f"  \u2502  Obs: [{' '.join(obs_parts)}]"
            active_obs = (v._observers.get(v._active_observer)
                          if v._active_observer else None)
            if active_obs is not None:
                sub += f"  h={active_obs.observer_elev:.3f}"
            if v.viewshed_enabled:
                sub += f"  Coverage: {v._viewshed_coverage:.1f}%"

        combined = f"{title}  |  {sub}"
        if combined != v._last_title:
            v._last_title = combined
            if v._glfw_window is not None:
                import glfw
                glfw.set_window_title(v._glfw_window, combined)

    def _build_overlay_rgba(self):
        """Build RGBA overlay for the interop path (transparent where empty).

        Sets ``v._overlay_rgba`` and bumps ``v._overlay_gen`` when
        content changes.  Returns None if nothing to draw.
        """
        v = self.v
        h, w = v.render_height, v.render_width

        _help_visible = (v._help_page_idx >= 0 and v._help_pages)
        needs_overlay = (
            (v._gtfs_rt_enabled and v._gtfs_rt_vehicles is not None)
            or v.show_minimap
            or v._title_overlay_rgba is not None
            or v._legend_rgba is not None
            or _help_visible
        )
        if not needs_overlay:
            if v._overlay_rgba is not None:
                v._overlay_rgba = None
                v._overlay_gen += 1
            return

        # Build a float32 RGB "canvas" to blit overlays onto, then convert to
        # RGBA.  We reuse the existing _blit_*_on_frame helpers which write
        # into an (H, W, 3) float32 array.  We composite with alpha by
        # tracking which pixels were written.
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)

        # Use a thin wrapper that also sets the alpha mask
        def _blit_with_mask(blit_fn, canvas, mask):
            before = canvas.copy()
            blit_fn(canvas)
            changed = np.any(canvas != before, axis=2)
            mask[changed] = 1.0

        if v._gtfs_rt_enabled and v._gtfs_rt_vehicles is not None:
            _blit_with_mask(v._draw_gtfs_rt_on_frame, canvas, mask)
        _blit_with_mask(v._blit_minimap_on_frame, canvas, mask)
        _blit_with_mask(v._blit_title_on_frame, canvas, mask)
        _blit_with_mask(v._blit_legend_on_frame, canvas, mask)
        if _help_visible:
            _blit_with_mask(v._blit_help_on_frame, canvas, mask)

        if mask.any():
            rgba = np.empty((h, w, 4), dtype=np.float32)
            rgba[:, :, :3] = canvas
            rgba[:, :, 3] = mask
            v._overlay_rgba = rgba
            v._overlay_gen += 1
        else:
            if v._overlay_rgba is not None:
                v._overlay_rgba = None
                v._overlay_gen += 1

    def _overlay_input_signature(self):
        """Return a lightweight signature of all overlay input state."""
        v = self.v
        pos = (round(v.position[0], 1), round(v.position[1], 1),
               round(v.position[2], 1)) if hasattr(v, 'position') else None
        return (
            id(v._gtfs_rt_vehicles) if v._gtfs_rt_enabled else 0,
            v.show_minimap,
            pos if v.show_minimap else None,
            id(v._title_overlay_rgba),
            id(v._legend_rgba),
            v._help_page_idx,
            v.render_width,
            v.render_height,
        )

    def _build_overlay_rgba_cached(self):
        """Skip overlay rebuild if inputs haven't changed since last build."""
        v = self.v
        sig = self._overlay_input_signature()
        if sig == getattr(v, '_overlay_last_sig', None):
            return
        self._build_overlay_rgba()
        v._overlay_last_sig = sig

    def _present_if_dirty(self, frame_tex, prog, ctx, vao,
                          glfw, window, moderngl):
        """Upload frame to GL texture and present if the frame changed."""
        v = self.v
        if not v._frame_dirty:
            return

        if v._interop_enabled:
            # Interop path: frame_tex already updated via PBO upload.
            # Handle overlay texture upload if needed.
            has_ov = (v._overlay_rgba is not None
                      and v._overlay_gen != v._overlay_uploaded)
            if has_ov:
                ov_w, ov_h = v._overlay_tex.size
                oh, ow = v._overlay_rgba.shape[:2]
                if ow != ov_w or oh != ov_h:
                    v._overlay_tex.release()
                    v._overlay_tex = ctx.texture((ow, oh), 4, dtype='f4')
                    v._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                v._overlay_tex.write(v._overlay_rgba)
                v._overlay_uploaded = v._overlay_gen

            prog['has_overlay'].value = 1 if v._overlay_rgba is not None else 0
            v._interop_frame_tex.use(location=0)
            v._overlay_tex.use(location=1)
            ctx.clear()
            vao.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)
            v._frame_dirty = False
        else:
            # Fallback path: frame composited on CPU, upload to texture.
            if v._display_frame is not None:
                tex_w, tex_h = frame_tex.size
                fh, fw = v._display_frame.shape[:2]
                if fw != tex_w or fh != tex_h:
                    frame_tex.release()
                    frame_tex = ctx.texture((fw, fh), 3, dtype='f4')
                    frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                prog['has_overlay'].value = 0
                frame_tex.write(v._display_frame)
                frame_tex.use(location=0)
                ctx.clear()
                vao.render(moderngl.TRIANGLE_STRIP)
                glfw.swap_buffers(window)
                v._frame_dirty = False
