"""Input handling methods for the interactive viewer."""

import numpy as np

from ..viewer.keybindings import (
    MOVEMENT_KEYS, SHIFT_BINDINGS, KEY_BINDINGS, SPECIAL_BINDINGS,
)


class InputHandler:
    """Methods for keyboard, mouse, and scroll input handling.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    # ------------------------------------------------------------------
    # Action methods (thin wrappers dispatched by key tables)
    # ------------------------------------------------------------------

    def _action_shift_o(self):
        v = self.v
        obs = v._observers.get(v._active_observer) if v._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 first.")
        else:
            v._cycle_drone_mode_for(obs)

    def _action_shift_v(self):
        v = self.v
        obs = v._observers.get(v._active_observer) if v._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 first.")
        else:
            v._snap_to_observer(obs)

    def _action_clear_observers(self):
        v = self.v
        v._clear_all_observers()

    def _action_toggle_firms(self):
        v = self.v
        v._toggle_firms()

    def _action_toggle_wind(self):
        v = self.v
        v._toggle_wind()

    def _action_toggle_terrain_vis(self):
        v = self.v
        from ..viewer.terrain_lod import is_terrain_lod_gid
        # Toggle all LOD terrain tiles together
        vis = None
        for gid in v.rtx.list_geometries():
            if is_terrain_lod_gid(gid):
                if vis is None:
                    e = v.rtx._geom_state.gas_entries.get(gid)
                    vis = not e.visible if e is not None else True
                v.rtx.set_geometry_visible(gid, vis)
        if vis is not None:
            print(f"Terrain {'shown' if vis else 'hidden'}")
            v._render_needed = True

    def _action_toggle_gtfs_rt(self):
        v = self.v
        v._toggle_gtfs_rt()

    def _action_cycle_pc_colors(self):
        v = self.v
        v._cycle_pointcloud_colors()

    def _action_toggle_denoiser(self):
        v = self.v
        v.denoise_enabled = not v.denoise_enabled
        v.render_settings.reset_accumulation()
        v._prev_cam_for_flow = None
        print(f"Denoiser: {'ON' if v.denoise_enabled else 'OFF'}")
        v._update_frame()

    def _action_cycle_gi(self):
        v = self.v
        v.gi_bounces = v.gi_bounces % 3 + 1
        v.render_settings.reset_accumulation()
        print(f"GI bounces: {v.gi_bounces}")
        v._update_frame()

    def _action_prev_help_page(self):
        v = self.v
        if v._help_pages:
            v._help_page_idx -= 1
            if v._help_page_idx < -1:
                v._help_page_idx = len(v._help_pages) - 1
        v._update_frame()

    def _action_toggle_drone_glow(self):
        v = self.v
        v._drone_glow = not v._drone_glow
        v._apply_drone_glow()
        print(f"Drone glow: {'ON' if v._drone_glow else 'OFF'}")

    def _action_cycle_time(self):
        v = self.v
        v._time_preset_idx = (v._time_preset_idx + 1) % len(v._time_presets)
        name, az, alt = v._time_presets[v._time_preset_idx]
        v.sun_azimuth = az
        v.sun_altitude = alt
        v.render_settings.reset_accumulation()
        print(f"Time of day: {name} (az={az:.0f}, alt={alt:.0f})")
        v._update_frame()

    def _action_toggle_shadows(self):
        v = self.v
        v.shadows = not v.shadows
        print(f"Shadows: {'ON' if v.shadows else 'OFF'}")
        v._update_frame()

    def _action_cycle_colormap(self):
        v = self.v
        v.colormap_idx = (v.colormap_idx + 1) % len(v.colormaps)
        v.colormap = v.colormaps[v.colormap_idx]
        print(f"Colormap: {v.colormap}")
        v._update_frame()

    def _action_jump_prev_geom(self):
        v = self.v
        v._jump_to_geometry(-1)

    def _action_next_help_page(self):
        v = self.v
        if v._help_pages:
            v._help_page_idx += 1
            if v._help_page_idx >= len(v._help_pages):
                v._help_page_idx = -1
        else:
            v._help_page_idx = -1
        v._update_frame()

    def _action_toggle_minimap(self):
        v = self.v
        v.show_minimap = not v.show_minimap
        v._update_frame()

    def _action_place_observer(self):
        v = self.v
        obs = v._observers.get(v._active_observer) if v._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 to create one.")
        elif obs.drone_mode == 'off':
            v._place_observer_at(obs)

    def _action_observer_elev_down(self):
        v = self.v
        v._adjust_observer_elevation(-0.01)

    def _action_observer_elev_up(self):
        v = self.v
        v._adjust_observer_elevation(0.01)

    def _action_cycle_color_stretch(self):
        v = self.v
        v._color_stretch_idx = (v._color_stretch_idx + 1) % len(v._color_stretches)
        v.color_stretch = v._color_stretches[v._color_stretch_idx]
        print(f"Color stretch: {v.color_stretch}")
        v._update_frame()

    def _action_cycle_basemap_fwd(self):
        v = self.v
        v._cycle_basemap()

    def _action_cycle_basemap_rev(self):
        v = self.v
        v._cycle_basemap(reverse=True)

    def _action_overlay_alpha_down(self):
        v = self.v
        v._overlay_alpha = max(0.0, round(v._overlay_alpha - 0.1, 1))
        print(f"Overlay alpha: {int(v._overlay_alpha * 100)}%")
        v._update_frame()

    def _action_overlay_alpha_up(self):
        v = self.v
        v._overlay_alpha = min(1.0, round(v._overlay_alpha + 0.1, 1))
        print(f"Overlay alpha: {int(v._overlay_alpha * 100)}%")
        v._update_frame()

    def _action_speed_up(self):
        v = self.v
        H, W = v.terrain_shape
        world_diag = np.sqrt((W * v.pixel_spacing_x)**2 + (H * v.pixel_spacing_y)**2)
        max_speed = world_diag * 0.1
        v.move_speed = min(max_speed, v.move_speed * 1.2)
        print(f"Speed: {v.move_speed:.3f}")

    def _action_speed_down(self):
        v = self.v
        v.move_speed = max(0.001, v.move_speed / 1.2)
        print(f"Speed: {v.move_speed:.3f}")

    def _action_resolution_coarser(self):
        v = self.v
        new_factor = min(8, v.subsample_factor * 2)
        if new_factor != v.subsample_factor:
            v._rebuild_at_resolution(new_factor)

    def _action_resolution_finer(self):
        v = self.v
        new_factor = max(1, v.subsample_factor // 2)
        if new_factor != v.subsample_factor:
            v._rebuild_at_resolution(new_factor)

    def _action_ve_down(self):
        v = self.v
        new_ve = max(0.1, round(v.vertical_exaggeration - 0.1, 1))
        if new_ve != v.vertical_exaggeration:
            v._rebuild_vertical_exaggeration(new_ve)

    def _action_ve_up(self):
        v = self.v
        new_ve = min(10.0, round(v.vertical_exaggeration + 0.1, 1))
        if new_ve != v.vertical_exaggeration:
            v._rebuild_vertical_exaggeration(new_ve)

    def _action_toggle_ao(self):
        v = self.v
        v.ao_enabled = not v.ao_enabled
        v.render_settings.reset_accumulation()
        print(f"Ambient Occlusion: {'ON' if v.ao_enabled else 'OFF'}")
        v._update_frame()

    def _action_toggle_dof(self):
        v = self.v
        v.dof_enabled = not v.dof_enabled
        v.render_settings.reset_accumulation()
        print(f"Depth of Field: {'ON' if v.dof_enabled else 'OFF'}")
        v._update_frame()

    def _action_dof_aperture_down(self):
        v = self.v
        v._dof_aperture = max(1.0, v._dof_aperture * 0.7)
        v.render_settings.reset_accumulation()
        print(f"DOF aperture: {v._dof_aperture:.1f}")
        v._update_frame()

    def _action_dof_aperture_up(self):
        v = self.v
        v._dof_aperture = min(200.0, v._dof_aperture * 1.4)
        v.render_settings.reset_accumulation()
        print(f"DOF aperture: {v._dof_aperture:.1f}")
        v._update_frame()

    def _action_dof_focal_down(self):
        v = self.v
        v._dof_focal_distance = max(10.0, v._dof_focal_distance * 0.7)
        v.render_settings.reset_accumulation()
        print(f"DOF focal distance: {v._dof_focal_distance:.0f}")
        v._update_frame()

    def _action_dof_focal_up(self):
        v = self.v
        v._dof_focal_distance = min(10000.0, v._dof_focal_distance * 1.4)
        v.render_settings.reset_accumulation()
        print(f"DOF focal distance: {v._dof_focal_distance:.0f}")
        v._update_frame()

    def _action_exit(self):
        v = self.v
        v.running = False

    # ------------------------------------------------------------------
    # Key dispatch
    # ------------------------------------------------------------------

    def _handle_key_press(self, raw_key, key):
        """Handle key press — table-driven dispatch.

        Parameters
        ----------
        raw_key : str
            Key with original case (uppercase if SHIFT held).
        key : str
            Lowercase version of the key.
        """
        v = self.v
        # 1. Shift bindings (uppercase raw_key)
        if raw_key in SHIFT_BINDINGS:
            getattr(v, SHIFT_BINDINGS[raw_key])()
            return

        # 2. Movement keys -> add to held set
        if key in MOVEMENT_KEYS:
            v._held_keys.add(key)
            return

        # 3. Observer slots 1-8
        if key in ('1', '2', '3', '4', '5', '6', '7', '8'):
            v._select_or_create_observer(int(key))
            return

        # 4. Special bindings (need both raw_key and key)
        pair = (raw_key, key)
        if pair in SPECIAL_BINDINGS:
            getattr(v, SPECIAL_BINDINGS[pair])()
            return

        # 5. Regular key bindings (lowercase key)
        if key in KEY_BINDINGS:
            getattr(v, KEY_BINDINGS[key])()
            return

    def _handle_key_release(self, key):
        """Handle key release - remove from held keys.

        Parameters
        ----------
        key : str
            Lowercase key name.
        """
        v = self.v
        v._held_keys.discard(key)

    # ------------------------------------------------------------------
    # Mouse & scroll
    # ------------------------------------------------------------------

    def _handle_scroll(self, yoffset):
        """Handle mouse scroll wheel for zoom.

        Parameters
        ----------
        yoffset : float
            Scroll amount (positive = scroll up = zoom in).
        """
        v = self.v
        if yoffset > 0:
            v.fov = max(20, v.fov - 3)
        else:
            v.fov = min(120, v.fov + 3)
        print(f"FOV: {v.fov:.0f}")
        v._update_frame()

    def _handle_mouse_press(self, button, xpos, ypos):
        """Start drag on left-click, or teleport if click is on minimap.

        Parameters
        ----------
        button : int
            Mouse button (0 = left, 1 = right, 2 = middle).
        xpos, ypos : float
            Cursor position in window pixels.
        """
        v = self.v
        if button == 0:  # left click
            # Check for minimap click-to-teleport
            if v._minimap_rect is not None and v.show_minimap:
                mx0, my0, mw, mh = v._minimap_rect
                # Convert window coords to frame (render) coords
                frame_x = xpos * v.render_width / max(1, v.width)
                frame_y = ypos * v.render_height / max(1, v.height)
                if (mx0 <= frame_x < mx0 + mw and my0 <= frame_y < my0 + mh):
                    # Convert minimap-local -> world XY
                    local_x = frame_x - mx0
                    local_y = frame_y - my0
                    ext = v._minimap_world_extent
                    if ext is not None:
                        wx_min, wy_min, wx_max, wy_max = ext
                        world_x = wx_min + local_x / mw * (wx_max - wx_min)
                        world_y = wy_min + local_y / mh * (wy_max - wy_min)
                    else:
                        H, W = v.terrain_shape
                        world_x = local_x / mw * W * v.pixel_spacing_x
                        world_y = local_y / mh * H * v.pixel_spacing_y
                    v.position[0] = world_x
                    v.position[1] = world_y
                    v._update_frame()
                    return

            v._mouse_dragging = True
            v._mouse_last_x = xpos
            v._mouse_last_y = ypos

        elif button == 1:  # right click — object picking
            origin, direction = v._screen_to_ray(xpos, ypos)
            result = v.rtx.pick(origin, direction)
            if result['hit']:
                gid = result['geometry_id'] or '?'
                px, py, pz = result['position']
                print(f"Pick: geometry='{gid}'  pos=({px:.1f}, {py:.1f}, {pz:.1f})  "
                      f"t={result['t']:.1f}  prim={result['primitive_id']}  "
                      f"instance={result['instance_id']}")
            else:
                print("Pick: no geometry hit")

    def _handle_mouse_release(self, button):
        """End drag on button release."""
        v = self.v
        v._mouse_dragging = False

    def _handle_mouse_motion(self, xpos, ypos):
        """Pan camera on mouse drag (slippy-map style).

        Parameters
        ----------
        xpos, ypos : float
            Cursor position in screen pixels.
        """
        v = self.v
        if not v._mouse_dragging or v._mouse_last_x is None:
            return

        dx = xpos - v._mouse_last_x
        # GLFW Y is top-down; invert so dragging up -> positive dy
        dy = -(ypos - v._mouse_last_y)
        v._mouse_last_x = xpos
        v._mouse_last_y = ypos

        if dx == 0 and dy == 0:
            return

        H, W = v.terrain_shape
        world_diag = np.sqrt(
            (W * v.pixel_spacing_x) ** 2
            + (H * v.pixel_spacing_y) ** 2
        )
        sensitivity = world_diag * 0.20 / v.width

        right = v._get_right()
        front = v._get_front()
        front_horiz = np.array([front[0], front[1], 0], dtype=np.float32)
        norm = np.linalg.norm(front_horiz)
        if norm > 1e-8:
            front_horiz /= norm
        else:
            front_horiz = np.array([0, 1, 0], dtype=np.float32)

        # Scene follows cursor: drag right -> camera left
        v.position -= right * dx * sensitivity
        v.position -= front_horiz * dy * sensitivity

        v._update_frame()

    # ------------------------------------------------------------------
    # Layer cycling / jumping
    # ------------------------------------------------------------------

    def _cycle_terrain_layer(self):
        """Cycle terrain color: elevation -> overlay1 -> overlay2 -> ... -> elevation.

        Only affects terrain coloring. Does NOT touch basemap or geometry.
        """
        v = self.v
        if not v._terrain_layer_order:
            print("No terrain layers available")
            return

        v._terrain_layer_idx = (v._terrain_layer_idx + 1) % len(v._terrain_layer_order)
        layer_name = v._terrain_layer_order[v._terrain_layer_idx]

        if layer_name == 'elevation':
            v._active_color_data = None
            v._active_overlay_data = None
            v._overlay_as_water = False
            v._active_overlay_color_lut = None
            print(f"Terrain: elevation")
        else:
            v._active_color_data = None
            v._active_overlay_data = v._overlay_layers[layer_name]
            v._overlay_as_water = (
                layer_name.startswith('flood_')
                or (layer_name == 'stream_link' and v._hydro_enabled))
            v._active_overlay_color_lut = v._overlay_color_luts.get(
                layer_name)
            if v._overlay_as_water:
                print(f"Terrain: {layer_name} (water)")
            else:
                alpha_pct = int(v._overlay_alpha * 100)
                print(f"Terrain: {layer_name} (alpha {alpha_pct}%, ,/. to adjust)")

        v._update_frame()

    def _cycle_basemap(self, reverse=False):
        """Cycle basemap: none -> satellite -> osm -> none.

        Auto-creates XYZTileService on-the-fly if needed.
        """
        v = self.v
        step = -1 if reverse else 1
        v._basemap_idx = (v._basemap_idx + step) % len(v._basemap_options)
        provider = v._basemap_options[v._basemap_idx]

        if provider == 'none':
            v._tiles_enabled = False
            if v._texture_tile_mgr is not None:
                v._texture_tile_mgr.clear()
            print("Basemap: none")
        else:
            from ..tiles import XYZTileService
            # Create or switch tile service
            if v._tile_service is not None:
                if v._tile_service.provider_name != provider:
                    v._tile_service.shutdown()
                    v._tile_service = XYZTileService(
                        url_template=provider, raster=v._base_raster,
                    )
            else:
                v._tile_service = XYZTileService(
                    url_template=provider, raster=v._base_raster,
                )
            v._tiles_enabled = True
            print(f"Basemap: {provider}")
            # When LOD active, clear and re-fetch basemap lazily per tile.
            # When LOD not active, use monolithic fetch.
            if (v._texture_tile_mgr is not None
                    and v.lod_enabled
                    and v._terrain_lod_manager is not None):
                v._texture_tile_mgr.clear()
                # Re-fire tile callbacks for all visible tiles so basemap
                # gets fetched for them in background threads.
                mgr = v._terrain_lod_manager
                if mgr._on_tile_added is not None:
                    for (tr, tc) in list(mgr._tile_lods.keys()):
                        mgr._fire_tile_added(tr, tc)
            else:
                v._tile_service.fetch_visible_tiles()

        v._update_frame()

    def _cycle_geometry_layer(self):
        """Cycle geometry visibility: none -> all -> group1 -> group2 -> ... -> none.

        Uses rtx.set_geometry_visible() to show/hide geometry groups.
        """
        v = self.v
        if v.rtx is None or len(v._geometry_layer_order) <= 2:
            # Only 'none' and 'all' with no actual groups
            if v.rtx is None:
                print("No geometries in scene")
                return

        v._geometry_layer_idx = (v._geometry_layer_idx + 1) % len(v._geometry_layer_order)
        layer_name = v._geometry_layer_order[v._geometry_layer_idx]

        if layer_name == 'none':
            # Hide all non-terrain geometries
            for geom_id in v._all_geometries:
                if geom_id != 'terrain':
                    v.rtx.set_geometry_visible(geom_id, False)
            print("Geometry: none")

        elif layer_name == 'all':
            # Show all geometries
            for geom_id in v._all_geometries:
                v.rtx.set_geometry_visible(geom_id, True)
            print("Geometry: all")

        else:
            # Show only this geometry group + terrain
            visible_count = 0
            for geom_id in v._all_geometries:
                parts = geom_id.rsplit('_', 1)
                base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else geom_id
                if base_name == layer_name or geom_id == layer_name:
                    v.rtx.set_geometry_visible(geom_id, True)
                    visible_count += 1
                else:
                    v.rtx.set_geometry_visible(geom_id, False)
            print(f"Geometry: {layer_name} ({visible_count} visible)")

        v._current_geom_idx = 0
        v._update_frame()

    def _cycle_pointcloud_colors(self):
        """Cycle point cloud color mode: elevation -> intensity -> classification -> rgb."""
        v = self.v
        acc = v._accessor
        if acc is None or not hasattr(acc, '_pc_attributes') or not acc._pc_attributes:
            print("No point cloud geometries in scene")
            return

        from ..pointcloud import build_colors

        v._pc_color_mode_idx = (v._pc_color_mode_idx + 1) % len(v._pc_color_modes)
        mode = v._pc_color_modes[v._pc_color_mode_idx]

        updated = 0
        for gid, (centers, attributes) in acc._pc_attributes.items():
            # Check the geometry still exists
            if v.rtx is None or not v.rtx.has_geometry(gid):
                continue

            # Build new colors
            new_colors = build_colors(centers, attributes, color_mode=mode)
            colors_flat = new_colors.ravel().astype(np.float32)

            # Update per-point colors on the RTX geometry state
            gs = v.rtx._geom_state
            gs.point_colors_per_gas[gid] = colors_flat
            # Invalidate concatenated GPU buffer so it gets rebuilt
            gs.point_colors = None
            gs.point_color_offsets = None

            # Update baked mesh colors (3rd element of 4-tuple)
            if gid in acc._baked_meshes:
                baked = acc._baked_meshes[gid]
                acc._baked_meshes[gid] = (baked[0], baked[1], colors_flat.copy(), baked[3])

            updated += 1

        if updated > 0:
            print(f"Point cloud color: {mode}")
            v._d_ao_accum = None
            v._ao_frame_count = 0
            v._prev_cam_state = None
            v._update_frame()
        else:
            print(f"No active point cloud geometries to recolor")

    def _jump_to_geometry(self, direction):
        """Jump camera to next/previous geometry in current layer.

        Parameters
        ----------
        direction : int
            1 for next, -1 for previous.
        """
        v = self.v
        if v.rtx is None:
            print("No geometries in scene")
            return

        # Get current geometry layer name
        mode = v._geometry_layer_order[v._geometry_layer_idx]

        if mode == 'none':
            print("No geometry layer selected. Press N to select one.")
            return

        if mode == 'all':
            # Cycle through all geometry positions across all groups
            all_positions = []
            for layer_name, positions in sorted(v._layer_positions.items()):
                all_positions.extend(positions)
            if not all_positions:
                print("No geometry positions available")
                return
            v._current_geom_idx = (v._current_geom_idx + direction) % len(all_positions)
            x, y, z, geom_id = all_positions[v._current_geom_idx]
            yaw_rad = np.radians(v.yaw)
            forward_level = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0], dtype=np.float32)
            v.position = np.array([
                x - forward_level[0] * 100,
                y - forward_level[1] * 100,
                z + 50
            ], dtype=np.float32)
            v.pitch = -15.0
            print(f"Jumped to {geom_id} ({v._current_geom_idx + 1}/{len(all_positions)})")
            print(f"  Position: ({x:.0f}, {y:.0f}, {z:.0f})")
            v._update_frame()
            return

        # Get positions for current layer
        if mode not in v._layer_positions:
            print(f"No positions for layer: {mode}")
            return

        positions = v._layer_positions[mode]
        if not positions:
            print(f"No geometries in layer: {mode}")
            return

        # Cycle through geometries
        v._current_geom_idx = (v._current_geom_idx + direction) % len(positions)
        x, y, z, geom_id = positions[v._current_geom_idx]

        # Position camera at geometry location, slightly above and behind
        # Calculate offset based on current viewing direction
        height_offset = 50  # Height above geometry
        distance_back = 100  # Distance behind geometry

        # Get current forward direction (but level, no pitch)
        yaw_rad = np.radians(v.yaw)
        forward_level = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0], dtype=np.float32)

        # Position camera behind and above the geometry
        v.position = np.array([
            x - forward_level[0] * distance_back,
            y - forward_level[1] * distance_back,
            z + height_offset
        ], dtype=np.float32)

        # Look at the geometry
        v.pitch = -15.0  # Look slightly down

        print(f"Jumped to {geom_id} ({v._current_geom_idx + 1}/{len(positions)})")
        print(f"  Position: ({x:.0f}, {y:.0f}, {z:.0f})")
        v._update_frame()

    def _get_terrain_z(self, world_x, world_y):
        """Sample terrain elevation at a world-coordinate position."""
        v = self.v
        if (np.isnan(world_x) or np.isnan(world_y)
                or np.isnan(v.pixel_spacing_x)
                or np.isnan(v.pixel_spacing_y)
                or v.pixel_spacing_x == 0
                or v.pixel_spacing_y == 0):
            return 0.0
        H, W = v.terrain_shape
        col = int(np.clip(world_x / v.pixel_spacing_x, 0, W - 1))
        row = int(np.clip(world_y / v.pixel_spacing_y, 0, H - 1))
        terrain_data = v.raster.data
        if hasattr(terrain_data, 'get'):
            z = float(terrain_data[row, col].get())
        else:
            z = float(terrain_data[row, col])
        if np.isnan(z):
            z = 0.0
        return z
