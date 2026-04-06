"""Observer / drone / viewshed controller for InteractiveViewer."""

import time

import numpy as np

from ..viewer.observers import Observer, OBSERVER_COLORS


class ObserverController:
    """Drone control, observer management, and viewshed computation.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    # ------------------------------------------------------------------
    # Drone helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_drone_front_for(obs):
        """Get forward direction for drone flight (uses observer yaw/pitch)."""
        yaw_rad = np.radians(obs.yaw)
        pitch_rad = np.radians(obs.pitch)
        return np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)

    @staticmethod
    def _get_drone_right_for(obs):
        """Get right direction for drone flight."""
        yaw_rad = np.radians(obs.yaw)
        pitch_rad = np.radians(obs.pitch)
        front = np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, front)
        return right / (np.linalg.norm(right) + 1e-8)

    def _clamp_drone_pos(self, pos):
        """Clamp drone position to stay within terrain extent and above surface."""
        v = self.v
        H, W = v.terrain_shape
        x_max = (W - 1) * v.pixel_spacing_x
        y_max = (H - 1) * v.pixel_spacing_y
        pos[0] = np.clip(pos[0], 0, x_max)
        pos[1] = np.clip(pos[1], 0, y_max)
        terrain_z = v._get_terrain_z(pos[0], pos[1])
        if pos[2] < terrain_z:
            pos[2] = terrain_z
        return pos

    def _sync_drone_from_pos_for(self, obs, pos):
        """Update an observer's position and drone mesh from a 3D position."""
        v = self.v
        pos = self._clamp_drone_pos(pos)
        obs.position = (float(pos[0]), float(pos[1]))
        obs.observer_elev = float(pos[2]) - v._get_terrain_z(
            pos[0], pos[1])
        if obs.observer_elev < 0:
            obs.observer_elev = 0.0
        self._update_observer_drone_for(obs)

        # Dynamically recalculate viewshed as the drone moves (throttled)
        if obs.viewshed_enabled:
            now = time.monotonic()
            if now - v._last_viewshed_time >= v._viewshed_recalc_interval:
                v._last_viewshed_time = now
                obs.viewshed_cache = None
                self._calculate_viewshed(quiet=True)

    # ------------------------------------------------------------------
    # Drone model loading / placement
    # ------------------------------------------------------------------

    def _load_drone_parts(self):
        """Load drone GLB split by material, returning per-part geometry + color."""
        v = self.v
        import os
        drone_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'examples', 'models', 'drone.glb'
        )
        if not os.path.exists(drone_path):
            print(f"  Drone model not found at {drone_path}")
            return []

        try:
            import trimesh
        except ImportError:
            print("  trimesh required for drone model")
            return []

        scene = trimesh.load(drone_path)
        if not isinstance(scene, trimesh.Scene):
            return []

        # First pass: collect sub-meshes and compute shared bounding box
        raw_parts = []
        all_verts = []
        for name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            verts = geom.vertices.copy().astype(np.float32)
            faces = geom.faces.copy().astype(np.int32)
            # Swap Y/Z (Y-up → Z-up)
            verts[:, [1, 2]] = verts[:, [2, 1]]
            # Extract material base color
            color = (0.6, 0.6, 0.6)  # fallback grey
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                if hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None:
                    c = mat.baseColorFactor
                    color = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            raw_parts.append((verts, faces, color))
            all_verts.append(verts)

        if not raw_parts:
            return []

        # Compute shared centre and Z-base across all parts
        combined = np.concatenate(all_verts, axis=0)
        center_xy = (combined[:, :2].min(axis=0) + combined[:, :2].max(axis=0)) / 2
        z_min = combined[:, 2].min()

        # Second pass: apply shared transform
        parts = []
        for verts, faces, color in raw_parts:
            verts[:, 0] -= center_xy[0]
            verts[:, 1] -= center_xy[1]
            verts[:, 2] -= z_min
            parts.append((verts.flatten(), faces.flatten(), color))
        return parts

    def _update_observer_drone_for(self, obs):
        """Place or update the drone mesh at an observer's position."""
        v = self.v
        if obs.position is None or v.rtx is None:
            return

        from ..mesh import make_transform

        # Lazy-load drone parts once (shared across all observers)
        if v._shared_drone_parts is None:
            v._shared_drone_parts = self._load_drone_parts()
            if not v._shared_drone_parts:
                return

        obs_x, obs_y = obs.position
        terrain_z = v._get_terrain_z(obs_x, obs_y)
        obs_z = terrain_z + obs.observer_elev

        # Scale drone to ~0.05× pixel_spacing so it's visible but not huge
        drone_scale = 0.0125 * max(v.pixel_spacing_x, v.pixel_spacing_y)

        transform = make_transform(x=obs_x, y=obs_y, z=obs_z, scale=drone_scale)

        # Tint base colors toward observer slot color
        slot_color = obs.color
        for i, (verts, idxs, base_color) in enumerate(v._shared_drone_parts):
            gid = obs.geometry_id(i)
            # Mix: 50% base + 50% slot tint
            tinted = tuple(0.5 * base_color[c] + 0.5 * slot_color[c]
                           for c in range(3))
            if obs.drone_placed:
                v.rtx.update_transform(gid, transform)
            else:
                v.rtx.add_geometry(gid, verts, idxs, transform=transform)

        # Set geometry colors (needs the accessor's color dict)
        if not obs.drone_placed:
            builder = getattr(v, '_geometry_colors_builder', None)
            if builder is not None:
                acc = getattr(builder, '__self__', None)
                if acc is not None and hasattr(acc, '_geometry_colors'):
                    for i, (_, _, base_color) in enumerate(v._shared_drone_parts):
                        if v._drone_glow:
                            color = (*slot_color, 1.8)
                        else:
                            color = tuple(0.5 * base_color[c] + 0.5 * slot_color[c]
                                          for c in range(3))
                        acc._geometry_colors[obs.geometry_id(i)] = color
                    acc._geometry_colors_dirty = True

        obs.drone_placed = True

    def _apply_drone_glow(self):
        """Toggle emissive glow on/off for all placed drone geometries."""
        v = self.v
        builder = getattr(v, '_geometry_colors_builder', None)
        if builder is None:
            return
        acc = getattr(builder, '__self__', None)
        if acc is None or not hasattr(acc, '_geometry_colors'):
            return
        parts = v._shared_drone_parts
        if not parts:
            return
        changed = False
        for obs in v._observers.values():
            if not obs.drone_placed:
                continue
            slot_color = obs.color
            for i, (_, _, base_color) in enumerate(parts):
                gid = obs.geometry_id(i)
                if v._drone_glow:
                    acc._geometry_colors[gid] = (*slot_color, 1.8)
                else:
                    acc._geometry_colors[gid] = tuple(
                        0.5 * base_color[c] + 0.5 * slot_color[c]
                        for c in range(3))
            changed = True
        if changed:
            acc._geometry_colors_dirty = True
            v._update_frame()

    def _set_drone_visibility_for(self, obs, visible):
        """Show or hide all drone sub-mesh geometries for an observer."""
        v = self.v
        if obs.drone_placed and v.rtx is not None:
            for i in range(len(v._shared_drone_parts or [])):
                v.rtx.set_geometry_visible(obs.geometry_id(i), visible)

    def _cycle_drone_mode_for(self, obs):
        """Cycle drone mode for observer: off -> 3rd person -> FPV -> off."""
        v = self.v
        if obs.position is None:
            print(f"Observer {obs.slot} has no position.")
            return

        if obs.drone_mode == 'off':
            # --- Enter 3rd person ---
            obs.saved_camera = (
                v.position.copy(),
                float(v.yaw),
                float(v.pitch),
            )
            obs.yaw = float(v.yaw)
            obs.pitch = 0.0
            obs.drone_mode = '3rd'
            print(f"Observer {obs.slot} DRONE 3RD PERSON: ON")

        elif obs.drone_mode == '3rd':
            # --- 3rd person -> FPV ---
            obs_x, obs_y = obs.position
            terrain_z = v._get_terrain_z(obs_x, obs_y)
            obs_z = terrain_z + obs.observer_elev
            v.position = np.array([obs_x, obs_y, obs_z], dtype=float)
            v.yaw = obs.yaw
            v.pitch = obs.pitch
            self._set_drone_visibility_for(obs, False)
            obs.drone_mode = 'fpv'
            print(f"Observer {obs.slot} DRONE FPV: ON")

        else:
            # --- FPV -> off ---
            self._sync_drone_from_pos_for(obs, v.position)
            self._set_drone_visibility_for(obs, True)
            if obs.saved_camera is not None:
                v.position = obs.saved_camera[0]
                v.yaw = obs.saved_camera[1]
                v.pitch = obs.saved_camera[2]
                obs.saved_camera = None
            obs.drone_mode = 'off'
            print(f"Observer {obs.slot} DRONE: OFF")

        v._update_frame()

    # ------------------------------------------------------------------
    # Observer selection / placement
    # ------------------------------------------------------------------

    def _snap_to_observer(self, obs):
        """Snap external camera to look at an observer's drone from nearby."""
        v = self.v
        if obs.position is None:
            print(f"Observer {obs.slot} has no position.")
            return
        if obs.drone_mode == 'fpv':
            return

        obs_x, obs_y = obs.position
        terrain_z = v._get_terrain_z(obs_x, obs_y)
        obs_z = terrain_z + obs.observer_elev

        spacing = max(v.pixel_spacing_x, v.pixel_spacing_y)
        offset = spacing * 8.0
        dx = v.position[0] - obs_x
        dy = v.position[1] - obs_y
        dist_xy = np.sqrt(dx * dx + dy * dy)
        if dist_xy > 1e-6:
            dx /= dist_xy
            dy /= dist_xy
        else:
            dx, dy = 1.0, 0.0

        v.position = np.array([
            obs_x + dx * offset,
            obs_y + dy * offset,
            obs_z + spacing * 3.0,
        ], dtype=float)

        to_drone = np.array([obs_x - v.position[0],
                             obs_y - v.position[1],
                             obs_z - v.position[2]])
        to_drone /= (np.linalg.norm(to_drone) + 1e-8)
        v.yaw = float(np.degrees(np.arctan2(to_drone[1], to_drone[0])))
        v.pitch = float(np.degrees(np.arcsin(np.clip(to_drone[2], -1, 1))))

        print(f"Snapped to observer {obs.slot} at ({obs_x:.0f}, {obs_y:.0f})")
        v._update_frame()

    def _place_observer_at(self, obs, x=None, y=None):
        """Move an observer to a position (defaults to camera XY).

        Parameters
        ----------
        obs : Observer
            The observer to position.
        x, y : float, optional
            World coordinates. If None, use current camera position.
        """
        v = self.v
        H, W = v.terrain_shape
        cam_x = x if x is not None else v.position[0]
        cam_y = y if y is not None else v.position[1]

        max_x = (W - 1) * v.pixel_spacing_x
        max_y = (H - 1) * v.pixel_spacing_y

        obs_x = float(np.clip(cam_x, 0, max_x))
        obs_y = float(np.clip(cam_y, 0, max_y))

        obs.position = (obs_x, obs_y)
        self._update_observer_drone_for(obs)

        print(f"Observer {obs.slot} placed at ({obs_x:.0f}, {obs_y:.0f})")

        if obs.viewshed_enabled:
            self._calculate_viewshed(quiet=True)

        v._update_frame()

    def _select_or_create_observer(self, slot):
        """Handle number key 1-8: select/create/deselect observer slot."""
        v = self.v
        if v._active_observer == slot:
            # Deselect — exit FPV first if active
            obs = v._observers.get(slot)
            if obs is not None and obs.drone_mode == 'fpv':
                self._exit_fpv_for(obs)
            v._active_observer = None
            v.viewshed_enabled = False
            v._viewshed_cache = None
            print(f"Observer {slot}: deselected")
            v._update_frame()
            return

        # If switching away from an FPV observer, exit FPV first
        if v._active_observer is not None:
            prev_obs = v._observers.get(v._active_observer)
            if prev_obs is not None and prev_obs.drone_mode == 'fpv':
                self._exit_fpv_for(prev_obs)

        if slot in v._observers:
            # Select existing — auto-enter FPV
            v._active_observer = slot
            obs = v._observers[slot]
            # Sync viewer-level viewshed from this observer
            v.viewshed_enabled = obs.viewshed_enabled
            v._viewshed_cache = obs.viewshed_cache
            # Enter FPV: save camera, snap to observer, hide drone
            obs.saved_camera = (
                v.position.copy(),
                float(v.yaw),
                float(v.pitch),
            )
            obs_x, obs_y = obs.position
            terrain_z = v._get_terrain_z(obs_x, obs_y)
            obs_z = terrain_z + obs.observer_elev
            v.position = np.array([obs_x, obs_y, obs_z], dtype=float)
            v.yaw = obs.yaw
            v.pitch = obs.pitch
            self._set_drone_visibility_for(obs, False)
            obs.drone_mode = 'fpv'
            print(f"Observer {slot}: FPV at ({obs.position[0]:.0f}, {obs.position[1]:.0f})")
        else:
            # Create new just in front of camera, matching altitude and angle
            front = v._get_front()
            spacing = max(v.pixel_spacing_x, v.pixel_spacing_y)
            offset = spacing * 3  # A few pixels in front
            obs_x = v.position[0] + front[0] * offset
            obs_y = v.position[1] + front[1] * offset
            # Clamp to terrain bounds
            H, W = v.terrain_shape
            obs_x = float(np.clip(obs_x, 0, (W - 1) * v.pixel_spacing_x))
            obs_y = float(np.clip(obs_y, 0, (H - 1) * v.pixel_spacing_y))
            terrain_z = v._get_terrain_z(obs_x, obs_y)
            cam_elev = max(0.0, v.position[2] - terrain_z)
            obs = Observer(slot, position=(obs_x, obs_y),
                           observer_elev=cam_elev)
            obs.yaw = v.yaw
            obs.pitch = v.pitch
            v._observers[slot] = obs
            v._active_observer = slot
            self._update_observer_drone_for(obs)
            print(f"Observer {slot} placed at ({obs_x:.0f}, {obs_y:.0f}), "
                  f"h={cam_elev:.3f}, yaw={v.yaw:.0f}, pitch={v.pitch:.0f}")
            if obs.viewshed_enabled:
                self._calculate_viewshed(quiet=True)
            v._update_frame()
            return

        v._update_frame()

    def _exit_fpv_for(self, obs):
        """Exit FPV mode for an observer, restoring camera."""
        v = self.v
        if obs.drone_mode != 'fpv':
            return
        self._sync_drone_from_pos_for(obs, v.position)
        self._set_drone_visibility_for(obs, True)
        if obs.saved_camera is not None:
            v.position = obs.saved_camera[0]
            v.yaw = obs.saved_camera[1]
            v.pitch = obs.saved_camera[2]
            obs.saved_camera = None
        obs.drone_mode = 'off'

    # ------------------------------------------------------------------
    # Observer cleanup
    # ------------------------------------------------------------------

    def _clear_observer_slot(self, slot):
        """Remove a single observer and its geometry."""
        v = self.v
        obs = v._observers.get(slot)
        if obs is None:
            return

        # Stop tour if running
        obs.stop_tour()

        # Exit drone mode (restore camera if FPV)
        if obs.drone_mode != 'off':
            if obs.drone_mode == 'fpv':
                self._set_drone_visibility_for(obs, True)
            if obs.saved_camera is not None:
                v.position = obs.saved_camera[0]
                v.yaw = obs.saved_camera[1]
                v.pitch = obs.saved_camera[2]
                obs.saved_camera = None
            obs.drone_mode = 'off'

        # Remove drone geometry
        if obs.drone_placed and v.rtx is not None:
            n = len(v._shared_drone_parts) if v._shared_drone_parts else 0
            builder = getattr(v, '_geometry_colors_builder', None)
            acc = getattr(builder, '__self__', None) if builder else None
            for i in range(n):
                gid = obs.geometry_id(i)
                v.rtx.remove_geometry(gid)
                if acc is not None and hasattr(acc, '_geometry_colors'):
                    acc._geometry_colors.pop(gid, None)
            if acc is not None and hasattr(acc, '_geometry_colors_dirty'):
                acc._geometry_colors_dirty = True
            obs.drone_placed = False

        del v._observers[slot]
        if v._active_observer == slot:
            v._active_observer = None

        print(f"Observer {slot} removed")
        v._update_frame()

    def _clear_all_observers(self):
        """Kill all observers -- stop tours, exit drone modes, remove geometry."""
        v = self.v
        # Find if any observer is in FPV and restore camera
        for obs in v._observers.values():
            if obs.drone_mode == 'fpv' and obs.saved_camera is not None:
                v.position = obs.saved_camera[0]
                v.yaw = obs.saved_camera[1]
                v.pitch = obs.saved_camera[2]
                break  # Only one can be in FPV at a time

        for slot in list(v._observers.keys()):
            obs = v._observers[slot]
            obs.stop_tour()
            # Remove drone geometry
            if obs.drone_placed and v.rtx is not None:
                n = len(v._shared_drone_parts) if v._shared_drone_parts else 0
                builder = getattr(v, '_geometry_colors_builder', None)
                acc = getattr(builder, '__self__', None) if builder else None
                for i in range(n):
                    gid = obs.geometry_id(i)
                    v.rtx.remove_geometry(gid)
                    if acc is not None and hasattr(acc, '_geometry_colors'):
                        acc._geometry_colors.pop(gid, None)
                if acc is not None and hasattr(acc, '_geometry_colors_dirty'):
                    acc._geometry_colors_dirty = True

        v._observers.clear()
        v._active_observer = None
        v.viewshed_enabled = False
        v._viewshed_cache = None
        print("All observers removed")
        v._update_frame()

    # ------------------------------------------------------------------
    # Viewshed
    # ------------------------------------------------------------------

    def _calculate_viewshed(self, quiet=False):
        """Calculate viewshed from the placed observer position.

        Uses GPU ray tracing to compute visibility from the fixed observer.
        Observer position is in world coordinates; this method converts to
        pixel indices for the viewshed calculation.

        Parameters
        ----------
        quiet : bool
            If True, suppress verbose output (used during dynamic updates).
        """
        v = self.v
        from ..analysis.viewshed import _viewshed_rt

        # Get observer position: from _calculate_viewshed_for compat bridge,
        # or from the active observer
        obs_pos = getattr(v, '_observer_position_compat', None)
        if obs_pos is None:
            # Try active observer
            obs = v._observers.get(v._active_observer) if v._active_observer else None
            if obs is not None:
                obs_pos = obs.position
        if obs_pos is None:
            if not quiet:
                print("No observer placed. Press 1-8 to create one.")
            return None

        world_x, world_y = obs_pos
        H, W = v.terrain_shape

        # Convert world coords to pixel indices
        px_x = world_x / v.pixel_spacing_x
        px_y = world_y / v.pixel_spacing_y

        # Validate coordinates are within terrain bounds (in pixel space)
        if px_x < 0 or px_x >= W or px_y < 0 or px_y >= H:
            if not quiet:
                print(f"Observer position pixel ({px_x:.1f}, {px_y:.1f}) outside terrain bounds")
            return None

        if not quiet:
            print(f"Computing viewshed... (observer height: {v.viewshed_observer_elev:.3f})")
            print(f"  Raster shape: {v.raster.shape}, pixel_spacing: ({v.pixel_spacing_x:.1f}, {v.pixel_spacing_y:.1f})")

        try:
            # Use the scene's existing RTX which includes all geometries
            # (terrain, buildings, etc.) so viewshed rays are occluded by them.
            rtx = v.rtx
            if not quiet:
                print(f"  Using scene RTX ({rtx.get_geometry_count()} geometries)")

            # Two-phase visibility for realistic viewshed:
            #  Phase 1 (primary rays): only terrain visible — rays hit ground,
            #           not building rooftops.
            #  Phase 2 (occlusion rays): terrain + structures visible — buildings
            #           block line-of-sight from ground to observer.
            #
            # The between_traces_cb callback switches from phase 1 → 2.
            saved_visibility = {}
            non_terrain_ids = []
            for geom_id in v._all_geometries:
                entry = rtx._geom_state.gas_entries.get(geom_id)
                if entry is not None:
                    saved_visibility[geom_id] = entry.visible
                    if geom_id != 'terrain':
                        non_terrain_ids.append(geom_id)
                        # Phase 1: hide non-terrain so primary rays hit ground
                        rtx.set_geometry_visible(geom_id, False)
                    elif not entry.visible:
                        rtx.set_geometry_visible(geom_id, True)

            # Hide all observer drones so they don't block viewshed
            for obs in v._observers.values():
                if obs.drone_placed and v._shared_drone_parts:
                    for i in range(len(v._shared_drone_parts)):
                        gid = obs.geometry_id(i)
                        saved_visibility[gid] = True
                        rtx.set_geometry_visible(gid, False)

            def _enable_structures():
                """Callback: make structures visible for occlusion trace."""
                for gid in non_terrain_ids:
                    rtx.set_geometry_visible(gid, True)

            # Convert pixel indices to raster coords
            y_coords = v.raster.indexes.get('y').values
            x_coords = v.raster.indexes.get('x').values

            # Clamp to valid range and get actual coord values
            x_idx = int(np.clip(px_x, 0, W - 1))
            y_idx = int(np.clip(px_y, 0, H - 1))
            x_coord = x_coords[x_idx] if x_idx < len(x_coords) else x_coords[-1]
            y_coord = y_coords[y_idx] if y_idx < len(y_coords) else y_coords[-1]

            if not quiet:
                print(f"  Observer at raster coords: ({x_coord:.1f}, {y_coord:.1f})")

            viewshed = _viewshed_rt(
                v.raster, rtx,
                x_coord, y_coord,
                v.viewshed_observer_elev,
                v.viewshed_target_elev,
                pixel_spacing_x=v.pixel_spacing_x,
                pixel_spacing_y=v.pixel_spacing_y,
                between_traces_cb=_enable_structures,
            )

            # Restore original visibility state
            for geom_id, was_visible in saved_visibility.items():
                rtx.set_geometry_visible(geom_id, was_visible)

            # Calculate coverage percentage
            vis_data = viewshed.data
            if hasattr(vis_data, 'get'):
                vis_np = vis_data.get()
            else:
                vis_np = vis_data
            visible_cells = np.sum(vis_np >= 0)
            total_cells = vis_np.size
            v._viewshed_coverage = 100.0 * visible_cells / total_cells

            # Cache result
            v._viewshed_cache = viewshed
            v._last_viewshed_time = time.monotonic()

            if not quiet:
                print(f"  Coverage: {v._viewshed_coverage:.1f}% terrain visible")
            return viewshed

        except Exception as e:
            if not quiet:
                import traceback
                print(f"Viewshed calculation failed: {e}")
                traceback.print_exc()
            return None

    def _apply_viewshed_overlay(self, img):
        """Apply viewshed overlay to rendered image.

        Visible areas get a teal glow, invisible areas remain unchanged.

        Parameters
        ----------
        img : ndarray
            RGB image array (H, W, 3) with values 0-255.

        Returns
        -------
        ndarray
            Image with viewshed overlay applied.
        """
        v = self.v
        if v._viewshed_cache is None:
            return img

        vis_data = v._viewshed_cache.data
        if hasattr(vis_data, 'get'):
            vis_np = vis_data.get()
        else:
            vis_np = np.asarray(vis_data)

        # Resize viewshed to match render resolution
        scale_y = img.shape[0] / vis_np.shape[0]
        scale_x = img.shape[1] / vis_np.shape[1]
        if scale_y != 1.0 or scale_x != 1.0:
            try:
                from scipy.ndimage import zoom
                vis_resized = zoom(vis_np, (scale_y, scale_x), order=0)
            except ImportError:
                # Fallback: use cv2 for resizing
                try:
                    import cv2
                    vis_resized = cv2.resize(vis_np, (img.shape[1], img.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                except ImportError:
                    # Last resort: nearest neighbor with numpy
                    y_idx = np.linspace(0, vis_np.shape[0]-1, img.shape[0]).astype(int)
                    x_idx = np.linspace(0, vis_np.shape[1]-1, img.shape[1]).astype(int)
                    vis_resized = vis_np[np.ix_(y_idx, x_idx)]
        else:
            vis_resized = vis_np

        # Create result image
        img_float = img.astype(np.float32)
        result = img_float.copy()

        # Visible areas: apply teal glow
        # Teal color: RGB(0, 200, 200) - cyan/teal
        visible_mask = vis_resized >= 0

        # Intensity based on viewing angle (0-90 degrees)
        # Lower angle = more direct view = brighter glow
        vis_angles = np.clip(vis_resized, 0, 90)
        glow_intensity = 1.0 - (vis_angles / 90.0)  # 1.0 at 0°, 0.0 at 90°
        glow_intensity = np.clip(glow_intensity, 0.4, 1.0)  # Min glow level

        # Teal glow color
        teal_r, teal_g, teal_b = 0, 220, 210  # Bright teal/cyan

        # Apply glow only to visible areas using additive blending
        alpha = v.viewshed_opacity
        for c, teal_val in enumerate([teal_r, teal_g, teal_b]):
            channel = result[:, :, c]
            glow = glow_intensity * teal_val * alpha
            channel[visible_mask] = np.clip(
                channel[visible_mask] * (1 - alpha * 0.3) + glow[visible_mask],
                0, 255
            )

        return result.astype(np.uint8)

    def _toggle_viewshed(self):
        """Toggle viewshed overlay on/off for the active observer."""
        v = self.v
        obs = v._observers.get(v._active_observer) if v._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 to select/create one.")
            return

        obs.viewshed_enabled = not obs.viewshed_enabled

        if obs.viewshed_enabled:
            print("Calculating viewshed...")
            # Temporarily set position/elev for _calculate_viewshed
            viewshed = self._calculate_viewshed_for(obs)
            if viewshed is None:
                obs.viewshed_enabled = False
                print("Viewshed: OFF (calculation failed)")
            else:
                v.viewshed_enabled = True
                v._viewshed_cache = obs.viewshed_cache
                print(f"Viewshed: ON ({v._viewshed_coverage:.1f}% coverage)")
        else:
            print("Viewshed: OFF")
            v.viewshed_enabled = False
            v._viewshed_cache = None

        v._update_frame()

    def _calculate_viewshed_for(self, obs, quiet=False):
        """Calculate viewshed using an observer's position/elevation."""
        v = self.v
        # Temporarily bridge to existing _calculate_viewshed by setting compat state
        old_pos = getattr(v, '_observer_position_compat', None)
        old_elev = v.viewshed_observer_elev
        v._observer_position_compat = obs.position
        v.viewshed_observer_elev = obs.observer_elev
        result = self._calculate_viewshed(quiet=quiet)
        obs.viewshed_cache = v._viewshed_cache
        v._observer_position_compat = old_pos
        v.viewshed_observer_elev = old_elev
        return result

    def _adjust_observer_elevation(self, delta):
        """Adjust active observer's elevation."""
        v = self.v
        obs = v._observers.get(v._active_observer) if v._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 first.")
            return

        obs.observer_elev = max(0, obs.observer_elev + delta)
        print(f"Observer {obs.slot} height: {obs.observer_elev:.3f}")

        self._update_observer_drone_for(obs)

        if obs.viewshed_enabled:
            obs.viewshed_cache = None
            self._calculate_viewshed_for(obs)
            v._viewshed_cache = obs.viewshed_cache
            v._update_frame()
