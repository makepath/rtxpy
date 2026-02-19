"""Tour playback engine for the explore() interactive viewer.

Defines keyframe-based camera tours with smooth interpolation and
optional frame recording for video assembly.

Typical usage from the REPL::

    tour = [
        {'time': 0, 'position': [100, 200, 50], 'yaw': 90, 'pitch': -20},
        {'time': 5, 'position': [300, 200, 80], 'yaw': 120, 'pitch': -30},
        {'time': 10, 'position': [300, 400, 60], 'yaw': 180, 'pitch': -25},
    ]
    v.tour(tour)
    v.tour(tour, record=True, output_dir='frames/')
"""

import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Easing functions
# ---------------------------------------------------------------------------

def ease_linear(t):
    """Linear interpolation (no easing)."""
    return t


def ease_in_out(t):
    """Smoothstep — gentle acceleration and deceleration."""
    return t * t * (3 - 2 * t)


def ease_in(t):
    """Quadratic ease-in — slow start."""
    return t * t


def ease_out(t):
    """Quadratic ease-out — slow finish."""
    return 1 - (1 - t) * (1 - t)


_EASING = {
    'linear': ease_linear,
    'ease_in_out': ease_in_out,
    'ease_in': ease_in,
    'ease_out': ease_out,
}


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _lerp(a, b, t):
    """Linear interpolation between scalars or arrays."""
    return a + (b - a) * t


def _lerp_angle(a, b, t):
    """Interpolate angles (degrees) via the shortest arc."""
    diff = (b - a) % 360
    if diff > 180:
        diff -= 360
    return a + diff * t


# ---------------------------------------------------------------------------
# Camera state capture
# ---------------------------------------------------------------------------

def mark_camera(proxy):
    """Capture the current camera state as a keyframe dict (without time).

    Returns a dict with ``position``, ``yaw``, ``pitch``, and ``fov``.
    """
    return {
        'position': proxy.position.tolist(),
        'yaw': float(proxy.yaw),
        'pitch': float(proxy.pitch),
        'fov': proxy.run(lambda v: v.fov),
    }


# ---------------------------------------------------------------------------
# Tour playback
# ---------------------------------------------------------------------------

def play_tour(proxy, keyframes, fps=30, record=False, output_dir='.',
              loop=False):
    """Play a camera tour through the viewer.

    Parameters
    ----------
    proxy : ViewerProxy
        The ``v`` handle from ``explore(repl=True)``.
    keyframes : list of dict
        Each dict may contain:

        - ``time`` (float, required) — seconds from tour start.
        - ``position`` (list[3]) — camera position ``[x, y, z]``.
        - ``yaw``, ``pitch``, ``fov`` (float) — camera orientation/FOV.
        - ``layer`` (str) — switch terrain layer.
        - ``colormap`` (str) — switch colormap.
        - ``geometry`` (str) — show only this geometry group.
        - ``shadows`` (bool) — toggle shadows.
        - ``screenshot`` (bool) — take a screenshot at this keyframe.
        - ``ease`` (str) — easing function for interpolation arriving
          at this keyframe (default ``'ease_in_out'``).

        Camera fields are interpolated between keyframes.  Action
        fields trigger once when the keyframe time is crossed.
    fps : int
        Target playback framerate (default 30).
    record : bool
        If True, save a frame after each interpolated step.
    output_dir : str or Path
        Directory for recorded frames (``frame_0001.png``, ...).
    loop : bool
        If True, repeat the tour indefinitely until the viewer closes.
    """
    if not keyframes:
        print("Tour: no keyframes")
        return

    keyframes = sorted(keyframes, key=lambda k: k['time'])
    duration = keyframes[-1]['time']
    dt = 1.0 / fps

    if record:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    cam_fields = ('position', 'yaw', 'pitch', 'fov')

    loop_label = " (looping)" if loop else ""
    print(f"Tour: {len(keyframes)} keyframes, {duration:.1f}s "
          f"@ {fps} fps{' (recording)' if record else ''}{loop_label}")

    frame_num = 0
    lap = 0

    while True:
        # Reset action triggers each lap
        action_fired = [False] * len(keyframes)
        t_start = time.monotonic()
        t_tour = 0.0

        while t_tour <= duration + 1e-9:
            # Check if viewer is still alive
            if not proxy._viewer.running:
                if frame_num and record:
                    print(f"Tour stopped. {frame_num} frames saved "
                          f"to {output_dir}")
                return

            # --- Interpolate camera state ---
            cam_state = {}
            for field in cam_fields:
                defined = [(kf['time'], kf[field],
                            kf.get('ease', 'ease_in_out'))
                           for kf in keyframes if field in kf]
                if not defined:
                    continue

                if t_tour <= defined[0][0]:
                    cam_state[field] = defined[0][1]
                    continue
                if t_tour >= defined[-1][0]:
                    cam_state[field] = defined[-1][1]
                    continue

                for i in range(len(defined) - 1):
                    t0, v0, _ = defined[i]
                    t1, v1, ease_name = defined[i + 1]
                    if t0 <= t_tour <= t1:
                        raw_t = ((t_tour - t0) / (t1 - t0)
                                 if t1 > t0 else 1.0)
                        ease_fn = _EASING.get(ease_name, ease_in_out)
                        t_eased = ease_fn(raw_t)
                        if field == 'position':
                            v0 = np.asarray(v0, dtype=np.float64)
                            v1 = np.asarray(v1, dtype=np.float64)
                            cam_state[field] = _lerp(v0, v1, t_eased)
                        elif field == 'yaw':
                            cam_state[field] = _lerp_angle(
                                v0, v1, t_eased)
                        else:
                            cam_state[field] = _lerp(v0, v1, t_eased)
                        break

            # Apply camera state on the render thread
            if cam_state:
                snapshot = dict(cam_state)

                def _apply(v, s=snapshot):
                    if 'position' in s:
                        v.position[:] = s['position']
                    if 'yaw' in s:
                        v.yaw = s['yaw']
                    if 'pitch' in s:
                        v.pitch = s['pitch']
                    if 'fov' in s:
                        v.fov = s['fov']
                    v._update_frame()

                proxy.run(_apply)

            # --- Fire action triggers ---
            for i, kf in enumerate(keyframes):
                if action_fired[i]:
                    continue
                if t_tour >= kf['time']:
                    action_fired[i] = True
                    if 'layer' in kf:
                        proxy.show_layer(kf['layer'])
                    if 'colormap' in kf:
                        proxy.set_colormap(kf['colormap'])
                    if 'geometry' in kf:
                        proxy.show_geometry(kf['geometry'])
                    if 'shadows' in kf:
                        proxy.shadows = kf['shadows']
                    if kf.get('screenshot'):
                        proxy.screenshot()

            # --- Record frame ---
            if record:
                frame_num += 1
                fname = out / f"frame_{frame_num:05d}.png"

                def _save(v, path=str(fname)):
                    from PIL import Image
                    frame = v._pinned_frame
                    if frame is not None:
                        rgb = np.clip(frame[:, :, :3] * 255, 0, 255
                                      ).astype(np.uint8)
                        img = Image.fromarray(rgb)
                        img.save(path)

                proxy.run(_save)

            # --- Timing ---
            t_tour += dt
            t_elapsed = time.monotonic() - t_start
            sleep_time = t_tour - t_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        lap += 1
        if not loop:
            break

    print(f"Tour complete. {frame_num} frames"
          + (f" saved to {output_dir}" if record else ""))


# ---------------------------------------------------------------------------
# Interpolation helper (shared by camera tour and observer tour)
# ---------------------------------------------------------------------------

def _interpolate_fields(keyframes, t_tour, fields):
    """Interpolate keyframe fields at time *t_tour*.

    Returns a dict of interpolated values for each field found in
    *keyframes*.
    """
    state = {}
    for field in fields:
        defined = [(kf['time'], kf[field],
                     kf.get('ease', 'ease_in_out'))
                    for kf in keyframes if field in kf]
        if not defined:
            continue

        if t_tour <= defined[0][0]:
            state[field] = defined[0][1]
            continue
        if t_tour >= defined[-1][0]:
            state[field] = defined[-1][1]
            continue

        for i in range(len(defined) - 1):
            t0, v0, _ = defined[i]
            t1, v1, ease_name = defined[i + 1]
            if t0 <= t_tour <= t1:
                raw_t = ((t_tour - t0) / (t1 - t0)
                         if t1 > t0 else 1.0)
                ease_fn = _EASING.get(ease_name, ease_in_out)
                t_eased = ease_fn(raw_t)
                if field == 'position':
                    v0 = np.asarray(v0, dtype=np.float64)
                    v1 = np.asarray(v1, dtype=np.float64)
                    state[field] = _lerp(v0, v1, t_eased)
                elif field == 'yaw':
                    state[field] = _lerp_angle(v0, v1, t_eased)
                else:
                    state[field] = _lerp(v0, v1, t_eased)
                break
    return state


# ---------------------------------------------------------------------------
# Observer tour playback
# ---------------------------------------------------------------------------

def play_observer_tour(proxy, slot, keyframes, fps=30, loop=False):
    """Animate an observer drone along a keyframe path.

    Runs in a daemon thread. The observer's ``tour_stop`` event is
    checked each tick for cooperative cancellation.

    Parameters
    ----------
    proxy : ViewerProxy
        The ``v`` handle from ``explore(repl=True)``.
    slot : int
        Observer slot (1-8).
    keyframes : list of dict
        Each dict may contain ``time``, ``position`` (x,y,z list),
        ``yaw``, ``pitch``, ``observer_elev``.  Only ``position``
        and ``time`` are required.
    fps : int
        Target playback framerate.
    loop : bool
        Repeat indefinitely until stopped.
    """
    if not keyframes:
        print(f"Observer {slot} tour: no keyframes")
        return

    keyframes = sorted(keyframes, key=lambda k: k['time'])
    duration = keyframes[-1]['time']
    dt = 1.0 / fps

    tour_fields = ('position', 'yaw', 'pitch', 'observer_elev')

    loop_label = " (looping)" if loop else ""
    print(f"Observer {slot} tour: {len(keyframes)} keyframes, "
          f"{duration:.1f}s @ {fps} fps{loop_label}")

    while True:
        t_start = time.monotonic()
        t_tour = 0.0

        while t_tour <= duration + 1e-9:
            # Check cancellation
            obs = proxy._viewer._observers.get(slot)
            if obs is None or obs.tour_stop.is_set():
                print(f"Observer {slot} tour stopped")
                return
            if not proxy._viewer.running:
                return

            state = _interpolate_fields(keyframes, t_tour, tour_fields)

            if state:
                snapshot = dict(state)
                snap_slot = slot

                def _apply(v, s=snapshot, sl=snap_slot):
                    o = v._observers.get(sl)
                    if o is None:
                        return
                    if 'position' in s:
                        pos = s['position']
                        o.position = (float(pos[0]), float(pos[1]))
                        if len(pos) > 2:
                            terrain_z = v._get_terrain_z(pos[0], pos[1])
                            o.observer_elev = max(0.0,
                                                  float(pos[2]) - terrain_z)
                    if 'observer_elev' in s:
                        o.observer_elev = float(s['observer_elev'])
                    if 'yaw' in s:
                        o.yaw = float(s['yaw'])
                    if 'pitch' in s:
                        o.pitch = float(s['pitch'])
                    v._update_observer_drone_for(o)
                    # If this observer is in FPV and active, camera follows
                    if (o.drone_mode == 'fpv'
                            and v._active_observer == sl):
                        ox, oy = o.position
                        tz = v._get_terrain_z(ox, oy)
                        v.position[:] = [ox, oy, tz + o.observer_elev]
                        v.yaw = o.yaw
                        v.pitch = o.pitch
                    v._render_needed = True

                proxy._submit_fire_and_forget(_apply)

            t_tour += dt
            t_elapsed = time.monotonic() - t_start
            sleep_time = t_tour - t_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if not loop:
            break

    print(f"Observer {slot} tour complete")
