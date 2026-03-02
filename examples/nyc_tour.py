"""NYC circular flyover tour.

Orbits around the five boroughs in a continuous loop.
The camera circles the study area, always looking toward the center.

Usage:
    python new_york_city.py --tour nyc_tour.py

Or from the REPL:
    v.tour('nyc_tour.py')
"""

import math

# --- Orbit parameters ---
# World coordinates: (0,0) is raster top-left corner,
# units are col * pixel_spacing (~26.4 m per pixel).
# Terrain extent: ~47887 x 48310 m
CX = 23_943          # world center x
CY = 24_155          # world center y

RADIUS = 18_000      # orbit radius (m) — near terrain edges
ALTITUDE = 5_000     # altitude above sea level (m)
GROUND_ELEV = 100    # approximate mean terrain elevation
FOV = 60
ORBIT_SECONDS = 45   # time for one full revolution
EASE = 'linear'      # constant speed around the circle

# Pitch: aim camera at the center of the terrain
PITCH = -math.degrees(math.atan2(ALTITUDE - GROUND_ELEV, RADIUS))

# Loop forever
loop = True

# --- Generate keyframes around the circle ---
N_KEYFRAMES = 12     # every 30 degrees
tour = []

for i in range(N_KEYFRAMES + 1):
    # Angle around the circle (start from south, go clockwise
    # when viewed from above: S -> W -> N -> E -> S)
    theta = 2 * math.pi * i / N_KEYFRAMES
    t = ORBIT_SECONDS * i / N_KEYFRAMES

    x = CX + RADIUS * math.sin(theta)
    y = CY - RADIUS * math.cos(theta)

    # Yaw: camera looks toward center.
    # atan2 gives the angle of (center - camera) from +X axis.
    # Engine convention: yaw 0 = +X, yaw 90 = +Y.
    dx = CX - x
    dy = CY - y
    yaw = math.degrees(math.atan2(dy, dx))

    kf = {
        'time': t,
        'position': [x, y, ALTITUDE],
        'yaw': yaw,
        'pitch': PITCH,
        'fov': FOV,
        'ease': EASE,
    }

    # Show all geometry on the first keyframe
    if i == 0:
        kf['geometry'] = 'all'

    tour.append(kf)
