"""Predefined bounding boxes with recommended CRS for countries, cities,
and landscapes.

Each entry is a ``Location`` — a tuple subclass that unpacks as
``(west, south, east, north)`` in WGS 84 degrees and carries a ``.crs``
attribute with the recommended projected CRS for that area.

Usage::

    from rtxpy.scene_locations import CITIES, LANDSCAPES
    from rtxpy.scene import build_scene

    loc = LANDSCAPES['grand_canyon']
    build_scene(loc, 'grand_canyon.zarr', crs=loc.crs)

    # Still works as a plain bounds tuple
    west, south, east, north = loc
"""


# ---------------------------------------------------------------------------
# Location type — tuple that also carries a CRS
# ---------------------------------------------------------------------------

class Location(tuple):
    """Bounding box with a recommended projected CRS.

    Inherits from ``tuple`` so it unpacks as ``(west, south, east, north)``
    and can be passed directly to ``build_scene()`` or ``fetch_dem()``.

    Attributes
    ----------
    crs : str
        EPSG code string (e.g. ``'EPSG:32612'``) for the recommended
        projected coordinate reference system.
    units : str
        Linear unit of the CRS (``'meters'``, ``'feet'``, or ``'degrees'``).
    bounds : tuple
        The ``(west, south, east, north)`` values (same as unpacking).
    """

    def __new__(cls, bounds, crs, units='meters'):
        obj = super().__new__(cls, bounds)
        obj.crs = crs
        obj.units = units
        return obj

    @property
    def bounds(self):
        return (self[0], self[1], self[2], self[3])

    def __repr__(self):
        w, s, e, n = self
        return f"Location(({w}, {s}, {e}, {n}), crs='{self.crs}', units='{self.units}')"


def _utm_epsg(west, south, east, north):
    """Compute the UTM EPSG code for the center of a bounding box."""
    lon = (west + east) / 2
    lat = (south + north) / 2
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f'EPSG:326{zone:02d}'
    return f'EPSG:327{zone:02d}'


# ---------------------------------------------------------------------------
# Country-level CRS overrides — well-established national systems
# Everything else falls back to UTM at the centroid.
# ---------------------------------------------------------------------------

_COUNTRY_CRS = {
    'united_states': 'EPSG:5070',    # Conus Albers Equal Area
    'canada': 'EPSG:3978',           # NAD83 Canada Atlas Lambert
    'mexico': 'EPSG:6372',           # Mexico ITRF2008 LCC
    'brazil': 'EPSG:5880',           # SIRGAS 2000 Polyconic
    'argentina': 'EPSG:5343',        # POSGAR 2007 Argentina zone 5
    'chile': 'EPSG:5880',            # SIRGAS 2000 Polyconic (continent)
    'united_kingdom': 'EPSG:27700',  # British National Grid
    'france': 'EPSG:2154',           # RGF93 v1 / Lambert-93
    'germany': 'EPSG:25832',         # ETRS89 / UTM 32N
    'italy': 'EPSG:25832',           # ETRS89 / UTM 32N
    'spain': 'EPSG:25830',           # ETRS89 / UTM 30N
    'norway': 'EPSG:25833',          # ETRS89 / UTM 33N
    'switzerland': 'EPSG:2056',      # CH1903+ / LV95
    'iceland': 'EPSG:3057',          # ISN93 / Lambert 1993
    'greece': 'EPSG:2100',           # GGRS87 / Greek Grid
    'portugal': 'EPSG:3763',         # ETRS89 / PT-TM06
    'japan': 'EPSG:6677',            # JGD2011 / Japan Plane IX (central)
    'south_korea': 'EPSG:5186',      # Korea 2000 / Central Belt 2010
    'india': 'EPSG:7755',            # WGS 84 / India NSF LCC
    'turkey': 'EPSG:5254',           # TUREF / TM33
    'south_africa': 'EPSG:2048',     # Hartebeesthoek94 / Lo19
    'egypt': 'EPSG:22992',           # Egypt 1907 Red Belt
    'morocco': 'EPSG:26191',         # Merchich / Nord Maroc
    'australia': 'EPSG:3577',        # GDA94 Australian Albers
    'new_zealand': 'EPSG:2193',      # NZGD2000 / NZTM 2000
}


# CRS units — every CRS in our set uses meters, but this table exists
# so adding a feet-based CRS (e.g. US state plane) just means one entry here.
_CRS_UNITS = {
    # 'EPSG:2229': 'feet',  # example: NAD83 / California zone 5 (ftUS)
}


def _make(raw, crs_overrides=None):
    """Convert a {name: (w,s,e,n)} dict to {name: Location}."""
    out = {}
    for name, bounds in raw.items():
        if crs_overrides and name in crs_overrides:
            crs = crs_overrides[name]
        else:
            crs = _utm_epsg(*bounds)
        units = _CRS_UNITS.get(crs, 'meters')
        out[name] = Location(bounds, crs, units)
    return out


# ---------------------------------------------------------------------------
# Countries — rough mainland bounds, not including distant overseas territories
# ---------------------------------------------------------------------------

COUNTRIES = _make({
    # Americas
    'united_states': (-125.0, 24.5, -66.9, 49.4),
    'canada': (-141.0, 41.7, -52.6, 83.1),
    'mexico': (-118.4, 14.5, -86.7, 32.7),
    'brazil': (-73.99, -33.75, -34.79, 5.27),
    'argentina': (-73.58, -55.06, -53.59, -21.78),
    'colombia': (-79.0, -4.23, -66.87, 12.46),
    'chile': (-75.64, -55.98, -66.96, -17.51),
    'peru': (-81.33, -18.35, -68.65, -0.04),

    # Europe
    'united_kingdom': (-8.65, 49.86, 1.77, 60.86),
    'france': (-5.14, 42.33, 8.23, 51.09),
    'germany': (5.87, 47.27, 15.04, 55.06),
    'italy': (6.63, 36.65, 18.52, 47.09),
    'spain': (-9.30, 36.00, 3.32, 43.79),
    'norway': (4.65, 57.97, 31.08, 71.19),
    'switzerland': (5.96, 45.82, 10.49, 47.81),
    'iceland': (-24.53, 63.39, -13.50, 66.53),
    'greece': (19.37, 34.80, 29.65, 41.75),
    'portugal': (-9.50, 36.96, -6.19, 42.15),

    # Asia
    'japan': (129.56, 31.03, 145.82, 45.52),
    'china': (73.50, 18.16, 134.77, 53.56),
    'india': (68.18, 6.75, 97.40, 35.50),
    'south_korea': (125.89, 33.19, 129.58, 38.61),
    'thailand': (97.35, 5.61, 105.64, 20.46),
    'vietnam': (102.14, 8.56, 109.46, 23.39),
    'nepal': (80.06, 26.35, 88.20, 30.45),
    'indonesia': (95.01, -11.01, 141.02, 5.91),
    'turkey': (25.66, 35.82, 44.82, 42.11),

    # Africa
    'south_africa': (16.46, -34.84, 32.89, -22.13),
    'egypt': (24.70, 22.00, 36.87, 31.67),
    'kenya': (33.89, -4.68, 41.86, 5.02),
    'morocco': (-13.17, 27.66, -1.01, 35.93),
    'tanzania': (29.33, -11.75, 40.44, -0.99),
    'ethiopia': (32.99, 3.40, 47.99, 14.89),
    'namibia': (11.72, -28.97, 25.26, -16.96),

    # Oceania
    'australia': (113.16, -43.63, 153.64, -10.68),
    'new_zealand': (166.43, -47.29, 178.55, -34.39),
}, _COUNTRY_CRS)

# ---------------------------------------------------------------------------
# Cities — roughly 10-20 km across, centered on the urban core
# All use UTM computed from bbox center.
# ---------------------------------------------------------------------------

CITIES = _make({
    # Americas
    'new_york': (-74.05, 40.68, -73.90, 40.82),
    'los_angeles': (-118.35, 33.95, -118.15, 34.10),
    'san_francisco': (-122.52, 37.71, -122.36, 37.82),
    'chicago': (-87.72, 41.83, -87.58, 41.93),
    'mexico_city': (-99.22, 19.35, -99.07, 19.47),
    'rio_de_janeiro': (-43.30, -23.02, -43.12, -22.90),
    'buenos_aires': (-58.50, -34.65, -58.34, -34.54),
    'bogota': (-74.13, 4.58, -74.00, 4.72),
    'vancouver': (-123.22, 49.23, -123.05, 49.32),

    # Europe
    'london': (-0.18, 51.47, 0.01, 51.55),
    'paris': (2.28, 48.83, 2.42, 48.90),
    'rome': (12.43, 41.87, 12.54, 41.93),
    'barcelona': (2.11, 41.36, 2.23, 41.43),
    'berlin': (13.32, 52.47, 13.47, 52.56),
    'amsterdam': (4.84, 52.34, 4.96, 52.40),
    'istanbul': (28.90, 40.97, 29.10, 41.10),
    'zurich': (8.49, 47.34, 8.60, 47.41),
    'reykjavik': (-22.02, 64.12, -21.82, 64.17),

    # Asia
    'tokyo': (139.68, 35.63, 139.82, 35.73),
    'beijing': (116.30, 39.86, 116.48, 39.98),
    'shanghai': (121.40, 31.17, 121.55, 31.30),
    'hong_kong': (114.10, 22.25, 114.26, 22.35),
    'singapore': (103.77, 1.25, 103.90, 1.36),
    'seoul': (126.90, 37.52, 127.06, 37.60),
    'mumbai': (72.80, 18.90, 72.95, 19.10),
    'dubai': (55.20, 25.14, 55.38, 25.28),
    'kathmandu': (85.28, 27.67, 85.38, 27.75),
    'bangkok': (100.47, 13.71, 100.60, 13.80),

    # Africa
    'cape_town': (18.38, -34.00, 18.52, -33.88),
    'cairo': (31.19, 30.01, 31.34, 30.10),
    'nairobi': (36.76, -1.34, 36.88, -1.24),
    'marrakech': (-8.05, 31.60, -7.94, 31.67),

    # Oceania
    'sydney': (151.14, -33.90, 151.30, -33.82),
    'melbourne': (144.90, -37.84, 145.02, -37.77),
    'auckland': (174.70, -36.90, 174.84, -36.82),
    'queenstown': (168.62, -45.06, 168.72, -44.98),
})

# ---------------------------------------------------------------------------
# Landscapes — scenic or geologically interesting areas
# All use UTM computed from bbox center.
# ---------------------------------------------------------------------------

LANDSCAPES = _make({
    # North America
    'grand_canyon': (-112.20, 36.00, -112.00, 36.20),
    'yosemite': (-119.65, 37.70, -119.50, 37.80),
    'mount_rainier': (-121.82, 46.82, -121.68, 46.92),
    'yellowstone_falls': (-110.40, 44.70, -110.28, 44.78),
    'monument_valley': (-110.15, 36.95, -110.00, 37.05),
    'denali': (-151.10, 62.95, -150.80, 63.15),
    'niagara_falls': (-79.10, 43.06, -79.04, 43.10),
    'glacier_national_park': (-114.00, 48.60, -113.70, 48.80),
    'bryce_canyon': (-112.22, 37.55, -112.05, 37.65),
    'crater_lake': (-122.20, 42.88, -122.02, 43.00),
    'big_sur': (-121.92, 36.10, -121.75, 36.28),
    'hawaii_volcanoes': (-155.35, 19.35, -155.15, 19.50),

    # South America
    'iguazu_falls': (-54.48, -25.72, -54.40, -25.64),
    'torres_del_paine': (-73.15, -51.05, -72.90, -50.85),
    'angel_falls': (-62.58, 5.94, -62.48, 6.02),
    'salar_de_uyuni': (-68.00, -20.40, -67.60, -20.10),
    'machu_picchu': (-72.58, -13.20, -72.50, -13.14),

    # Europe
    'matterhorn': (7.60, 45.95, 7.72, 46.02),
    'norwegian_fjords': (6.80, 61.80, 7.20, 62.00),
    'santorini': (25.35, 36.37, 25.48, 36.46),
    'dolomites': (11.70, 46.40, 12.00, 46.60),
    'cliffs_of_moher': (-9.48, 52.96, -9.38, 53.00),
    'plitvice_lakes': (15.57, 44.85, 15.65, 44.92),
    'lauterbrunnen': (7.87, 46.55, 7.95, 46.61),
    'trolltunga': (6.70, 60.11, 6.78, 60.15),
    'amalfi_coast': (14.50, 40.60, 14.70, 40.68),
    'iceland_highlands': (-19.20, 63.90, -18.80, 64.10),

    # Asia
    'mount_everest': (86.85, 27.92, 86.98, 28.02),
    'mount_fuji': (138.68, 35.32, 138.82, 35.42),
    'halong_bay': (107.00, 20.85, 107.10, 20.95),
    'zhangjiajie': (110.38, 29.28, 110.52, 29.40),
    'himalayas_annapurna': (83.80, 28.50, 84.00, 28.65),
    'bali_volcanoes': (115.35, -8.40, 115.55, -8.25),
    'cappadocia': (34.75, 38.60, 34.95, 38.75),
    'petra': (35.40, 30.30, 35.50, 30.38),

    # Africa
    'kilimanjaro': (37.28, -3.12, 37.42, -3.00),
    'victoria_falls': (25.82, -17.95, 25.92, -17.88),
    'sahara_erg_chebbi': (-3.98, 31.10, -3.88, 31.20),
    'namib_dunes': (15.20, -24.80, 15.50, -24.60),
    'table_mountain': (18.38, -34.02, 18.46, -33.94),
    'ngorongoro_crater': (35.50, -3.24, 35.64, -3.14),
    'blyde_river_canyon': (30.75, -24.62, 30.88, -24.52),

    # Oceania
    'milford_sound': (167.80, -44.72, 167.98, -44.60),
    'uluru': (131.00, -25.40, 131.10, -25.30),
    'blue_mountains': (150.28, -33.78, 150.42, -33.68),
    'tongariro': (175.55, -39.30, 175.70, -39.18),
    'great_barrier_reef': (146.00, -18.40, 146.30, -18.20),
})


# Combined lookup for convenience
ALL = {}
ALL.update({f'country/{k}': v for k, v in COUNTRIES.items()})
ALL.update({f'city/{k}': v for k, v in CITIES.items()})
ALL.update({f'landscape/{k}': v for k, v in LANDSCAPES.items()})


def find(query):
    """Search locations by name substring.

    Parameters
    ----------
    query : str
        Case-insensitive substring to match against location names.

    Returns
    -------
    dict
        Matching ``{name: Location}`` entries.

    Example
    -------
    >>> from rtxpy.scene_locations import find
    >>> find('canyon')
    {'landscape/grand_canyon': Location(..., crs='EPSG:32612'), ...}
    """
    q = query.lower()
    return {k: v for k, v in ALL.items() if q in k.lower()}


def list_locations(category=None):
    """Print available locations grouped by category.

    Parameters
    ----------
    category : str, optional
        Filter to ``'country'``, ``'city'``, or ``'landscape'``.
        Prints all categories if omitted.
    """
    groups = [
        ('Countries', COUNTRIES),
        ('Cities', CITIES),
        ('Landscapes', LANDSCAPES),
    ]
    for label, locs in groups:
        tag = label.lower().rstrip('s').rstrip('ie') + ('y' if 'ies' in label.lower() else '')
        if category and category.lower() not in (label.lower(), tag):
            continue
        print(f"\n{label} ({len(locs)}):")
        for name, loc in sorted(locs.items()):
            w, s, e, n = loc
            print(f"  {name:30s} ({w:8.2f}, {s:7.2f}, {e:8.2f}, {n:7.2f})  {loc.crs} ({loc.units})")
