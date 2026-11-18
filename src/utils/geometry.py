"""Per-trip local metric projection and geometric utilities."""

import math
import numpy as np


def build_projection(track):
    """Build a per-trip local metric projection from track bounding box centroid.

    Parameters
    ----------
    track : list of (lon, lat, ts) tuples

    Returns
    -------
    dict with keys: lat_ref, m_per_deg_lon, m_per_deg_lat, lon_origin, lat_origin
    """
    lons = [p[0] for p in track]
    lats = [p[1] for p in track]
    lat_ref = (min(lats) + max(lats)) / 2.0
    lon_origin = (min(lons) + max(lons)) / 2.0
    lat_origin = lat_ref

    # Metres per degree latitude (approximately constant)
    m_per_deg_lat = 111_132.0  # standard approximation

    # Metres per degree longitude varies with latitude
    m_per_deg_lon = 111_132.0 * math.cos(math.radians(lat_ref))

    return {
        "lat_ref": lat_ref,
        "m_per_deg_lon": m_per_deg_lon,
        "m_per_deg_lat": m_per_deg_lat,
        "lon_origin": lon_origin,
        "lat_origin": lat_origin,
    }


def project(lon, lat, proj):
    """Convert geographic coordinates to local metres.

    Parameters
    ----------
    lon, lat : float
    proj : dict from build_projection

    Returns
    -------
    (x_m, y_m) in metres relative to projection origin
    """
    x = (lon - proj["lon_origin"]) * proj["m_per_deg_lon"]
    y = (lat - proj["lat_origin"]) * proj["m_per_deg_lat"]
    return x, y


def point_to_segment(px, py, ax, ay, bx, by):
    """Compute distance from point P to line segment AB and projection parameter t.

    Parameters
    ----------
    px, py : float  — point coordinates
    ax, ay : float  — segment start
    bx, by : float  — segment end

    Returns
    -------
    (distance_m, t) where t in [0, 1] is the clamped projection parameter
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-12:
        # Degenerate segment: return distance to A
        dist = math.hypot(px - ax, py - ay)
        return dist, 0.0

    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    cx = ax + t * dx
    cy = ay + t * dy
    dist = math.hypot(px - cx, py - cy)
    return dist, t


def arc_length(nodes, from_id, to_id, proj):
    """Compute arc length in metres.

    Parameters
    ----------
    nodes : list of (lon, lat) tuples
    from_id, to_id : int — node indices
    proj : dict from build_projection

    Returns
    -------
    float — length in metres
    """
    lon_a, lat_a = nodes[from_id]
    lon_b, lat_b = nodes[to_id]
    ax, ay = project(lon_a, lat_a, proj)
    bx, by = project(lon_b, lat_b, proj)
    return math.hypot(bx - ax, by - ay)


def bearing(ax, ay, bx, by):
    """Compute bearing from point A to point B.

    Uses navigational convention: north = 0, east = pi/2.

    Parameters
    ----------
    ax, ay, bx, by : float — coordinates in any consistent unit

    Returns
    -------
    float — bearing in radians
    """
    dx = bx - ax
    dy = by - ay
    return math.atan2(dx, dy)


def wrap_angle(a):
    """Wrap angle to [-pi, pi].

    Parameters
    ----------
    a : float — angle in radians

    Returns
    -------
    float
    """
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a
