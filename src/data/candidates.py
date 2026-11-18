"""Spatial indexing and candidate arc generation."""

import math
import numpy as np
from scipy.spatial import cKDTree

from src.utils.geometry import project, point_to_segment


def build_spatial_index(nodes, arcs, proj):
    """Build a cKDTree on arc midpoints in metric space.

    Parameters
    ----------
    nodes : list of (lon, lat) tuples
    arcs  : list of (from_node_id, to_node_id) tuples
    proj  : dict from build_projection

    Returns
    -------
    (tree, midpoints_array, arc_ids_array, max_arc_half_length)
      tree              : cKDTree
      midpoints_array   : (N, 2) float64 array of arc midpoint coordinates in metres
      arc_ids_array     : (N,) int32 array mapping tree indices to arc_ids
      max_arc_half_length : float — half the maximum arc length across all arcs
    """
    n_arcs = len(arcs)
    midpoints = np.empty((n_arcs, 2), dtype=np.float64)
    half_lengths = np.empty(n_arcs, dtype=np.float64)
    arc_ids = np.arange(n_arcs, dtype=np.int32)

    for i, (from_id, to_id) in enumerate(arcs):
        lon_a, lat_a = nodes[from_id]
        lon_b, lat_b = nodes[to_id]
        ax, ay = project(lon_a, lat_a, proj)
        bx, by = project(lon_b, lat_b, proj)
        midpoints[i, 0] = (ax + bx) / 2.0
        midpoints[i, 1] = (ay + by) / 2.0
        half_lengths[i] = math.hypot(bx - ax, by - ay) / 2.0

    max_arc_half_length = float(half_lengths.max()) if n_arcs > 0 else 0.0

    tree = cKDTree(midpoints)
    return tree, midpoints, arc_ids, max_arc_half_length


def get_candidates(gps_lon, gps_lat, proj, tree, nodes, arcs, arc_ids_array,
                   max_arc_half_length, radius=50.0):
    """Return candidate arcs within radius metres of a GPS observation.

    The KDTree search radius is expanded by max_arc_half_length so that arcs
    whose midpoint is farther than radius but whose closest point is within
    radius are still considered.

    Parameters
    ----------
    gps_lon, gps_lat : float
    proj              : dict from build_projection
    tree              : cKDTree
    nodes             : list of (lon, lat) tuples
    arcs              : list of (from_node_id, to_node_id) tuples
    arc_ids_array     : (N,) int32 array
    max_arc_half_length : float
    radius            : float — search radius in metres (default 50.0)

    Returns
    -------
    list of (arc_id, snap_distance_m, snap_t) sorted by snap_distance_m ascending
    """
    px, py = project(gps_lon, gps_lat, proj)
    search_radius = radius + max_arc_half_length

    indices = tree.query_ball_point([px, py], r=search_radius)

    candidates = []
    for idx in indices:
        arc_id = int(arc_ids_array[idx])
        from_id, to_id = arcs[arc_id]
        lon_a, lat_a = nodes[from_id]
        lon_b, lat_b = nodes[to_id]
        ax, ay = project(lon_a, lat_a, proj)
        bx, by = project(lon_b, lat_b, proj)

        dist, t = point_to_segment(px, py, ax, ay, bx, by)
        if dist <= radius:
            candidates.append((arc_id, dist, t))

    candidates.sort(key=lambda c: c[1])
    return candidates
