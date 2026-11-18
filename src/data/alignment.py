"""Monotone DP label alignment: maps GPS observations to ground-truth route arcs."""

import numpy as np

from src.utils.geometry import project, point_to_segment


def _cost_row(gps_x, gps_y, route_arcs_xy, lo, hi):
    """Compute point-to-segment distances for GPS point to route arcs[lo:hi].

    Returns array of length (hi - lo).
    """
    seg = route_arcs_xy[lo:hi]  # shape (K, 4)
    ax = seg[:, 0]
    ay = seg[:, 1]
    bx = seg[:, 2]
    by = seg[:, 3]

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    # Projection parameter t, clamped to [0, 1]
    dot = (gps_x - ax) * dx + (gps_y - ay) * dy
    # Avoid division by zero for degenerate segments
    safe_len_sq = np.where(seg_len_sq < 1e-12, 1.0, seg_len_sq)
    t = dot / safe_len_sq
    t = np.where(seg_len_sq < 1e-12, 0.0, np.clip(t, 0.0, 1.0))

    cx = ax + t * dx
    cy = ay + t * dy
    dist = np.sqrt((gps_x - cx) ** 2 + (gps_y - cy) ** 2)
    return dist


def align_observations(track, nodes, arcs, route, proj, band_width=100):
    """Align GPS observations to route arcs using band-restricted monotone DP.

    Transition model is NON-DECREASING: any forward jump is allowed.

    cost(i, j) = point_to_segment distance from GPS obs i to route arc route[j]

    Recurrence:
        dp[i][j] = cost(i, j) + min_{k <= j} dp[i-1][k]

    Band restriction: for obs i (out of N), expected_j = round(i * (M-1) / (N-1)).
    Search lo = max(0, expected_j - bw) to hi = min(M, expected_j + bw + 1).
    For obs following a gap >= 5 s, bw is doubled for that row.

    Uses prefix-min trick to avoid O(M^2) inner loop.

    Parameters
    ----------
    track      : list of (lon, lat, ts) tuples, length N
    nodes      : list of (lon, lat) tuples
    arcs       : list of (from_node_id, to_node_id) tuples
    route      : list of arc_ids (ints), length M
    proj       : dict from build_projection
    band_width : int — half-width of the DP band (default 100)

    Returns
    -------
    obs_to_route_pos : np.ndarray of int, shape (N,) — non-decreasing route position indices
    obs_to_arc       : np.ndarray of int, shape (N,) — corresponding arc_ids
    """
    N = len(track)
    M = len(route)

    if N == 0 or M == 0:
        return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64)

    # Precompute projected GPS positions
    gps_xy = np.array([project(lon, lat, proj) for lon, lat, _ in track])  # (N, 2)

    # Precompute projected arc endpoints for route arcs: (ax, ay, bx, by)
    route_arcs_xy = np.empty((M, 4), dtype=np.float64)
    for j, arc_id in enumerate(route):
        from_id, to_id = arcs[arc_id]
        lon_a, lat_a = nodes[from_id]
        lon_b, lat_b = nodes[to_id]
        ax, ay = project(lon_a, lat_a, proj)
        bx, by = project(lon_b, lat_b, proj)
        route_arcs_xy[j] = ax, ay, bx, by

    timestamps = np.array([ts for _, _, ts in track])

    INF = np.float64(1e18)

    # We keep only prev_dp (shape M) and bp (shape N x M)
    # bp[i][j] stores the best predecessor route position k <= j
    # To save memory, we use int16 offsets from j (j - k) which fits band_width*2
    prev_dp = np.full(M, INF, dtype=np.float64)

    # Backtrack: store chosen route pos for each (i, j) in band
    # We'll store the absolute chosen predecessor index as int32
    # Only store within-band values; out-of-band entries are unused
    bp = np.full((N, M), -1, dtype=np.int32)

    # --- Row 0 ---
    expected_j0 = 0
    lo0 = 0
    hi0 = min(M, band_width + 1)
    costs0 = _cost_row(gps_xy[0, 0], gps_xy[0, 1], route_arcs_xy, lo0, hi0)
    prev_dp[lo0:hi0] = costs0
    # bp[0][j] = j for all j in band (no predecessor)
    bp[0, lo0:hi0] = np.arange(lo0, hi0, dtype=np.int32)

    curr_dp = np.full(M, INF, dtype=np.float64)

    # --- Rows 1..N-1 ---
    for i in range(1, N):
        curr_dp[:] = INF

        expected_j = int(round(i * (M - 1) / (N - 1))) if N > 1 else 0
        gap = (timestamps[i] - timestamps[i - 1]) >= 5.0
        bw = band_width * 2 if gap else band_width
        lo = max(0, expected_j - bw)
        hi = min(M, expected_j + bw + 1)

        # Compute cost vector for this row's band
        costs = _cost_row(gps_xy[i, 0], gps_xy[i, 1], route_arcs_xy, lo, hi)

        # Prefix-min over prev_dp[lo:hi]
        # prefix_min[k] = min(prev_dp[lo], ..., prev_dp[lo+k])
        # We also track argmin for backtracking
        band_prev = prev_dp[lo:hi]  # view, shape K
        K = hi - lo

        prefix_min = np.empty(K, dtype=np.float64)
        prefix_argmin = np.empty(K, dtype=np.int32)
        running_min = INF
        running_argmin = lo
        for k in range(K):
            v = band_prev[k]
            if v < running_min:
                running_min = v
                running_argmin = lo + k
            prefix_min[k] = running_min
            prefix_argmin[k] = running_argmin

        # For each j in [lo, hi): curr_dp[j] = costs[j-lo] + prefix_min[j-lo]
        total = costs + prefix_min
        valid = prefix_min < INF
        curr_dp[lo:hi] = np.where(valid, total, INF)
        # Store predecessor
        bp[i, lo:hi] = np.where(valid, prefix_argmin, -1).astype(np.int32)

        prev_dp, curr_dp = curr_dp, prev_dp

    # --- Backtrack ---
    last_i = N - 1
    expected_j_last = int(round(last_i * (M - 1) / (N - 1))) if N > 1 else 0
    bw_last = band_width
    lo_last = max(0, expected_j_last - bw_last)
    hi_last = min(M, expected_j_last + bw_last + 1)

    # Find best final j in last row's band
    slice_last = prev_dp[lo_last:hi_last]
    best_offset = int(np.argmin(slice_last))
    best_j = lo_last + best_offset

    # If the last row is entirely INF (can happen with very narrow band), widen search
    if prev_dp[best_j] >= INF * 0.5:
        best_j = int(np.argmin(prev_dp))

    obs_to_route_pos = np.empty(N, dtype=np.int64)
    obs_to_route_pos[last_i] = best_j

    j = best_j
    for i in range(last_i, 0, -1):
        pred = int(bp[i][j])
        if pred < 0:
            # Fallback: stay at same position
            pred = j
        j = pred
        obs_to_route_pos[i - 1] = j

    obs_to_arc = np.array([route[int(p)] for p in obs_to_route_pos], dtype=np.int64)

    # Assert non-decreasing
    assert np.all(np.diff(obs_to_route_pos) >= 0), (
        "obs_to_route_pos is not non-decreasing — DP bug"
    )

    return obs_to_route_pos, obs_to_arc
