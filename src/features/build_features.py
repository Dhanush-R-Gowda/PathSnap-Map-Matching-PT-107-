"""Phase 1 feature engineering: build candidate-level feature DataFrames."""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd

from src.data.loader import load_trip, discover_trips
from src.utils.geometry import (
    build_projection, project, arc_length, bearing, wrap_angle
)
from src.data.candidates import build_spatial_index, get_candidates
from src.data.alignment import align_observations


# ---------------------------------------------------------------------------
# Degree lookup helpers
# ---------------------------------------------------------------------------

def _build_degree_maps(arcs):
    """Return (out_degree, in_degree) dicts keyed by node_id.

    out_degree[node] = number of arcs whose from_node == node
    in_degree[node]  = number of arcs whose to_node == node
    """
    out_deg = {}
    in_deg = {}
    for from_id, to_id in arcs:
        out_deg[from_id] = out_deg.get(from_id, 0) + 1
        in_deg[to_id] = in_deg.get(to_id, 0) + 1
    return out_deg, in_deg


# ---------------------------------------------------------------------------
# Single-trip pipeline
# ---------------------------------------------------------------------------

def build_trip_features(trip_id, trip_dir, radius=50.0):
    """Run the full Phase 1 pipeline for one trip and return a feature DataFrame.

    Parameters
    ----------
    trip_id  : str — 8-digit zero-padded trip ID
    trip_dir : str — full path to the trip subdirectory
    radius   : float — candidate search radius in metres

    Returns
    -------
    pd.DataFrame with the Phase 1 schema (see specification)
    """
    data = load_trip(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    N = len(track)
    M = len(route)

    proj = build_projection(track)

    # --- Spatial index ---
    tree, midpoints_array, arc_ids_array, max_arc_half_length = build_spatial_index(
        nodes, arcs, proj
    )

    # --- Candidate generation ---
    # candidates_per_obs[i] = list of (arc_id, snap_dist, snap_t)
    candidates_per_obs = []
    for lon, lat, _ts in track:
        cands = get_candidates(
            lon, lat, proj, tree, nodes, arcs, arc_ids_array,
            max_arc_half_length, radius=radius
        )
        candidates_per_obs.append(cands)

    # --- Label alignment ---
    obs_to_route_pos, obs_to_arc = align_observations(
        track, nodes, arcs, route, proj
    )

    # --- Degree maps ---
    out_deg, in_deg = _build_degree_maps(arcs)

    # --- Arc lengths and bearings (cached) ---
    arc_len_cache = {}
    arc_bearing_cache = {}

    def get_arc_len(arc_id):
        if arc_id not in arc_len_cache:
            from_id, to_id = arcs[arc_id]
            arc_len_cache[arc_id] = arc_length(nodes, from_id, to_id, proj)
        return arc_len_cache[arc_id]

    def get_arc_bearing(arc_id):
        if arc_id not in arc_bearing_cache:
            from_id, to_id = arcs[arc_id]
            lon_a, lat_a = nodes[from_id]
            lon_b, lat_b = nodes[to_id]
            ax, ay = project(lon_a, lat_a, proj)
            bx, by = project(lon_b, lat_b, proj)
            arc_bearing_cache[arc_id] = bearing(ax, ay, bx, by)
        return arc_bearing_cache[arc_id]

    # --- GPS kinematics ---
    # dt_prev_s[i], dt_next_s[i], heading_reliable[i], gps_speed, gps_heading
    timestamps = np.array([ts for _, _, ts in track])
    dt_prev = np.zeros(N, dtype=np.float32)
    dt_next = np.zeros(N, dtype=np.float32)

    if N > 1:
        dt_prev[1:] = (timestamps[1:] - timestamps[:-1]).astype(np.float32)
        dt_next[:-1] = dt_prev[1:]  # dt_next[i] = dt_prev[i+1]
        # Last obs: dt_next = 0.0 (already)

    heading_reliable = np.zeros(N, dtype=np.int8)
    gps_speed = np.zeros(N, dtype=np.float32)
    gps_heading = np.zeros(N, dtype=np.float32)

    for i in range(1, N):
        if dt_prev[i] < 5.0 and dt_prev[i] > 0.0:
            heading_reliable[i] = 1
            lon0, lat0, _ = track[i - 1]
            lon1, lat1, _ = track[i]
            x0, y0 = project(lon0, lat0, proj)
            x1, y1 = project(lon1, lat1, proj)
            dist_m = math.hypot(x1 - x0, y1 - y0)
            hdg = bearing(x0, y0, x1, y1)
            gps_speed[i] = dist_m / float(dt_prev[i])
            gps_heading[i] = hdg

    is_gap_prev = (dt_prev >= 5.0).astype(np.int8)
    is_gap_next = (dt_next >= 5.0).astype(np.int8)
    # obs 0: dt_prev=0 so not a gap, is_gap_prev[0]=0 (correct)

    # --- Build rows ---
    rows = []
    miss_obs = []

    for i in range(N):
        gt_arc = int(obs_to_arc[i])
        cands = candidates_per_obs[i]

        # Is GT arc in candidates?
        cand_arc_ids = {c[0] for c in cands}
        gt_in_cands = gt_arc in cand_arc_ids

        if not gt_in_cands:
            miss_obs.append(i)

        for arc_id, snap_dist, snap_t in cands:
            from_id, to_id = arcs[arc_id]
            a_len = get_arc_len(arc_id)
            a_bearing = get_arc_bearing(arc_id)

            if heading_reliable[i]:
                bdiff = wrap_angle(a_bearing - float(gps_heading[i]))
                abs_bdiff = abs(bdiff)
            else:
                bdiff = 0.0
                abs_bdiff = 0.0

            od = out_deg.get(to_id, 0)
            ind = in_deg.get(from_id, 0)

            label = int(arc_id == gt_arc)

            rows.append({
                "trip_id": trip_id,
                "obs_index": np.int32(i),
                "arc_id": np.int32(arc_id),
                "snap_distance_m": np.float32(snap_dist),
                "snap_t": np.float32(snap_t),
                "arc_length_m": np.float32(a_len),
                "bearing_diff_rad": np.float32(bdiff),
                "abs_bearing_diff_rad": np.float32(abs_bdiff),
                "gps_speed_mps": np.float32(gps_speed[i]),
                "gps_heading_rad": np.float32(gps_heading[i]),
                "dt_prev_s": np.float32(dt_prev[i]),
                "dt_next_s": np.float32(dt_next[i]),
                "out_degree": np.int16(od),
                "in_degree": np.int16(ind),
                "is_gap_prev": np.int8(is_gap_prev[i]),
                "is_gap_next": np.int8(is_gap_next[i]),
                "heading_reliable": np.int8(heading_reliable[i]),
                "label": np.int8(label),
                "is_gt_in_candidates": gt_in_cands,
            })

    if not rows:
        # Return empty DataFrame with correct schema
        return _empty_schema_df()

    df = pd.DataFrame(rows)
    df = df.astype({
        "trip_id": "object",
        "obs_index": "int32",
        "arc_id": "int32",
        "snap_distance_m": "float32",
        "snap_t": "float32",
        "arc_length_m": "float32",
        "bearing_diff_rad": "float32",
        "abs_bearing_diff_rad": "float32",
        "gps_speed_mps": "float32",
        "gps_heading_rad": "float32",
        "dt_prev_s": "float32",
        "dt_next_s": "float32",
        "out_degree": "int16",
        "in_degree": "int16",
        "is_gap_prev": "int8",
        "is_gap_next": "int8",
        "heading_reliable": "int8",
        "label": "int8",
        "is_gt_in_candidates": "bool",
    })

    return df


def _empty_schema_df():
    """Return an empty DataFrame with the correct Phase 1 schema."""
    return pd.DataFrame(columns=[
        "trip_id", "obs_index", "arc_id", "snap_distance_m", "snap_t",
        "arc_length_m", "bearing_diff_rad", "abs_bearing_diff_rad",
        "gps_speed_mps", "gps_heading_rad", "dt_prev_s", "dt_next_s",
        "out_degree", "in_degree", "is_gap_prev", "is_gap_next",
        "heading_reliable", "label", "is_gt_in_candidates",
    ]).astype({
        "trip_id": "object",
        "obs_index": "int32",
        "arc_id": "int32",
        "snap_distance_m": "float32",
        "snap_t": "float32",
        "arc_length_m": "float32",
        "bearing_diff_rad": "float32",
        "abs_bearing_diff_rad": "float32",
        "gps_speed_mps": "float32",
        "gps_heading_rad": "float32",
        "dt_prev_s": "float32",
        "dt_next_s": "float32",
        "out_degree": "int16",
        "in_degree": "int16",
        "is_gap_prev": "int8",
        "is_gap_next": "int8",
        "heading_reliable": "int8",
        "label": "int8",
        "is_gt_in_candidates": "bool",
    })


# ---------------------------------------------------------------------------
# Partition-level pipeline
# ---------------------------------------------------------------------------

def build_partition_features(partition_dir, output_path, radius=50.0):
    """Process all trips in a partition and save features to parquet.

    Parameters
    ----------
    partition_dir : str — path to train/ or student_test/ directory
    output_path   : str — output parquet path
    radius        : float — candidate search radius in metres

    Returns
    -------
    dict with keys:
      total_rows, num_trips, class_balance, overall_miss_rate,
      trip_miss_rates (dict trip_id -> miss_rate float)
    """
    trip_ids = discover_trips(partition_dir)
    all_dfs = []
    trip_miss_rates = {}
    total_obs = 0
    total_misses = 0

    for trip_id in trip_ids:
        trip_dir = os.path.join(partition_dir, trip_id)
        print(f"  Processing trip {trip_id} ...", flush=True)
        try:
            df = build_trip_features(trip_id, trip_dir, radius=radius)
        except Exception as exc:
            warnings.warn(f"Trip {trip_id} failed: {exc}")
            continue

        if df.empty:
            trip_miss_rates[trip_id] = 0.0
            continue

        # Per-trip miss stats
        per_obs = df.groupby("obs_index")["is_gt_in_candidates"].first()
        n_obs = len(per_obs)
        n_miss = int((~per_obs).sum())
        miss_rate = n_miss / n_obs if n_obs > 0 else 0.0
        trip_miss_rates[trip_id] = miss_rate
        total_obs += n_obs
        total_misses += n_miss

        all_dfs.append(df)
        print(
            f"    {len(df)} rows, {n_obs} obs, miss_rate={miss_rate:.3%}",
            flush=True
        )

    if not all_dfs:
        full_df = _empty_schema_df()
    else:
        full_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(full_df)} rows to {output_path}", flush=True)

    total_rows = len(full_df)
    if total_rows > 0:
        n_pos = int((full_df["label"] == 1).sum())
        class_balance = n_pos / total_rows
    else:
        class_balance = 0.0

    overall_miss_rate = total_misses / total_obs if total_obs > 0 else 0.0

    return {
        "total_rows": total_rows,
        "num_trips": len(all_dfs),
        "class_balance": class_balance,
        "overall_miss_rate": overall_miss_rate,
        "trip_miss_rates": trip_miss_rates,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))

    parser = argparse.ArgumentParser(description="Build Phase 1 candidate features")
    parser.add_argument(
        "--partition", choices=["train", "student_test", "both"], default="both"
    )
    parser.add_argument("--radius", type=float, default=50.0)
    args = parser.parse_args()

    raw_root = os.path.join(repo_root, "data", "raw", "Student folder")
    processed_root = os.path.join(repo_root, "data", "processed")
    results_root = os.path.join(repo_root, "results", "experiments")

    summaries = {}

    if args.partition in ("train", "both"):
        print("\n=== Processing TRAIN partition ===", flush=True)
        train_dir = os.path.join(raw_root, "train")
        train_out = os.path.join(processed_root, "train_candidates.parquet")
        summaries["train"] = build_partition_features(
            train_dir, train_out, radius=args.radius
        )
        print(f"\nTrain summary: {summaries['train']}", flush=True)

    if args.partition in ("student_test", "both"):
        print("\n=== Processing STUDENT_TEST partition ===", flush=True)
        test_dir = os.path.join(raw_root, "student_test")
        test_out = os.path.join(processed_root, "student_test_candidates.parquet")
        summaries["student_test"] = build_partition_features(
            test_dir, test_out, radius=args.radius
        )
        print(f"\nStudent test summary: {summaries['student_test']}", flush=True)

    # Save miss rate report
    from src.data.miss_rate import compute_miss_rates

    if "train" in summaries:
        miss_rate_path = os.path.join(results_root, "candidate_miss_rates.csv")
        compute_miss_rates(summaries["train"], miss_rate_path, radius=args.radius)
