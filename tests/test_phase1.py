"""Phase 1 unit tests."""

import os
import math
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ROOT = os.path.join(REPO_ROOT, "data", "raw", "Student folder", "train")
PARQUET_PATH = os.path.join(REPO_ROOT, "data", "processed", "train_candidates.parquet")

TRIP_DIR_1 = os.path.join(RAW_ROOT, "00000001")
TRIP_DIR_2 = os.path.join(RAW_ROOT, "00000002")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.data.loader import load_track, load_nodes, load_arcs, load_route
from src.utils.geometry import point_to_segment, bearing, build_projection, project
from src.data.candidates import build_spatial_index, get_candidates
from src.data.alignment import align_observations


# ---------------------------------------------------------------------------
# 1-4: Row counts
# ---------------------------------------------------------------------------

def test_load_track_trip1():
    rows = load_track(TRIP_DIR_1)
    assert len(rows) == 1315, f"Expected 1315 track rows for trip 00000001, got {len(rows)}"


def test_load_track_trip2():
    rows = load_track(TRIP_DIR_2)
    assert len(rows) == 7243, f"Expected 7243 track rows for trip 00000002, got {len(rows)}"


def test_load_nodes_trip1():
    rows = load_nodes(TRIP_DIR_1)
    assert len(rows) == 88809, f"Expected 88809 node rows for trip 00000001, got {len(rows)}"


def test_load_nodes_trip2():
    rows = load_nodes(TRIP_DIR_2)
    assert len(rows) == 7936, f"Expected 7936 node rows for trip 00000002, got {len(rows)}"


def test_load_arcs_trip1():
    rows = load_arcs(TRIP_DIR_1)
    assert len(rows) == 165790, f"Expected 165790 arc rows for trip 00000001, got {len(rows)}"


def test_load_arcs_trip2():
    rows = load_arcs(TRIP_DIR_2)
    assert len(rows) == 16139, f"Expected 16139 arc rows for trip 00000002, got {len(rows)}"


def test_load_route_trip1():
    rows = load_route(TRIP_DIR_1)
    assert len(rows) == 525, f"Expected 525 route rows for trip 00000001, got {len(rows)}"


def test_load_route_trip2():
    rows = load_route(TRIP_DIR_2)
    assert len(rows) == 262, f"Expected 262 route rows for trip 00000002, got {len(rows)}"


# ---------------------------------------------------------------------------
# 5: Arc entry check
# ---------------------------------------------------------------------------

def test_load_arcs_trip1_index2641():
    rows = load_arcs(TRIP_DIR_1)
    entry = rows[2641]
    assert entry == (1584, 32308), (
        f"Expected arcs[2641] == (1584, 32308), got {entry}"
    )


# ---------------------------------------------------------------------------
# 6: First track row longitude
# ---------------------------------------------------------------------------

def test_load_track_trip1_first_lon():
    rows = load_track(TRIP_DIR_1)
    lon = rows[0][0]
    assert abs(lon - 8.595203) < 1e-5, (
        f"Expected first lon ≈ 8.595203, got {lon}"
    )


# ---------------------------------------------------------------------------
# 7-8: point_to_segment
# ---------------------------------------------------------------------------

def test_point_to_segment_midpoint():
    ax, ay = 0.0, 0.0
    bx, by = 4.0, 0.0
    px, py = 2.0, 0.0  # midpoint of AB
    dist, t = point_to_segment(px, py, ax, ay, bx, by)
    assert dist < 1e-9, f"Expected dist ≈ 0, got {dist}"
    assert abs(t - 0.5) < 1e-9, f"Expected t ≈ 0.5, got {t}"


def test_point_to_segment_endpoint_a():
    ax, ay = 0.0, 0.0
    bx, by = 4.0, 0.0
    px, py = 0.0, 0.0  # exactly at A
    dist, t = point_to_segment(px, py, ax, ay, bx, by)
    assert dist < 1e-9, f"Expected dist ≈ 0, got {dist}"
    assert abs(t - 0.0) < 1e-9, f"Expected t ≈ 0.0, got {t}"


# ---------------------------------------------------------------------------
# 9-10: Bearing
# ---------------------------------------------------------------------------

def test_bearing_north():
    b = bearing(0.0, 0.0, 0.0, 1.0)
    assert abs(b - 0.0) < 1e-9, f"Expected bearing north ≈ 0.0, got {b}"


def test_bearing_east():
    b = bearing(0.0, 0.0, 1.0, 0.0)
    assert abs(b - math.pi / 2) < 1e-9, f"Expected bearing east ≈ pi/2, got {b}"


# ---------------------------------------------------------------------------
# 11: All candidates have snap_distance_m <= radius
# ---------------------------------------------------------------------------

def test_candidates_within_radius():
    track = load_track(TRIP_DIR_1)
    nodes = load_nodes(TRIP_DIR_1)
    arcs = load_arcs(TRIP_DIR_1)
    proj = build_projection(track)

    tree, midpoints_array, arc_ids_array, max_arc_half_length = build_spatial_index(
        nodes, arcs, proj
    )

    radius = 50.0
    # Check first 20 observations for speed
    for lon, lat, _ts in track[:20]:
        cands = get_candidates(
            lon, lat, proj, tree, nodes, arcs, arc_ids_array,
            max_arc_half_length, radius=radius
        )
        for arc_id, snap_dist, snap_t in cands:
            assert snap_dist <= radius + 1e-6, (
                f"Candidate arc {arc_id} has snap_distance_m={snap_dist:.4f} > radius={radius}"
            )


# ---------------------------------------------------------------------------
# 12: obs_to_route_pos is non-decreasing for trip 00000001
# ---------------------------------------------------------------------------

def test_obs_to_route_pos_nondecreasing():
    track = load_track(TRIP_DIR_1)
    nodes = load_nodes(TRIP_DIR_1)
    arcs = load_arcs(TRIP_DIR_1)
    route = load_route(TRIP_DIR_1)
    proj = build_projection(track)

    obs_to_route_pos, obs_to_arc = align_observations(track, nodes, arcs, route, proj)
    diffs = np.diff(obs_to_route_pos)
    assert np.all(diffs >= 0), (
        f"obs_to_route_pos is not non-decreasing; found {(diffs < 0).sum()} violations"
    )


# ---------------------------------------------------------------------------
# 13-16: Parquet-based tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_df():
    assert os.path.exists(PARQUET_PATH), (
        f"Parquet file not found: {PARQUET_PATH} — run the pipeline first"
    )
    return pd.read_parquet(PARQUET_PATH)


def test_label_values(train_df):
    unique_labels = set(train_df["label"].unique())
    assert unique_labels <= {0, 1, np.int8(0), np.int8(1)}, (
        f"Label column contains unexpected values: {unique_labels}"
    )


def test_at_most_one_label1_per_obs(train_df):
    counts = train_df[train_df["label"] == 1].groupby(["trip_id", "obs_index"]).size()
    over_one = counts[counts > 1]
    assert len(over_one) == 0, (
        f"{len(over_one)} (trip_id, obs_index) groups have more than one label=1"
    )


def test_heading_reliable_zero_on_gaps(train_df):
    gap_rows = train_df[train_df["dt_prev_s"] >= 5.0]
    if len(gap_rows) > 0:
        assert (gap_rows["heading_reliable"] == 0).all(), (
            "heading_reliable != 0 for some rows where dt_prev_s >= 5"
        )


def test_is_gap_prev_on_gaps(train_df):
    gap_rows = train_df[train_df["dt_prev_s"] >= 5.0]
    if len(gap_rows) > 0:
        assert (gap_rows["is_gap_prev"] == 1).all(), (
            "is_gap_prev != 1 for some rows where dt_prev_s >= 5"
        )
