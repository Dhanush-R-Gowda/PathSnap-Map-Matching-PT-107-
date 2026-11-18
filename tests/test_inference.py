"""Route-free inference: no .route needed, features identical to training."""

import os
import shutil

import pandas as pd
import pytest

from src.data.loader import load_trip_for_inference
from src.features.build_features import build_trip_features, build_trip_features_inference

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIP_DIR = os.path.join(REPO_ROOT, "data", "raw", "Student folder", "train", "00000001")
FEATURES = [
    "snap_distance_m", "snap_t", "arc_length_m", "bearing_diff_rad",
    "abs_bearing_diff_rad", "gps_speed_mps", "gps_heading_rad", "dt_prev_s",
    "dt_next_s", "out_degree", "in_degree", "is_gap_prev", "is_gap_next",
    "heading_reliable",
]
META = ["trip_id", "obs_index", "arc_id"]

pytestmark = pytest.mark.skipif(not os.path.isdir(TRIP_DIR), reason="raw data not available")


@pytest.fixture()
def no_route_dir(tmp_path):
    d = tmp_path / "00000001"
    d.mkdir()
    for ext in ("track", "nodes", "arcs"):
        shutil.copy(os.path.join(TRIP_DIR, f"00000001.{ext}"), d)
    assert not (d / "00000001.route").exists()
    return str(d)


def test_loader_works_without_route(no_route_dir):
    data = load_trip_for_inference(no_route_dir)
    assert set(data) == {"track", "nodes", "arcs"}


def test_inference_features_match_training(no_route_dir):
    data = load_trip_for_inference(no_route_dir)
    inf = build_trip_features_inference("00000001", data["track"], data["nodes"], data["arcs"])
    train = build_trip_features("00000001", TRIP_DIR)
    assert list(inf.columns) == META + FEATURES
    assert "label" not in inf.columns and "is_gt_in_candidates" not in inf.columns
    assert len(inf) == len(train) > 0
    cols = META + FEATURES
    pd.testing.assert_frame_equal(
        inf.reset_index(drop=True), train[cols].reset_index(drop=True)
    )
