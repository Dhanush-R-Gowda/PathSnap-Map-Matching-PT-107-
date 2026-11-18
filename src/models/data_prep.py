"""Phase 2 data preparation: load and prepare training/test sets for ML."""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_PARQUET = os.path.join(REPO_ROOT, "data", "processed", "train_candidates.parquet")
TEST_PARQUET = os.path.join(REPO_ROOT, "data", "processed", "student_test_candidates.parquet")

FEATURE_COLUMNS = [
    "snap_distance_m",
    "snap_t",
    "arc_length_m",
    "bearing_diff_rad",
    "abs_bearing_diff_rad",
    "gps_speed_mps",
    "gps_heading_rad",
    "dt_prev_s",
    "dt_next_s",
    "out_degree",
    "in_degree",
    "is_gap_prev",
    "is_gap_next",
    "heading_reliable",
]

META_COLUMNS = ["trip_id", "obs_index", "arc_id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_prepare(
    train_path: str = TRAIN_PARQUET,
    test_path: str = TEST_PARQUET,
) -> dict:
    """Load parquet files and prepare train/test splits for ML.

    Training filter: rows where ``is_gt_in_candidates == True`` only.
    Rows where the GT arc is absent from the candidate set are excluded from
    the classifier training matrix, but the parquet files are not modified.

    Returns
    -------
    dict with keys:
        X_train, y_train, groups_train  — numpy arrays for training
        X_test, y_test, groups_test     — numpy arrays for test
        meta_train                      — DataFrame with trip_id, obs_index, arc_id
        meta_test                       — DataFrame with trip_id, obs_index, arc_id
    """
    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    print(f"Loading training data from: {train_path}")
    train_raw = pd.read_parquet(train_path)
    print(f"  Raw train rows: {len(train_raw):,}")

    print(f"Loading test data from: {test_path}")
    test_raw = pd.read_parquet(test_path)
    print(f"  Raw test rows:  {len(test_raw):,}")

    # ------------------------------------------------------------------
    # Filter: keep only rows where GT arc is in the candidate set
    # ------------------------------------------------------------------
    train_df = train_raw[train_raw["is_gt_in_candidates"] == True].copy()
    test_df = test_raw[test_raw["is_gt_in_candidates"] == True].copy()

    n_train_dropped = len(train_raw) - len(train_df)
    n_test_dropped = len(test_raw) - len(test_df)

    print(f"\nAfter is_gt_in_candidates filter:")
    print(f"  Train rows remaining: {len(train_df):,}  (dropped {n_train_dropped:,} miss rows)")
    print(f"  Test  rows remaining: {len(test_df):,}  (dropped {n_test_dropped:,} miss rows)")

    # ------------------------------------------------------------------
    # Class balance report
    # ------------------------------------------------------------------
    train_pos = int(train_df["label"].sum())
    train_neg = len(train_df) - train_pos
    test_pos = int(test_df["label"].sum())
    test_neg = len(test_df) - test_pos

    print(f"\nClass balance (train): pos={train_pos:,} ({100*train_pos/len(train_df):.2f}%)  "
          f"neg={train_neg:,} ({100*train_neg/len(train_df):.2f}%)")
    print(f"Class balance (test):  pos={test_pos:,} ({100*test_pos/len(test_df):.2f}%)  "
          f"neg={test_neg:,} ({100*test_neg/len(test_df):.2f}%)")

    # ------------------------------------------------------------------
    # Validate feature columns
    # ------------------------------------------------------------------
    missing_train = [c for c in FEATURE_COLUMNS if c not in train_df.columns]
    missing_test = [c for c in FEATURE_COLUMNS if c not in test_df.columns]
    if missing_train:
        raise ValueError(f"Missing feature columns in train: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing feature columns in test: {missing_test}")

    # ------------------------------------------------------------------
    # Separate X, y, groups, meta
    # ------------------------------------------------------------------
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_train = train_df["label"].values.astype(np.int32)
    groups_train = train_df["trip_id"].values
    meta_train = train_df[META_COLUMNS].reset_index(drop=True)

    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float64)
    y_test = test_df["label"].values.astype(np.int32)
    groups_test = test_df["trip_id"].values
    meta_test = test_df[META_COLUMNS].reset_index(drop=True)

    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}  groups_train: {groups_train.shape}")
    print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}  groups_test:  {groups_test.shape}")

    n_train_trips = len(np.unique(groups_train))
    n_test_trips = len(np.unique(groups_test))
    print(f"\n  Unique trips — train: {n_train_trips}  test: {n_test_trips}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "groups_train": groups_train,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
    }


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_and_prepare()
    print("\nData preparation complete.")
    print(f"14 feature columns: {FEATURE_COLUMNS}")
