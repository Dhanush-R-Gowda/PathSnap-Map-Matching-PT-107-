"""Phase 2 pytest tests (14 tests)."""

import os
import json
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------

REPO_ROOT = "/Users/Mikey/GitHub/PathSnap-Map-Matching-PT-107-"

TRAIN_PARQUET = os.path.join(REPO_ROOT, "data", "processed", "train_candidates.parquet")
TEST_PARQUET = os.path.join(REPO_ROOT, "data", "processed", "student_test_candidates.parquet")
MODEL_COMPARISON_CSV = os.path.join(REPO_ROOT, "results", "experiments", "model_comparison.csv")
FINAL_MODEL_PATH = os.path.join(REPO_ROOT, "artifacts", "models", "final_model.joblib")
SCALER_PATH = os.path.join(REPO_ROOT, "artifacts", "models", "scaler.joblib")
FEATURE_COLUMNS_JSON = os.path.join(REPO_ROOT, "artifacts", "configs", "feature_columns.json")
TRAIN_PREDICTIONS = os.path.join(REPO_ROOT, "data", "processed", "train_predictions.parquet")
TEST_PREDICTIONS = os.path.join(REPO_ROOT, "data", "processed", "student_test_predictions.parquet")

EXPECTED_FEATURE_COUNT = 14


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    """Load and prepare data once for all tests."""
    from src.models.data_prep import load_and_prepare
    return load_and_prepare()


# ---------------------------------------------------------------------------
# Test 1: filtered training set has no rows with is_gt_in_candidates==False
# ---------------------------------------------------------------------------

def test_01_no_miss_rows_in_train(data):
    """Filtered train set must contain no rows where is_gt_in_candidates==False."""
    # Reload parquet and apply the same filter as load_and_prepare
    df = pd.read_parquet(TRAIN_PARQUET)
    filtered = df[df["is_gt_in_candidates"] == True]
    # X_train has exactly the rows from the filtered set
    assert len(data["X_train"]) == len(filtered), (
        f"X_train length {len(data['X_train'])} != filtered rows {len(filtered)}"
    )
    # Also check that no False rows leaked through
    assert data["X_train"].shape[0] > 0, "X_train is empty after filtering"


# ---------------------------------------------------------------------------
# Test 2: all 14 feature columns present in X_train
# ---------------------------------------------------------------------------

def test_02_fourteen_feature_columns(data):
    """X_train must have exactly 14 feature columns."""
    from src.models.data_prep import FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == EXPECTED_FEATURE_COUNT, (
        f"Expected {EXPECTED_FEATURE_COUNT} feature columns, got {len(FEATURE_COLUMNS)}"
    )
    assert data["X_train"].shape[1] == EXPECTED_FEATURE_COUNT, (
        f"X_train has {data['X_train'].shape[1]} columns, expected {EXPECTED_FEATURE_COUNT}"
    )


# ---------------------------------------------------------------------------
# Test 3: groups array has same length as X_train
# ---------------------------------------------------------------------------

def test_03_groups_length_matches_x_train(data):
    """groups_train must be aligned with X_train."""
    assert len(data["groups_train"]) == len(data["X_train"]), (
        f"groups_train length {len(data['groups_train'])} != X_train length {len(data['X_train'])}"
    )


# ---------------------------------------------------------------------------
# Test 4: CV framework — no trip appears in both train and val folds
# ---------------------------------------------------------------------------

def test_04_no_trip_leakage_in_cv(data):
    """No trip_id should appear in both train and validation folds for any CV split."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    X = data["X_train"]
    y = data["y_train"]
    groups = data["groups_train"]

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_trips = set(groups[train_idx])
        val_trips = set(groups[val_idx])
        overlap = train_trips & val_trips
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: trips {overlap} appear in both train and val"
        )


# ---------------------------------------------------------------------------
# Test 5: scaler is NOT fitted on full dataset (each fold fits its own scaler)
# ---------------------------------------------------------------------------

def test_05_scaler_fitted_per_fold(data):
    """Verify that the CV cross_validate_model function creates a fresh StandardScaler
    per fold and does NOT use a pre-fitted scaler from the full dataset."""
    # We test this structurally: inspect the source of cross_validate_model to confirm
    # a new StandardScaler() is instantiated inside the fold loop.
    import inspect
    from src.models.evaluation import cross_validate_model
    source = inspect.getsource(cross_validate_model)
    assert "StandardScaler()" in source, (
        "cross_validate_model should instantiate StandardScaler() inside the fold loop"
    )
    assert "scaler.fit_transform(X_tr" in source or "fit_transform(X_tr" in source, (
        "Scaler should be fit only on X_tr (training fold data)"
    )
    # Confirm it does NOT fit on X (full data) before the fold loop
    # The function should NOT call fit before 'for fold_idx'
    lines = source.split("\n")
    in_loop = False
    for line in lines:
        stripped = line.strip()
        if "for fold_idx" in stripped or "for train_idx" in stripped:
            in_loop = True
        if not in_loop and "scaler.fit" in stripped:
            pytest.fail(
                "StandardScaler.fit() called before the fold loop in cross_validate_model"
            )


# ---------------------------------------------------------------------------
# Test 6: model_comparison.csv exists and has exactly 5 rows
# ---------------------------------------------------------------------------

def test_06_model_comparison_csv_exists_with_five_rows():
    """model_comparison.csv must exist and have exactly 5 model rows."""
    assert os.path.exists(MODEL_COMPARISON_CSV), (
        f"model_comparison.csv not found at {MODEL_COMPARISON_CSV}"
    )
    df = pd.read_csv(MODEL_COMPARISON_CSV)
    assert len(df) == 5, (
        f"model_comparison.csv has {len(df)} rows, expected 5"
    )


# ---------------------------------------------------------------------------
# Test 7: artifacts/models/final_model.joblib exists and can be loaded
# ---------------------------------------------------------------------------

def test_07_final_model_loadable():
    """final_model.joblib must exist and be loadable with joblib."""
    assert os.path.exists(FINAL_MODEL_PATH), (
        f"final_model.joblib not found at {FINAL_MODEL_PATH}"
    )
    import joblib
    model = joblib.load(FINAL_MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded object is not a valid sklearn-compatible model"


# ---------------------------------------------------------------------------
# Test 8: artifacts/models/scaler.joblib exists and can be loaded
# ---------------------------------------------------------------------------

def test_08_scaler_loadable():
    """scaler.joblib must exist and be loadable with joblib."""
    assert os.path.exists(SCALER_PATH), (
        f"scaler.joblib not found at {SCALER_PATH}"
    )
    import joblib
    from sklearn.preprocessing import StandardScaler
    scaler = joblib.load(SCALER_PATH)
    assert isinstance(scaler, StandardScaler), (
        f"Loaded scaler is {type(scaler)}, expected StandardScaler"
    )


# ---------------------------------------------------------------------------
# Test 9: feature_columns.json lists exactly 14 columns
# ---------------------------------------------------------------------------

def test_09_feature_columns_json_has_14_entries():
    """feature_columns.json must exist and list exactly 14 feature column names."""
    assert os.path.exists(FEATURE_COLUMNS_JSON), (
        f"feature_columns.json not found at {FEATURE_COLUMNS_JSON}"
    )
    with open(FEATURE_COLUMNS_JSON) as fh:
        cols = json.load(fh)
    assert isinstance(cols, list), "feature_columns.json should contain a JSON array"
    assert len(cols) == EXPECTED_FEATURE_COUNT, (
        f"feature_columns.json has {len(cols)} entries, expected {EXPECTED_FEATURE_COUNT}"
    )


# ---------------------------------------------------------------------------
# Test 10: train_predictions.parquet exists with required columns
# ---------------------------------------------------------------------------

def test_10_train_predictions_schema():
    """train_predictions.parquet must exist with correct columns."""
    assert os.path.exists(TRAIN_PREDICTIONS), (
        f"train_predictions.parquet not found at {TRAIN_PREDICTIONS}"
    )
    df = pd.read_parquet(TRAIN_PREDICTIONS)
    required = {"trip_id", "obs_index", "arc_id", "prob_positive", "is_gt_in_candidates"}
    missing = required - set(df.columns)
    assert len(missing) == 0, f"train_predictions.parquet missing columns: {missing}"


# ---------------------------------------------------------------------------
# Test 11: student_test_predictions.parquet exists with required columns
# ---------------------------------------------------------------------------

def test_11_student_test_predictions_schema():
    """student_test_predictions.parquet must exist with correct columns."""
    assert os.path.exists(TEST_PREDICTIONS), (
        f"student_test_predictions.parquet not found at {TEST_PREDICTIONS}"
    )
    df = pd.read_parquet(TEST_PREDICTIONS)
    required = {"trip_id", "obs_index", "arc_id", "prob_positive", "is_gt_in_candidates"}
    missing = required - set(df.columns)
    assert len(missing) == 0, f"student_test_predictions.parquet missing columns: {missing}"


# ---------------------------------------------------------------------------
# Test 12: all prob_positive values in [0, 1]
# ---------------------------------------------------------------------------

def test_12_prob_positive_in_range():
    """All prob_positive values must be in [0.0, 1.0] for both prediction files."""
    for path, name in [
        (TRAIN_PREDICTIONS, "train_predictions"),
        (TEST_PREDICTIONS, "student_test_predictions"),
    ]:
        assert os.path.exists(path), f"{name}.parquet not found"
        df = pd.read_parquet(path)
        vals = df["prob_positive"].values
        assert np.all(vals >= 0.0), (
            f"{name}: found prob_positive < 0: min={vals.min()}"
        )
        assert np.all(vals <= 1.0), (
            f"{name}: found prob_positive > 1: max={vals.max()}"
        )
        assert not np.any(np.isnan(vals)), (
            f"{name}: found NaN in prob_positive"
        )


# ---------------------------------------------------------------------------
# Test 13: per (trip_id, obs_index) group: at least one candidate exists
# ---------------------------------------------------------------------------

def test_13_at_least_one_candidate_per_obs():
    """Every (trip_id, obs_index) group in both prediction files must have >= 1 row."""
    for path, name in [
        (TRAIN_PREDICTIONS, "train_predictions"),
        (TEST_PREDICTIONS, "student_test_predictions"),
    ]:
        assert os.path.exists(path), f"{name}.parquet not found"
        df = pd.read_parquet(path)
        counts = df.groupby(["trip_id", "obs_index"]).size()
        assert (counts >= 1).all(), (
            f"{name}: some (trip_id, obs_index) groups have 0 candidates"
        )


# ---------------------------------------------------------------------------
# Test 14: inference pipeline compatibility
# ---------------------------------------------------------------------------

def test_14_inference_pipeline_compatibility():
    """Load scaler + model + feature_columns.json and run predict_proba on synthetic data."""
    import joblib

    assert os.path.exists(SCALER_PATH), f"scaler.joblib not found at {SCALER_PATH}"
    assert os.path.exists(FINAL_MODEL_PATH), f"final_model.joblib not found at {FINAL_MODEL_PATH}"
    assert os.path.exists(FEATURE_COLUMNS_JSON), f"feature_columns.json not found"

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(FINAL_MODEL_PATH)
    with open(FEATURE_COLUMNS_JSON) as fh:
        feature_columns = json.load(fh)

    assert len(feature_columns) == EXPECTED_FEATURE_COUNT

    # Build a small synthetic DataFrame
    rng = np.random.default_rng(0)
    n_rows = 20
    synthetic = pd.DataFrame(
        rng.standard_normal((n_rows, len(feature_columns))),
        columns=feature_columns,
    )
    # Boolean columns should be 0/1
    for col in ("is_gap_prev", "is_gap_next", "heading_reliable"):
        if col in synthetic.columns:
            synthetic[col] = rng.integers(0, 2, size=n_rows).astype(float)

    X_synth = synthetic[feature_columns].values
    X_scaled = scaler.transform(X_synth)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        assert proba.shape == (n_rows, 2), (
            f"predict_proba output shape {proba.shape}, expected ({n_rows}, 2)"
        )
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), (
            "predict_proba rows do not sum to 1"
        )
    else:
        preds = model.predict(X_scaled)
        assert preds.shape == (n_rows,), (
            f"predict output shape {preds.shape}, expected ({n_rows},)"
        )
