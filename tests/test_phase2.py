"""Unit tests for the Phase 2 candidate-classification workflow."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, log_loss

from src.models.phase2 import (
    APPROVED_FEATURE_COLUMNS,
    MODEL_NAMES,
    TARGET_COLUMN,
    assert_no_group_leakage,
    dataset_summary,
    evaluate_predictions,
    filter_eligible_observations,
    get_feature_columns,
    load_candidate_data,
    load_model,
    make_grouped_folds,
    make_model_pipeline,
    normalize_candidate_probabilities,
    predict_candidate_probabilities,
    save_model_artifacts,
    sanitize_and_normalize_probabilities,
    select_model,
    validate_candidate_frame,
)


def make_frame(trips=4, observations=4, candidates=2):
    rows = []
    for trip_num in range(trips):
        trip_id = f"{trip_num:08d}"
        for obs in range(observations):
            eligible = not (trip_num == trips - 1 and obs == observations - 1)
            for arc in range(candidates):
                row = {
                    "trip_id": trip_id,
                    "obs_index": obs,
                    "arc_id": arc,
                    "label": int(eligible and arc == (obs % candidates)),
                    "is_gt_in_candidates": eligible,
                }
                for ix, feature in enumerate(APPROVED_FEATURE_COLUMNS):
                    row[feature] = float((trip_num + 1) * (obs + 1) + arc + ix / 10)
                rows.append(row)
    frame = pd.DataFrame(rows)
    return frame[["trip_id", "obs_index", "arc_id", *APPROVED_FEATURE_COLUMNS, "label", "is_gt_in_candidates"]]


def test_parquet_loading_and_dataset_summary(tmp_path):
    frame = make_frame()
    path = tmp_path / "tiny.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_candidate_data(path)
    assert loaded.shape == frame.shape
    summary = dataset_summary(loaded, path, {t: 4 for t in loaded.trip_id.unique()})
    assert summary["trip_count"] == 4
    assert summary["target_column"] == "label"


def test_schema_validation_rejects_missing_and_invalid_targets():
    frame = make_frame()
    validate_candidate_frame(frame)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_candidate_frame(frame.drop(columns=["arc_id"]))
    invalid = frame.copy()
    invalid.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="target"):
        validate_candidate_frame(invalid)


def test_observation_filter_keeps_all_candidates_for_eligible_groups():
    frame = make_frame()
    filtered = filter_eligible_observations(frame)
    excluded_group = (frame.trip_id.eq("00000003") & frame.obs_index.eq(3))
    assert not ((filtered.trip_id == "00000003") & (filtered.obs_index == 3)).any()
    assert len(filtered) == int((~excluded_group).sum())
    assert (filtered.groupby(["trip_id", "obs_index"]).size() == 2).all()
    assert (filtered.groupby(["trip_id", "obs_index"]).label.sum() == 1).all()


def test_feature_selection_excludes_identifiers_and_targets():
    frame = make_frame()
    features = get_feature_columns(frame)
    assert features == APPROVED_FEATURE_COLUMNS
    assert not set(features) & {"trip_id", "obs_index", "arc_id", "label", "is_gt_in_candidates"}


def test_missing_value_handling_is_fitted_inside_pipeline():
    frame = make_frame()
    frame.loc[0, "snap_distance_m"] = np.nan
    model = make_model_pipeline("logistic_regression", {"max_iter": 30})
    model.fit(frame[APPROVED_FEATURE_COLUMNS], frame.label)
    probability = model.predict_proba(frame[APPROVED_FEATURE_COLUMNS])
    assert np.isfinite(probability).all()
    imputer = model.named_steps["preprocess"].named_steps["imputer"]
    assert np.isfinite(imputer.statistics_[0])


def test_groupkfold_keeps_trips_separate():
    frame = make_frame()
    folds = make_grouped_folds(frame, sorted(frame.trip_id.unique()), n_splits=2)
    assert len(folds) == 2
    for fold in folds:
        assert_no_group_leakage(fold["train_trips"], fold["validation_trips"])
        assert set(frame.iloc[fold["train_rows"]].trip_id).issubset(set(fold["train_trips"]))
        assert set(frame.iloc[fold["validation_rows"]].trip_id).issubset(set(fold["validation_trips"]))


def test_model_selection_accounts_for_training_and_inference_time():
    comparison = pd.DataFrame([
        {
            "model": "slow", "mean_top1_accuracy": 0.8, "mean_macro_f1": 0.8,
            "mean_route_continuity": 0.8, "mean_log_loss": 0.2, "std_macro_f1": 0.01,
            "cv_training_time_s": 100.0, "cv_inference_time_s": 10.0,
        },
        {
            "model": "fast", "mean_top1_accuracy": 0.8, "mean_macro_f1": 0.8,
            "mean_route_continuity": 0.8, "mean_log_loss": 0.2, "std_macro_f1": 0.01,
            "cv_training_time_s": 10.0, "cv_inference_time_s": 1.0,
        },
    ])
    name, details = select_model(comparison)
    assert name == "fast"
    assert "cv_training_time_s" in details["weights"]
    assert "cv_inference_time_s" in details["weights"]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_all_five_model_pipelines_fit_and_predict(model_name):
    frame = make_frame(trips=3, observations=3)
    params = {
        "lightgbm": {"n_estimators": 4, "n_jobs": 1},
        "random_forest": {"n_estimators": 4, "n_jobs": 1},
        "logistic_regression": {"max_iter": 30},
        "shallow_mlp": {"max_iter": 5, "batch_size": 16, "n_iter_no_change": 1, "tol": 1e9},
        "gaussian_naive_bayes": {},
    }[model_name]
    model = make_model_pipeline(model_name, params)
    model.fit(frame[APPROVED_FEATURE_COLUMNS], frame.label)
    probabilities = model.predict_proba(frame[APPROVED_FEATURE_COLUMNS])
    assert probabilities.shape == (len(frame), 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_probability_normalization_handles_invalid_and_empty_sets():
    values = sanitize_and_normalize_probabilities([0.7, 0.2, 0.1])
    assert np.allclose(values, [0.7, 0.2, 0.1])
    invalid = sanitize_and_normalize_probabilities([float("nan"), -4.0, float("inf")])
    assert np.allclose(invalid, [1 / 3] * 3)
    assert sanitize_and_normalize_probabilities([]).size == 0

    candidates = pd.DataFrame({
        "trip_id": ["a", "a", "b"],
        "obs_index": [0, 0, 0],
        "raw_probability": [0.25, 0.75, np.nan],
    })
    normalized = normalize_candidate_probabilities(candidates)
    assert np.allclose(normalized.groupby(["trip_id", "obs_index"]).probability.sum(), 1.0)


def test_metric_calculation_and_trip_scoped_arc_connectivity():
    frame = pd.DataFrame([
        {"trip_id": "trip_a", "obs_index": 0, "arc_id": 0, "label": 1, "is_gt_in_candidates": True},
        {"trip_id": "trip_a", "obs_index": 0, "arc_id": 2, "label": 0, "is_gt_in_candidates": True},
        {"trip_id": "trip_a", "obs_index": 1, "arc_id": 1, "label": 1, "is_gt_in_candidates": True},
        {"trip_id": "trip_a", "obs_index": 1, "arc_id": 3, "label": 0, "is_gt_in_candidates": True},
    ])
    truth = frame.label.to_numpy()
    probs = np.array([0.9, 0.1, 0.8, 0.2])
    context = {"trip_a": {"observation_count": 3, "arcs": [(0, 1), (1, 2), (7, 8), (8, 9)]}}
    metrics = evaluate_predictions(frame, probs, context)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["macro_f1"] == f1_score(truth, probs >= 0.5, average="macro")
    assert metrics["log_loss"] == pytest.approx(log_loss(truth, probs, labels=[0, 1]))
    assert metrics["candidate_miss_rate"] == pytest.approx(1 / 3)
    assert metrics["route_continuity"] == 1.0
    assert metrics["expected_calibration_error"] is not None


def test_candidate_mapping_preserves_trip_and_empty_observation(tmp_path):
    frame = make_frame(trips=3, observations=3)
    model = make_model_pipeline("logistic_regression", {"max_iter": 30})
    model.fit(frame[APPROVED_FEATURE_COLUMNS], frame.label)
    subset = frame[(frame.trip_id == "00000000") & (frame.obs_index < 2)].copy()
    expected = pd.DataFrame({"trip_id": ["00000000"] * 3, "obs_index": [0, 1, 2]})
    predicted, records = predict_candidate_probabilities(model, subset, APPROVED_FEATURE_COLUMNS, expected)
    assert predicted.groupby(["trip_id", "obs_index"]).probability.sum().to_numpy() == pytest.approx([1.0, 1.0])
    assert [r["observation_id"] for r in records] == [0, 1, 2]
    assert records[-1]["candidates"] == []
    assert all(r["trip_id"] == "00000000" for r in records)
    assert all(sum(c["probability"] for c in r["candidates"]) == pytest.approx(1.0) for r in records[:2])


def test_model_artifact_saving_and_loading_is_prediction_consistent(tmp_path):
    frame = make_frame(trips=3, observations=3)
    model = make_model_pipeline("logistic_regression", {"max_iter": 30})
    model.fit(frame[APPROVED_FEATURE_COLUMNS], frame.label)
    before = model.predict_proba(frame[APPROVED_FEATURE_COLUMNS])
    paths = save_model_artifacts(
        model,
        APPROVED_FEATURE_COLUMNS,
        "logistic_regression",
        {"C": 1.0},
        {"random_seed": 42},
        tmp_path,
    )
    loaded = load_model(paths["best_model"])
    after = loaded.predict_proba(frame[APPROVED_FEATURE_COLUMNS])
    assert np.allclose(before, after)
    assert (tmp_path / "preprocessing_pipeline.pkl").is_file()
    assert json.loads((tmp_path / "feature_schema.json").read_text())["feature_columns"] == APPROVED_FEATURE_COLUMNS

\n