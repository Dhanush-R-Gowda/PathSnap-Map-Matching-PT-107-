"""Phase 2 candidate-level map-matching models and evaluation utilities.

This module deliberately keeps trip, observation, arc and label columns out of
the feature matrix.  All fitted transformations live inside sklearn pipelines
so they are learned only from the training portion of each grouped fold.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


TRIP_COLUMN = "trip_id"
OBSERVATION_COLUMN = "obs_index"
ARC_COLUMN = "arc_id"
TARGET_COLUMN = "label"
ELIGIBLE_COLUMN = "is_gt_in_candidates"
IDENTIFIER_COLUMNS = [TRIP_COLUMN, OBSERVATION_COLUMN, ARC_COLUMN]
EVALUATION_ONLY_COLUMNS = [TARGET_COLUMN, ELIGIBLE_COLUMN]

# Exact approved Phase 1 feature schema from src/features/build_features.py.
APPROVED_FEATURE_COLUMNS = [
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

DEFAULT_SEED = 42
MODEL_NAMES = [
    "lightgbm",
    "random_forest",
    "logistic_regression",
    "shallow_mlp",
    "gaussian_naive_bayes",
]


def json_safe(value: Any) -> Any:
    """Convert common NumPy/pandas values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_candidate_data(path: str | Path) -> pd.DataFrame:
    """Read and structurally validate a Phase 1 candidate Parquet file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate Parquet file not found: {path}")
    frame = pd.read_parquet(path)
    validate_candidate_frame(frame, source=str(path))
    return frame


def validate_candidate_frame(frame: pd.DataFrame, source: str = "dataframe") -> None:
    """Validate identifiers, target, flags and Phase 1 feature availability."""
    required = set(IDENTIFIER_COLUMNS + [TARGET_COLUMN, ELIGIBLE_COLUMN])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    missing_features = [c for c in APPROVED_FEATURE_COLUMNS if c not in frame.columns]
    if missing_features:
        raise ValueError(f"{source} is missing approved Phase 1 features: {missing_features}")
    if frame.empty:
        raise ValueError(f"{source} contains no candidate rows")
    if frame[IDENTIFIER_COLUMNS].isna().any().any():
        raise ValueError(f"{source} has null trip, observation, or arc identifiers")
    if frame[TRIP_COLUMN].astype(str).str.len().eq(0).any():
        raise ValueError(f"{source} contains an empty trip_id")
    labels = set(pd.unique(frame[TARGET_COLUMN].dropna()).tolist())
    if frame[TARGET_COLUMN].isna().any() or not labels.issubset({0, 1}):
        raise ValueError(f"{source} target must contain only non-null 0/1 values; got {labels}")
    if frame[ELIGIBLE_COLUMN].isna().any():
        raise ValueError(f"{source} has null {ELIGIBLE_COLUMN} values")
    flag_values = set(pd.unique(frame[ELIGIBLE_COLUMN]).tolist())
    if not flag_values.issubset({True, False, 0, 1}):
        raise ValueError(f"{source} {ELIGIBLE_COLUMN} must be boolean")
    if frame.duplicated(IDENTIFIER_COLUMNS).any():
        raise ValueError(f"{source} has duplicate (trip_id, obs_index, arc_id) records")

    flag_counts = frame.groupby([TRIP_COLUMN, OBSERVATION_COLUMN], sort=False)[ELIGIBLE_COLUMN].nunique()
    if (flag_counts > 1).any():
        raise ValueError(f"{source} has inconsistent eligibility flags within observations")
    per_observation_positive = frame.groupby(
        [TRIP_COLUMN, OBSERVATION_COLUMN], sort=False
    )[TARGET_COLUMN].sum()
    eligible = frame.groupby([TRIP_COLUMN, OBSERVATION_COLUMN], sort=False)[ELIGIBLE_COLUMN].first().astype(bool)
    if (per_observation_positive[eligible] != 1).any():
        raise ValueError(f"{source} eligible observations must have exactly one positive candidate")
    if (per_observation_positive[~eligible] != 0).any():
        raise ValueError(f"{source} ineligible observations must have no positive candidate")

    non_numeric = [c for c in APPROVED_FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"{source} approved features must be numeric: {non_numeric}")


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the approved Phase 1 feature list in its defined order."""
    missing = [c for c in APPROVED_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Cannot select features; missing approved columns: {missing}")
    forbidden = set(IDENTIFIER_COLUMNS + EVALUATION_ONLY_COLUMNS)
    features = [c for c in APPROVED_FEATURE_COLUMNS if c not in forbidden]
    if not features:
        raise ValueError("No approved features remain after excluding identifiers and evaluation fields")
    return features


def filter_eligible_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep every candidate row for observations whose ground truth is present."""
    validate_candidate_frame(frame)
    group_keys = [TRIP_COLUMN, OBSERVATION_COLUMN]
    keep = frame.groupby(group_keys, sort=False)[ELIGIBLE_COLUMN].transform("first").astype(bool)
    eligible = frame.loc[keep].copy()
    # Guard against an accidental row-level filter that retains only positives.
    if len(eligible) and (eligible.groupby(group_keys)[TARGET_COLUMN].sum() != 1).any():
        raise AssertionError("Observation filtering did not retain the complete eligible candidate set")
    return eligible


def dataset_summary(frame: pd.DataFrame, path: str | Path, raw_observation_counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Collect a schema and data-quality summary for one candidate dataset."""
    validate_candidate_frame(frame, source=str(path))
    keys = [TRIP_COLUMN, OBSERVATION_COLUMN]
    candidate_counts = frame.groupby(keys, sort=False).size()
    obs_eligible = frame.groupby(keys, sort=False)[ELIGIBLE_COLUMN].first().astype(bool)
    positives = int((frame[TARGET_COLUMN] == 1).sum())
    row_miss_obs = int((~obs_eligible).sum())
    raw_total = sum(raw_observation_counts.values()) if raw_observation_counts else int(len(obs_eligible))
    candidate_miss_count = max(0, raw_total - positives)
    numeric = frame[get_feature_columns(frame)].replace([np.inf, -np.inf], np.nan)
    result = {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_types": {c: str(t) for c, t in frame.dtypes.items()},
        "trip_count": int(frame[TRIP_COLUMN].nunique()),
        "trip_ids": sorted(frame[TRIP_COLUMN].astype(str).unique().tolist()),
        "candidate_observation_count": int(len(candidate_counts)),
        "raw_observation_count": int(raw_total),
        "eligible_observation_count": int(positives),
        "ineligible_observations_with_candidates": row_miss_obs,
        "observations_without_candidate_rows": int(max(0, raw_total - len(candidate_counts))),
        "excluded_observation_count": int(candidate_miss_count),
        "candidate_miss_count": int(candidate_miss_count),
        "candidate_miss_rate": float(candidate_miss_count / raw_total) if raw_total else None,
        "candidate_rows_before_filtering": int(len(frame)),
        "candidate_rows_after_filtering": int(int(eligible_candidate_row_count(frame))),
        "candidate_rows_per_observation": {
            "min": int(candidate_counts.min()),
            "mean": float(candidate_counts.mean()),
            "median": float(candidate_counts.median()),
            "max": int(candidate_counts.max()),
        },
        "target_column": TARGET_COLUMN,
        "target_distribution": {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().sort_index().items()},
        "eligible_observation_distribution": {str(k): int(v) for k, v in obs_eligible.value_counts().items()},
        "feature_columns": get_feature_columns(frame),
        "feature_types": {c: str(frame[c].dtype) for c in get_feature_columns(frame)},
        "feature_missing_values": {c: int(frame[c].isna().sum()) for c in get_feature_columns(frame)},
        "feature_invalid_numeric_values": {
            c: int((~np.isfinite(pd.to_numeric(numeric[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan))).sum())
            for c in get_feature_columns(frame)
        },
        "missing_values_all_columns": {c: int(v) for c, v in frame.isna().sum().items()},
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "duplicate_candidate_keys": int(frame.duplicated(IDENTIFIER_COLUMNS).sum()),
    }
    return result


def eligible_candidate_row_count(frame: pd.DataFrame) -> int:
    """Count candidate rows retained by observation-level eligibility filtering."""
    mask = frame.groupby([TRIP_COLUMN, OBSERVATION_COLUMN], sort=False)[ELIGIBLE_COLUMN].transform("first").astype(bool)
    return int(mask.sum())


def make_grouped_folds(
    frame: pd.DataFrame,
    trip_ids: Sequence[str] | None = None,
    n_splits: int = 5,
) -> list[dict[str, Any]]:
    """Make deterministic GroupKFold splits over whole trips, including trips
    with no eligible observations. Training row indices are filtered afterward.
    """
    all_trip_ids = sorted(map(str, trip_ids or frame[TRIP_COLUMN].astype(str).unique().tolist()))
    if len(all_trip_ids) < n_splits:
        raise ValueError(f"GroupKFold requires at least {n_splits} trips; found {len(all_trip_ids)}")
    splitter = GroupKFold(n_splits=n_splits)
    x_dummy = np.zeros((len(all_trip_ids), 1), dtype=np.int8)
    groups = np.asarray(all_trip_ids, dtype=object)
    folds: list[dict[str, Any]] = []
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(x_dummy, groups=groups), start=1):
        train_trips = set(groups[train_idx].tolist())
        val_trips = set(groups[val_idx].tolist())
        assert_no_group_leakage(train_trips, val_trips)
        row_trip_ids = frame[TRIP_COLUMN].astype(str).to_numpy()
        row_eligible = frame[ELIGIBLE_COLUMN].astype(bool).to_numpy()
        train_rows = np.flatnonzero(np.isin(row_trip_ids, list(train_trips)) & row_eligible)
        val_rows = np.flatnonzero(np.isin(row_trip_ids, list(val_trips)))
        folds.append({
            "fold": fold_number,
            "train_trips": sorted(train_trips),
            "validation_trips": sorted(val_trips),
            "train_rows": train_rows,
            "validation_rows": val_rows,
        })
    return folds


def assert_no_group_leakage(train_groups: Iterable[Any], validation_groups: Iterable[Any]) -> None:
    overlap = set(train_groups) & set(validation_groups)
    if overlap:
        raise AssertionError(f"Group leakage detected: {sorted(map(str, overlap))}")


def make_model_pipeline(model_name: str, params: Mapping[str, Any] | None = None, seed: int = DEFAULT_SEED) -> Pipeline:
    """Create one of the five required estimators with fold-fitted preprocessing."""
    params = dict(params or {})
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")

    scale = model_name in {"logistic_regression", "shallow_mlp"}
    preprocess_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale:
        preprocess_steps.append(("scaler", StandardScaler()))
    preprocessing = Pipeline(preprocess_steps)

    if model_name == "lightgbm":
        defaults = dict(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=int(os.environ.get("PATHSNAP_N_JOBS", "-1")),
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        defaults.update(params)
        estimator = LGBMClassifier(**defaults)
    elif model_name == "random_forest":
        defaults = dict(
            n_estimators=40,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight=None,
            random_state=seed,
            n_jobs=max(1, min(2, int(os.environ.get("PATHSNAP_N_JOBS", "1")))),
        )
        defaults.update(params)
        estimator = RandomForestClassifier(**defaults)
    elif model_name == "logistic_regression":
        defaults = dict(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
        defaults.update(params)
        estimator = LogisticRegression(**defaults)
    elif model_name == "shallow_mlp":
        defaults = dict(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=8192,
            learning_rate_init=0.001,
            max_iter=35,
            early_stopping=False,
            random_state=seed,
            shuffle=True,
            tol=1e-4,
        )
        defaults.update(params)
        estimator = MLPClassifier(**defaults)
    else:
        defaults = dict(var_smoothing=1e-9)
        defaults.update(params)
        estimator = GaussianNB(**defaults)

    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def positive_class_probabilities(model: Any, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return P(label=1), respecting the estimator's learned class ordering."""
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(positive) != 1:
        raise ValueError(f"Model does not expose a unique positive class 1: {classes.tolist()}")
    return probabilities[:, int(positive[0])]


def sanitize_and_normalize_probabilities(values: Sequence[float], epsilon: float = 1e-12) -> np.ndarray:
    """Clip invalid/negative scores to zero, then normalize or return uniform."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Candidate probabilities must be a one-dimensional sequence")
    if arr.size == 0:
        return arr
    clean = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    total = float(clean.sum())
    if not math.isfinite(total) or total <= epsilon:
        return np.full(arr.size, 1.0 / arr.size, dtype=np.float64)
    normalized = clean / total
    # Correct any tiny floating-point drift in the final element.
    normalized[-1] += 1.0 - float(normalized.sum())
    return normalized


def normalize_candidate_probabilities(
    candidates: pd.DataFrame,
    probability_column: str = "raw_probability",
    output_column: str = "probability",
    group_columns: Sequence[str] = (TRIP_COLUMN, OBSERVATION_COLUMN),
) -> pd.DataFrame:
    """Normalize classifier scores inside each trip/observation candidate set."""
    required = set(group_columns) | {probability_column}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Cannot normalize candidate probabilities; missing columns: {missing}")
    result = candidates.copy()
    result[output_column] = 0.0
    if result.empty:
        return result
    raw = pd.to_numeric(result[probability_column], errors="coerce").to_numpy(dtype=np.float64)
    clean = np.where(np.isfinite(raw) & (raw > 0), raw, 0.0)
    group_codes = result.groupby(list(group_columns), sort=False, dropna=False).ngroup().to_numpy(dtype=np.int64)
    group_count = int(group_codes.max()) + 1 if len(group_codes) else 0
    totals = np.bincount(group_codes, weights=clean, minlength=group_count)
    counts = np.bincount(group_codes, minlength=group_count)
    valid = totals[group_codes] > 1e-12
    normalized = np.empty(len(result), dtype=np.float64)
    normalized[valid] = clean[valid] / totals[group_codes[valid]]
    normalized[~valid] = 1.0 / counts[group_codes[~valid]]
    result[output_column] = normalized
    return result


def predict_candidate_probabilities(
    model: Any,
    candidate_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    observations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Predict and normalize within observation; optionally include empty sets.

    ``observations`` may contain all expected trip_id/obs_index pairs from raw
    tracks. Observations with no candidate rows then appear with candidates=[].
    """
    required = set(IDENTIFIER_COLUMNS + list(feature_columns))
    missing = sorted(required - set(candidate_frame.columns))
    if missing:
        raise ValueError(f"Inference frame is missing columns: {missing}")
    result = candidate_frame.copy()
    result["_input_order"] = np.arange(len(result), dtype=np.int64)
    if len(result):
        x = result[list(feature_columns)].replace([np.inf, -np.inf], np.nan)
        result["raw_probability"] = positive_class_probabilities(model, x)
        result = normalize_candidate_probabilities(result)
        result.sort_values([TRIP_COLUMN, OBSERVATION_COLUMN, "_input_order"], kind="stable", inplace=True)
    else:
        result["raw_probability"] = pd.Series(dtype=float)
        result["probability"] = pd.Series(dtype=float)

    expected = observations
    if expected is None:
        expected = result[[TRIP_COLUMN, OBSERVATION_COLUMN]].drop_duplicates()
    expected = expected[[TRIP_COLUMN, OBSERVATION_COLUMN]].drop_duplicates().copy()
    expected[TRIP_COLUMN] = expected[TRIP_COLUMN].astype(str)
    expected.sort_values([TRIP_COLUMN, OBSERVATION_COLUMN], kind="stable", inplace=True)

    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    if len(result):
        for row in result.itertuples(index=False):
            grouped[(str(getattr(row, TRIP_COLUMN)), getattr(row, OBSERVATION_COLUMN))].append({
                "arc_id": int(getattr(row, ARC_COLUMN)),
                "probability": float(getattr(row, "probability")),
            })
    records: list[dict[str, Any]] = []
    for trip_id, obs_id in expected.itertuples(index=False, name=None):
        records.append({
            "trip_id": str(trip_id),
            "observation_id": int(obs_id),
            "candidates": grouped.get((str(trip_id), obs_id), []),
        })
    result.drop(columns=["_input_order"], inplace=True, errors="ignore")
    return result, records


def load_raw_context(raw_root: str | Path, partition: str, trip_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Read observation counts and directed arc endpoints from Phase 1 raw files."""
    from src.data.loader import load_arcs, load_track

    raw_root = Path(raw_root)
    context: dict[str, dict[str, Any]] = {}
    for trip_id in sorted(set(map(str, trip_ids))):
        trip_dir = raw_root / partition / trip_id
        if not trip_dir.is_dir():
            raise FileNotFoundError(f"Raw {partition} trip directory missing: {trip_dir}")
        track = load_track(str(trip_dir))
        arcs = load_arcs(str(trip_dir))
        context[trip_id] = {"observation_count": len(track), "arcs": arcs}
    return context


def candidate_miss_metrics(
    candidates: pd.DataFrame,
    raw_context: Mapping[str, Mapping[str, Any]],
    trip_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Count GT misses against raw GPS observations, including empty candidate sets."""
    selected = sorted(set(map(str, trip_ids if trip_ids is not None else raw_context.keys())))
    raw_total = sum(int(raw_context[t]["observation_count"]) for t in selected)
    if len(candidates):
        observation_flags = candidates.groupby(
            [TRIP_COLUMN, OBSERVATION_COLUMN], sort=False
        )[ELIGIBLE_COLUMN].first().astype(bool)
        eligible_by_trip = observation_flags.groupby(level=0).sum().to_dict()
    else:
        eligible_by_trip = {}
    eligible_total = sum(int(eligible_by_trip.get(t, 0)) for t in selected)
    miss_count = max(0, raw_total - eligible_total)
    return {
        "raw_observation_count": int(raw_total),
        "eligible_observation_count": int(eligible_total),
        "candidate_miss_count": int(miss_count),
        "candidate_miss_rate": float(miss_count / raw_total) if raw_total else None,
    }


def route_continuity_metrics(
    candidates_with_probabilities: pd.DataFrame,
    raw_context: Mapping[str, Mapping[str, Any]],
    trip_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Measure connected pairs among predictions for consecutive GPS indices.

    Formula: connected predicted arc pairs / consecutive observation pairs for
    which both observations have at least one candidate. A pair connects when
    arc_t.to_node == arc_(t+1).from_node in that trip's directed network.
    """
    selected = sorted(set(map(str, trip_ids if trip_ids is not None else raw_context.keys())))
    if candidates_with_probabilities.empty:
        return {"connected_pairs": 0, "evaluated_pairs": 0, "route_continuity": None}
    top_indices = candidates_with_probabilities.groupby(
        [TRIP_COLUMN, OBSERVATION_COLUMN], sort=False
    )["raw_probability"].idxmax()
    top = candidates_with_probabilities.loc[top_indices, [TRIP_COLUMN, OBSERVATION_COLUMN, ARC_COLUMN]]
    by_trip: dict[str, dict[int, int]] = defaultdict(dict)
    for row in top.itertuples(index=False, name=None):
        trip_id, obs_id, arc_id = row
        by_trip[str(trip_id)][int(obs_id)] = int(arc_id)

    connected = 0
    evaluated = 0
    for trip_id in selected:
        obs_to_arc = by_trip.get(trip_id, {})
        arcs = raw_context[trip_id]["arcs"]
        obs_count = int(raw_context[trip_id]["observation_count"])
        for obs_id in range(max(0, obs_count - 1)):
            if obs_id not in obs_to_arc or obs_id + 1 not in obs_to_arc:
                continue
            arc_a = obs_to_arc[obs_id]
            arc_b = obs_to_arc[obs_id + 1]
            evaluated += 1
            if 0 <= arc_a < len(arcs) and 0 <= arc_b < len(arcs) and arcs[arc_a][1] == arcs[arc_b][0]:
                connected += 1
    return {
        "connected_pairs": int(connected),
        "evaluated_pairs": int(evaluated),
        "route_continuity": float(connected / evaluated) if evaluated else None,
    }


def _top1_accuracy(frame: pd.DataFrame, probability: np.ndarray) -> float | None:
    if frame.empty:
        return None
    work = frame[[TRIP_COLUMN, OBSERVATION_COLUMN, ARC_COLUMN, TARGET_COLUMN]].copy()
    work["_prob"] = probability
    work["_order"] = np.arange(len(work))
    picked = work.sort_values(["_prob", "_order"], ascending=[False, True], kind="stable").drop_duplicates(
        [TRIP_COLUMN, OBSERVATION_COLUMN], keep="first"
    )
    return float(picked[TARGET_COLUMN].mean())


def expected_calibration_error(y_true: Sequence[int], probability: Sequence[float], bins: int = 10) -> float | None:
    """Return equal-width-bin ECE, using held-out predictions only."""
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    valid = np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0.0, 1.0)
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (p >= edges[index]) & (p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1])
        if selected.any():
            ece += float(selected.mean()) * abs(float(p[selected].mean()) - float(y[selected].mean()))
    return float(ece)


def evaluate_predictions(
    candidates: pd.DataFrame,
    raw_probability: Sequence[float],
    raw_context: Mapping[str, Mapping[str, Any]],
    trip_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute candidate-level and observation/route metrics for a prediction set."""
    if len(candidates) != len(raw_probability):
        raise ValueError("Candidate rows and probabilities have different lengths")
    selected = sorted(set(map(str, trip_ids if trip_ids is not None else candidates[TRIP_COLUMN].astype(str).unique())))
    trip_mask = candidates[TRIP_COLUMN].astype(str).isin(selected).to_numpy()
    frame = candidates.loc[trip_mask].copy()
    probs = np.asarray(raw_probability, dtype=np.float64)[trip_mask]
    safe = np.clip(np.where(np.isfinite(probs), probs, 0.0), 1e-15, 1.0 - 1e-15)
    eligible_mask = frame[ELIGIBLE_COLUMN].astype(bool).to_numpy()
    eligible = frame.loc[eligible_mask]
    eligible_p = safe[eligible_mask]
    if len(eligible):
        y_true = eligible[TARGET_COLUMN].astype(int).to_numpy()
        candidate_prediction = eligible_p >= 0.5
        binary_macro_f1 = float(f1_score(y_true, candidate_prediction, average="macro", zero_division=0))
        candidate_precision = float(precision_score(y_true, candidate_prediction, pos_label=1, zero_division=0))
        candidate_accuracy = float(accuracy_score(y_true, candidate_prediction))
        binary_log_loss = float(log_loss(y_true, eligible_p, labels=[0, 1]))
        brier = float(np.mean((eligible_p - y_true) ** 2))
        ece = expected_calibration_error(y_true, eligible_p)
        top1 = _top1_accuracy(eligible, eligible_p)
        normalized = normalize_candidate_probabilities(
            eligible.assign(raw_probability=eligible_p), probability_column="raw_probability"
        )
        correct_by_obs = normalized.loc[normalized[TARGET_COLUMN].astype(int).eq(1), "probability"].to_numpy()
        observation_log_loss = float(-np.log(np.clip(correct_by_obs, 1e-15, 1.0)).mean()) if len(correct_by_obs) else None
    else:
        binary_macro_f1 = candidate_precision = candidate_accuracy = None
        binary_log_loss = brier = ece = top1 = observation_log_loss = None

    miss = candidate_miss_metrics(frame, raw_context, selected)
    continuity = route_continuity_metrics(
        frame.assign(raw_probability=safe), raw_context, selected
    )
    return {
        "top1_accuracy": top1,
        "macro_f1": binary_macro_f1,
        "candidate_positive_precision": candidate_precision,
        "candidate_binary_accuracy": candidate_accuracy,
        "log_loss": binary_log_loss,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "normalized_observation_log_loss": observation_log_loss,
        **miss,
        **continuity,
        "candidate_rows_evaluated": int(len(frame)),
        "eligible_candidate_rows_evaluated": int(len(eligible)),
    }


def _fold_metric_summary(records: Sequence[Mapping[str, Any]], metric: str) -> tuple[float | None, float | None]:
    values = [float(r[metric]) for r in records if r.get(metric) is not None and math.isfinite(float(r[metric]))]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def cross_validate_model(
    model_name: str,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    raw_context: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    store_oof: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Fit and evaluate one model on the supplied common GroupKFold splits."""
    all_trip_ids = sorted(frame[TRIP_COLUMN].astype(str).unique().tolist())
    oof = np.full(len(frame), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    per_trip_fold: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    total_inference_seconds = 0.0
    features = list(feature_columns)

    for fold in folds:
        train_idx = np.asarray(fold["train_rows"], dtype=np.int64)
        val_idx = np.asarray(fold["validation_rows"], dtype=np.int64)
        if not len(train_idx):
            raise ValueError(f"Fold {fold['fold']} contains no eligible training rows")
        if not len(val_idx):
            raise ValueError(f"Fold {fold['fold']} contains no validation candidate rows")
        assert_no_group_leakage(fold["train_trips"], fold["validation_trips"])
        x_train = frame.iloc[train_idx][features].replace([np.inf, -np.inf], np.nan)
        y_train = frame.iloc[train_idx][TARGET_COLUMN].astype(np.int8).to_numpy()
        x_val = frame.iloc[val_idx][features].replace([np.inf, -np.inf], np.nan)
        start = time.perf_counter()
        model = make_model_pipeline(model_name, params, seed)
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        p_val = positive_class_probabilities(model, x_val)
        inference_seconds = time.perf_counter() - start
        if store_oof:
            oof[val_idx] = p_val.astype(np.float32)
        total_fit_seconds += fit_seconds
        total_inference_seconds += inference_seconds

        val_frame = frame.iloc[val_idx]
        metrics = evaluate_predictions(val_frame, p_val, raw_context, fold["validation_trips"])
        row = {
            "model": model_name,
            "row_type": "fold",
            "fold": int(fold["fold"]),
            "trip_id": None,
            "training_trip_ids": json.dumps(fold["train_trips"]),
            "validation_trip_ids": json.dumps(fold["validation_trips"]),
            "training_row_count": int(len(train_idx)),
            "validation_row_count": int(len(val_idx)),
            "training_time_s": float(fit_seconds),
            "inference_time_s": float(inference_seconds),
            **metrics,
        }
        fold_rows.append(row)
        if progress:
            print(
                f"  {model_name} fold {fold['fold']}/{len(folds)}: "
                f"Macro F1={metrics['macro_f1']:.4f} Top-1={metrics['top1_accuracy']:.4f} "
                f"LogLoss={metrics['log_loss']:.4f} train={fit_seconds:.1f}s infer={inference_seconds:.1f}s",
                flush=True,
            )

        # Per-trip validation metrics preserve each trip's identity.
        p_by_index = pd.Series(p_val, index=val_idx)
        for trip_id in fold["validation_trips"]:
            trip_frame = val_frame.loc[val_frame[TRIP_COLUMN].astype(str).eq(str(trip_id))]
            trip_probs = p_by_index.loc[trip_frame.index].to_numpy()
            trip_metric = evaluate_predictions(trip_frame, trip_probs, raw_context, [str(trip_id)])
            per_trip_fold.append({
                "model": model_name,
                "row_type": "per_trip",
                "fold": int(fold["fold"]),
                "trip_id": str(trip_id),
                "training_trip_ids": json.dumps(fold["train_trips"]),
                "validation_trip_ids": json.dumps(fold["validation_trips"]),
                "training_row_count": int(len(train_idx)),
                "validation_row_count": int(len(trip_frame)),
                "training_time_s": None,
                "inference_time_s": None,
                **trip_metric,
            })
        del model, x_train, x_val
        gc.collect()

    if store_oof and np.isnan(oof).any():
        missing_rows = int(np.isnan(oof).sum())
        raise AssertionError(f"OOF probability array has {missing_rows} unpredicted candidate rows")
    oof_float = oof.astype(np.float64)
    pooled = evaluate_predictions(frame, oof_float, raw_context, all_trip_ids) if store_oof else {}
    for row in fold_rows:
        row["oof_pooled_top1_accuracy"] = pooled.get("top1_accuracy")
        row["oof_pooled_macro_f1"] = pooled.get("macro_f1")
        row["oof_pooled_log_loss"] = pooled.get("log_loss")
    summary: dict[str, Any] = {
        "model": model_name,
        "fold_metrics": fold_rows,
        "per_trip_metrics": per_trip_fold,
        "pooled_metrics": pooled,
        "cv_training_time_s": float(total_fit_seconds),
        "cv_inference_time_s": float(total_inference_seconds),
    }
    for metric in ["top1_accuracy", "macro_f1", "candidate_positive_precision", "candidate_binary_accuracy", "log_loss", "brier_score", "expected_calibration_error", "candidate_miss_rate", "route_continuity", "normalized_observation_log_loss"]:
        mean, std = _fold_metric_summary(fold_rows, metric)
        summary[f"mean_{metric}"] = mean
        summary[f"std_{metric}"] = std
    if store_oof:
        summary["oof_probabilities"] = oof_float
    return summary


def select_model(comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Rank models across quality, stability and inference cost for selection."""
    metrics = {
        "mean_top1_accuracy": (0.23, False),
        "mean_macro_f1": (0.23, False),
        "mean_route_continuity": (0.18, False),
        "mean_log_loss": (0.14, True),
        "std_macro_f1": (0.10, True),
        "cv_training_time_s": (0.07, True),
        "cv_inference_time_s": (0.05, True),
    }
    scores = pd.Series(0.0, index=comparison.index)
    weights = 0.0
    rank_breakdown: dict[str, dict[str, float]] = {}
    for metric, (weight, ascending) in metrics.items():
        values = pd.to_numeric(comparison[metric], errors="coerce")
        if values.notna().sum() <= 1:
            continue
        percentile = 1.0 - values.rank(method="average", ascending=ascending, pct=True) + (1.0 / values.notna().sum())
        scores += percentile.fillna(0.0) * weight
        weights += weight
        rank_breakdown[metric] = {str(comparison.loc[idx, "model"]): float(value) for idx, value in percentile.items()}
    if not weights:
        raise ValueError("Unable to select model because comparison metrics are unavailable")
    comparison = comparison.copy()
    comparison["selection_score"] = scores / weights
    best_idx = comparison["selection_score"].idxmax()
    best_name = str(comparison.loc[best_idx, "model"])
    reason = {
        "selected_model": best_name,
        "selection_method": "weighted mean of percentile ranks (higher is better)",
        "weights": {key: val[0] for key, val in metrics.items()},
        "rank_breakdown": rank_breakdown,
        "model_scores": {str(comparison.loc[idx, "model"]): float(value) for idx, value in comparison["selection_score"].items()},
        "selected_metrics": comparison.loc[best_idx].to_dict(),
        "reasoning": (
            "The selected model has the strongest weighted rank across observation-level Top-1 accuracy, "
            "candidate-level Macro F1, route continuity, candidate-level Log Loss, fold stability, and CV inference time."
        ),
    }
    return best_name, json_safe(reason)


def suggest_hyperparameters(trial: Any, model_name: str) -> dict[str, Any]:
    """Model-appropriate Optuna search spaces for Bayesian optimization."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 9, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 100),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 16, 40),
            "max_depth": trial.suggest_categorical("max_depth", [8, 10, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
        }
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "max_iter": 1500,
        }
    if model_name == "shallow_mlp":
        return {
            "hidden_layer_sizes": (trial.suggest_categorical("hidden_units", [16, 32, 64]),),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "max_iter": 35,
        }
    return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}


def optimize_selected_model(
    model_name: str,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    raw_context: Mapping[str, Mapping[str, Any]],
    n_trials: int = 20,
    seed: int = DEFAULT_SEED,
    progress: bool = True,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Run Optuna TPE against mean fold candidate-level Log Loss."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    startup_trials = max(2, min(10, int(n_trials) // 3))
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials),
    )

    def objective(trial: Any) -> float:
        params = suggest_hyperparameters(trial, model_name)
        result = cross_validate_model(
            model_name, frame, feature_columns, folds, raw_context,
            params=params, seed=seed, store_oof=False, progress=False,
        )
        trial.set_user_attr("top1_accuracy", result.get("mean_top1_accuracy"))
        trial.set_user_attr("macro_f1", result.get("mean_macro_f1"))
        trial.set_user_attr("fold_log_losses", [r.get("log_loss") for r in result["fold_metrics"]])
        trial.set_user_attr("cv_training_time_s", result.get("cv_training_time_s"))
        return float(result["mean_log_loss"])

    study.optimize(objective, n_trials=int(n_trials), n_jobs=1, show_progress_bar=progress)
    rows = []
    for trial in study.trials:
        rows.append({
            "trial": int(trial.number),
            "state": trial.state.name,
            "objective_log_loss": trial.value,
            "top1_accuracy": trial.user_attrs.get("top1_accuracy"),
            "macro_f1": trial.user_attrs.get("macro_f1"),
            "cv_training_time_s": trial.user_attrs.get("cv_training_time_s"),
            "params": json.dumps(trial.params, sort_keys=True),
            **{f"param_{k}": v for k, v in trial.params.items()},
        })
    best_params = dict(study.best_params)
    if model_name == "shallow_mlp":
        best_params["hidden_layer_sizes"] = (best_params.pop("hidden_units"),)
    best = {
        "model": model_name,
        "best_params": best_params,
        "best_cv_log_loss": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "n_trials": int(len(study.trials)),
        "sampler": "Optuna TPESampler",
        "tpe_startup_trials": startup_trials,
        "seed": int(seed),
        "objective": "mean five-fold GroupKFold candidate-level Log Loss on eligible observations",
        "best_trial_top1_accuracy": study.best_trial.user_attrs.get("top1_accuracy"),
    }
    return study, json_safe(best), pd.DataFrame(rows)


def optimization_figure(study: Any, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    values = [t.value for t in study.trials if t.value is not None]
    trial_ids = [t.number for t in study.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(values) if values else []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trial_ids, values, marker="o", alpha=0.55, label="Trial CV Log Loss")
    if len(best_so_far):
        ax.plot(trial_ids, best_so_far, linewidth=2, label="Best so far")
    ax.set(title="Bayesian optimization history", xlabel="Trial", ylabel="Five-fold candidate-level Log Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def comparison_plots(comparison: pd.DataFrame, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("mean_macro_f1", "std_macro_f1", "Macro F1", "macro_f1_comparison.png", False),
        ("mean_candidate_positive_precision", "std_candidate_positive_precision", "Candidate Positive Precision", "candidate_precision_comparison.png", False),
        ("mean_candidate_binary_accuracy", "std_candidate_binary_accuracy", "Candidate Binary Accuracy", "candidate_accuracy_comparison.png", False),
        ("mean_top1_accuracy", "std_top1_accuracy", "Top-1 Accuracy", "top1_accuracy_comparison.png", False),
        ("mean_log_loss", "std_log_loss", "Candidate Log Loss", "log_loss_comparison.png", True),
        ("mean_route_continuity", "std_route_continuity", "Route Continuity", "route_continuity_comparison.png", False),
        ("cv_training_time_s", None, "Five-fold Training Time (seconds)", "training_time_comparison.png", True),
        ("cv_inference_time_s", None, "Five-fold Inference Time (seconds)", "inference_time_comparison.png", True),
    ]
    for metric, error, label, filename, _ in plot_specs:
        values = pd.to_numeric(comparison[metric], errors="coerce").to_numpy()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        errors = pd.to_numeric(comparison[error], errors="coerce").fillna(0).to_numpy() if error else None
        ax.bar(comparison["model"], values, yerr=errors, capsize=4 if error else 0, color="#3274A1")
        ax.set_ylabel(label)
        ax.set_title(f"Model comparison: {label}")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def save_probability_outputs(
    candidate_probabilities: pd.DataFrame,
    records: list[dict[str, Any]],
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_frame = candidate_probabilities[[TRIP_COLUMN, OBSERVATION_COLUMN, ARC_COLUMN, "probability"]].copy()
    output_frame.rename(columns={OBSERVATION_COLUMN: "observation_id"}, inplace=True)
    parquet_path = output_dir / f"{stem}.parquet"
    json_path = output_dir / f"{stem}.json"
    output_frame.to_parquet(parquet_path, index=False)
    write_json(json_path, records)
    return parquet_path, json_path


def save_model_artifacts(
    model: Any,
    feature_columns: Sequence[str],
    model_name: str,
    params: Mapping[str, Any],
    metadata: Mapping[str, Any],
    artifact_dir: str | Path = "artifacts",
) -> dict[str, str]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "best_model.pkl"
    preprocessing_path = artifact_dir / "preprocessing_pipeline.pkl"
    joblib.dump(model, model_path, compress=3)
    joblib.dump(model.named_steps["preprocess"], preprocessing_path, compress=3)
    write_json(artifact_dir / "feature_schema.json", {
        "feature_columns": list(feature_columns),
        "identifier_columns": IDENTIFIER_COLUMNS,
        "target_column": TARGET_COLUMN,
        "evaluation_only_columns": EVALUATION_ONLY_COLUMNS,
        "observation_id_output_name": "observation_id",
    })
    write_json(artifact_dir / "best_hyperparameters.json", {
        "model": model_name,
        "hyperparameters": dict(params),
    })
    write_json(artifact_dir / "model_metadata.json", dict(metadata))
    return {
        "best_model": str(model_path),
        "preprocessing_pipeline": str(preprocessing_path),
        "feature_schema": str(artifact_dir / "feature_schema.json"),
        "model_metadata": str(artifact_dir / "model_metadata.json"),
        "best_hyperparameters": str(artifact_dir / "best_hyperparameters.json"),
    }


def load_model(path: str | Path) -> Any:
    """Load a saved sklearn/lightgbm model pipeline."""
    return joblib.load(path)


def per_trip_metric_records(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    raw_context: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trip_id in sorted(frame[TRIP_COLUMN].astype(str).unique().tolist()):
        mask = frame[TRIP_COLUMN].astype(str).eq(trip_id).to_numpy()
        rows.append({"trip_id": trip_id, **evaluate_predictions(frame.loc[mask], np.asarray(probabilities)[mask], raw_context, [trip_id])})
    return rows
