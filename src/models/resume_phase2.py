"""Resume Phase 2 after the baseline five-model comparison is complete."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data.loader import discover_trips
from src.models.phase2 import (
    DEFAULT_SEED,
    TARGET_COLUMN,
    TRIP_COLUMN,
    cross_validate_model,
    evaluate_predictions,
    filter_eligible_observations,
    get_feature_columns,
    json_safe,
    load_candidate_data,
    load_raw_context,
    make_grouped_folds,
    make_model_pipeline,
    optimize_selected_model,
    optimization_figure,
    predict_candidate_probabilities,
    save_model_artifacts,
    save_probability_outputs,
    write_json,
)
from src.models.run_phase2 import _implementation_markdown


REPO_ROOT = Path(__file__).resolve().parents[2]


def _baseline_cv_rows(model_results: dict, train_trip_ids: list[str], eligible_row_count: int, all_row_count: int) -> list[dict]:
    rows = []
    for name, result in model_results.items():
        rows.extend({**row, "configuration": "baseline"} for row in result["fold_metrics"])
        rows.extend({**row, "configuration": "baseline"} for row in result["per_trip_metrics"])
        rows.append({
            "model": name,
            "row_type": "pooled_oof",
            "fold": None,
            "trip_id": None,
            "training_trip_ids": None,
            "validation_trip_ids": json.dumps(train_trip_ids),
            "training_row_count": int(eligible_row_count),
            "validation_row_count": int(all_row_count),
            "training_time_s": result["cv_training_time_s"],
            "inference_time_s": result["cv_inference_time_s"],
            **result["pooled_metrics"],
            "configuration": "baseline",
        })
    return rows


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("PATHSNAP_N_JOBS", str(args.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))

    results_root = Path(args.results_dir)
    artifact_root = Path(args.artifacts_dir)
    plots_dir = results_root / "plots"
    predictions_dir = results_root / "prediction_probabilities"
    baseline_path = results_root / "model_comparison.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline results are missing: {baseline_path}. Run src.models.run_phase2 first."
        )
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not {"selection", "models", "comparison"}.issubset(baseline_payload):
        raise ValueError("Existing model_comparison.json is not a complete baseline comparison")

    train = load_candidate_data(args.train_parquet)
    student_test = load_candidate_data(args.student_test_parquet)
    features = get_feature_columns(train)
    train_ids = discover_trips(str(Path(args.raw_root) / "train"))
    test_ids = discover_trips(str(Path(args.raw_root) / "student_test"))
    if train_ids != sorted(train[TRIP_COLUMN].astype(str).unique().tolist()):
        raise ValueError("Raw training trips and training Parquet trip IDs do not match")
    if test_ids != sorted(student_test[TRIP_COLUMN].astype(str).unique().tolist()):
        raise ValueError("Raw student-test trips and student-test Parquet trip IDs do not match")

    print("Loading raw trip context for optimization and student-test metrics…", flush=True)
    train_context = load_raw_context(args.raw_root, "train", train_ids)
    test_context = load_raw_context(args.raw_root, "student_test", test_ids)
    folds = make_grouped_folds(train, train_ids, n_splits=5)
    eligible_train = filter_eligible_observations(train)
    schema = json.loads((results_root / "dataset_schema.json").read_text(encoding="utf-8"))
    comparison = pd.DataFrame(baseline_payload["comparison"])
    selected_name = baseline_payload["selection"]["selected_model"]
    selection = baseline_payload["selection"]
    model_results = baseline_payload["models"]

    print(f"\n=== Bayesian optimization: {selected_name} ({args.trials} trials) ===", flush=True)
    study, optimization, trials = optimize_selected_model(
        selected_name, train, features, folds, train_context,
        n_trials=args.trials, seed=args.seed, progress=True,
    )
    trials.to_csv(results_root / "optimization_trials.csv", index=False)
    write_json(results_root / "best_hyperparameters.json", optimization)
    optimization_figure(study, plots_dir / "optimization_history.png")

    print("\n=== Tuned cross-validation on the common folds ===", flush=True)
    tuned_cv = cross_validate_model(
        selected_name, train, features, folds, train_context,
        params=optimization["best_params"], seed=args.seed, store_oof=True, progress=True,
    )
    cv_rows = _baseline_cv_rows(
        model_results, train_ids, len(eligible_train), len(train)
    )
    for row in tuned_cv["fold_metrics"]:
        cv_rows.append({**row, "configuration": "tuned"})
    for row in tuned_cv["per_trip_metrics"]:
        cv_rows.append({**row, "configuration": "tuned"})
    cv_rows.append({
        "model": selected_name,
        "row_type": "pooled_oof",
        "fold": None,
        "trip_id": None,
        "training_trip_ids": None,
        "validation_trip_ids": json.dumps(train_ids),
        "training_row_count": int(len(eligible_train)),
        "validation_row_count": int(len(train)),
        "training_time_s": tuned_cv["cv_training_time_s"],
        "inference_time_s": tuned_cv["cv_inference_time_s"],
        **tuned_cv["pooled_metrics"],
        "configuration": "tuned",
    })
    pd.DataFrame(cv_rows).to_csv(results_root / "cv_results.csv", index=False)

    baseline_result = model_results[selected_name]
    tuned_vs_untuned = pd.DataFrame([
        {
            "configuration": "untuned_baseline",
            "macro_f1_mean": baseline_result["mean_macro_f1"],
            "macro_f1_std": baseline_result["std_macro_f1"],
            "top1_accuracy_mean": baseline_result["mean_top1_accuracy"],
            "log_loss_mean": baseline_result["mean_log_loss"],
            "route_continuity_pooled": baseline_result["pooled_metrics"]["route_continuity"],
            "training_time_s": baseline_result["cv_training_time_s"],
            "inference_time_s": baseline_result["cv_inference_time_s"],
            "params": json.dumps({}, sort_keys=True),
        },
        {
            "configuration": "tuned",
            "macro_f1_mean": tuned_cv["mean_macro_f1"],
            "macro_f1_std": tuned_cv["std_macro_f1"],
            "top1_accuracy_mean": tuned_cv["mean_top1_accuracy"],
            "log_loss_mean": tuned_cv["mean_log_loss"],
            "route_continuity_pooled": tuned_cv["pooled_metrics"]["route_continuity"],
            "training_time_s": tuned_cv["cv_training_time_s"],
            "inference_time_s": tuned_cv["cv_inference_time_s"],
            "params": json.dumps(optimization["best_params"], sort_keys=True),
        },
    ])
    tuned_vs_untuned.to_csv(results_root / "tuned_vs_untuned.csv", index=False)

    print("\n=== Final model fit and student-test evaluation ===", flush=True)
    final_model = make_model_pipeline(selected_name, optimization["best_params"], args.seed)
    x_train = eligible_train[features].replace([np.inf, -np.inf], np.nan)
    y_train = eligible_train[TARGET_COLUMN].astype(np.int8).to_numpy()
    start = time.perf_counter()
    final_model.fit(x_train, y_train)
    final_training_seconds = time.perf_counter() - start

    observations = pd.DataFrame([
        {TRIP_COLUMN: trip_id, "obs_index": obs_id}
        for trip_id, context in test_context.items()
        for obs_id in range(int(context["observation_count"]))
    ])
    start = time.perf_counter()
    test_predictions, hmm_records = predict_candidate_probabilities(
        final_model, student_test, features, observations=observations
    )
    test_inference_seconds = time.perf_counter() - start
    sums = test_predictions.groupby([TRIP_COLUMN, "obs_index"])["probability"].sum()
    if len(sums) and not np.allclose(sums.to_numpy(), 1.0, atol=1e-6):
        raise AssertionError("HMM candidate probabilities do not sum to one per nonempty observation")
    if len(test_predictions) and (
        not np.isfinite(test_predictions["probability"].to_numpy()).all()
        or (test_predictions["probability"] < 0).any()
    ):
        raise AssertionError("HMM output contains invalid probability values")
    student_metrics = evaluate_predictions(
        student_test, test_predictions["raw_probability"].to_numpy(), test_context, test_ids
    )
    output_parquet, output_json = save_probability_outputs(
        test_predictions, hmm_records, predictions_dir,
        "student_test_candidate_probabilities",
    )

    metadata = {
        "selected_model": selected_name,
        "feature_columns": features,
        "target_column": TARGET_COLUMN,
        "identifier_columns": ["trip_id", "obs_index", "arc_id"],
        "random_seed": int(args.seed),
        "library_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "optuna": __import__("optuna").__version__, "joblib": joblib.__version__,
        },
        "training_partition": {
            "source": str(args.train_parquet), "trip_count": len(train_ids),
            "trip_ids": train_ids, "eligible_observation_count": schema["training"]["eligible_observation_count"],
            "candidate_rows_after_filtering": len(eligible_train),
        },
        "cross_validation": {
            "method": "GroupKFold(n_splits=5)", "group_column": TRIP_COLUMN,
            "baseline_selected_model": baseline_result["pooled_metrics"],
            "tuned_selected_model": tuned_cv["pooled_metrics"],
            "fold_metrics": tuned_cv["fold_metrics"],
            "mean_metrics": {k: v for k, v in tuned_cv.items() if k.startswith(("mean_", "std_"))},
        },
        "student_test": {
            "source": str(args.student_test_parquet), "trip_count": len(test_ids),
            "trip_ids": test_ids, "metrics": student_metrics,
            "final_model_training_time_s": final_training_seconds,
            "inference_time_s": test_inference_seconds,
            "prediction_parquet": str(output_parquet), "prediction_json": str(output_json),
        },
        "hyperparameters": optimization["best_params"],
        "bayesian_optimization": optimization,
        "preprocessing": "median imputation fitted within each training fold; StandardScaler for LogisticRegression and MLP",
        "calibration_method": "none; probability quality checked with held-out Brier score, ECE, and Log Loss; HMM scores normalized within candidate sets",
        "candidate_probability_normalization": "clip invalid or negative scores to zero, then divide by the within-observation sum; uniform fallback when all scores are unusable",
        "final_training_time_s": final_training_seconds,
        "student_test_inference_time_s": test_inference_seconds,
    }
    artifact_paths = save_model_artifacts(
        final_model, features, selected_name, optimization["best_params"], metadata, artifact_root
    )

    result_payload = {
        "selection": selection,
        "comparison": comparison.to_dict("records"),
        "optimization": optimization,
        "tuned_cv_metrics": tuned_cv["pooled_metrics"],
        "student_test_metrics": student_metrics,
        "artifact_paths": artifact_paths,
        "training_schema": schema["training"],
        "student_test_schema": schema["student_test"],
        "baseline_models": model_results,
    }
    write_json(results_root / "model_comparison.json", result_payload)
    (REPO_ROOT / "docs" / "PHASE2_ML_IMPLEMENTATION.md").write_text(
        _implementation_markdown(
            schema, comparison, selection, optimization, tuned_vs_untuned,
            student_metrics, artifact_paths,
            {"final_training_time_s": final_training_seconds, "student_test_inference_time_s": test_inference_seconds},
        ),
        encoding="utf-8",
    )
    print(f"Selected {selected_name}; best CV Log Loss={optimization['best_cv_log_loss']:.6f}", flush=True)
    print(f"Student-test metrics: {json.dumps(json_safe(student_metrics), sort_keys=True)}", flush=True)
    print(f"Prediction files: {output_parquet}, {output_json}", flush=True)
    print(f"Artifacts: {json.dumps(artifact_paths, sort_keys=True)}", flush=True)
    return result_payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", default=str(REPO_ROOT / "data/processed/train_candidates.parquet"))
    parser.add_argument("--student-test-parquet", default=str(REPO_ROOT / "data/processed/student_test_candidates.parquet"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "data/raw/Student folder"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
