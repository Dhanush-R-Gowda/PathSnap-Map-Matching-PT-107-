"""Run the complete Phase 2 training, comparison, tuning, and test workflow."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data.loader import discover_trips
from src.models.phase2 import (
    APPROVED_FEATURE_COLUMNS,
    DEFAULT_SEED,
    MODEL_NAMES,
    OBSERVATION_COLUMN,
    TARGET_COLUMN,
    TRIP_COLUMN,
    candidate_miss_metrics,
    comparison_plots,
    cross_validate_model,
    dataset_summary,
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
    select_model,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "(no rows)"
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("—")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def _schema_markdown(schema: dict) -> str:
    train = schema["training"]
    test = schema["student_test"]
    rows = [
        "# Phase 2 Dataset Schema",
        "",
        "Schema inspected from the tracked Phase 1 Parquet outputs. Identifier and evaluation columns are retained for grouping, evaluation, and HMM export; they are not model features.",
        "",
        "## Input dataset summary",
        "",
        "| Partition | Trips | Candidate rows | Raw GPS observations | Eligible observations | Excluded observations | Candidate misses | Miss rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Training | {train['trip_count']} | {train['rows']:,} | {train['raw_observation_count']:,} | {train['eligible_observation_count']:,} | {train['excluded_observation_count']:,} | {train['candidate_miss_count']:,} | {train['candidate_miss_rate']:.3%} |",
        f"| Student test | {test['trip_count']} | {test['rows']:,} | {test['raw_observation_count']:,} | {test['eligible_observation_count']:,} | {test['excluded_observation_count']:,} | {test['candidate_miss_count']:,} | {test['candidate_miss_rate']:.3%} |",
        "",
        "Training candidate rows after observation-level filtering: **{:,}**. The filter keeps every candidate for an eligible observation. The candidate-miss denominator uses raw track observations, so observations with no generated candidate row are counted as misses.".format(train["candidate_rows_after_filtering"]),
        "",
        "## Field roles",
        "",
        "| Role | Columns |",
        "|---|---|",
        "| Identifiers (not features) | `trip_id`, `obs_index`, `arc_id` |",
        "| Target | `label` (`1` is the aligned ground-truth arc; `0` otherwise) |",
        "| Evaluation-only | `is_gt_in_candidates` |",
        "| Approved numeric features | " + ", ".join(f"`{c}`" for c in train["feature_columns"]) + " |",
        "",
        "Candidate geometry is not stored as a geometry column in this Parquet schema. Candidate-to-road connectivity is supplied by each trip's raw `.arcs` file (directed from-node/to-node pairs), with `arc_id` indexing that trip-local arc list.",
        "",
        "## Exact Parquet columns and dtypes",
        "",
        "| Column | Type | Role | Missing training values |",
        "|---|---|---|---:|",
    ]
    for col, dtype in train["column_types"].items():
        if col in train["feature_columns"]:
            role = "feature"
        elif col in ("trip_id", "obs_index", "arc_id"):
            role = "identifier"
        elif col == "label":
            role = "target"
        else:
            role = "evaluation-only"
        rows.append(f"| `{col}` | `{dtype}` | {role} | {train['missing_values_all_columns'][col]} |")
    rows.extend([
        "",
        "## Data quality checks",
        "",
        f"- Training full-row duplicates: {train['duplicate_full_rows']:,}; duplicate candidate keys: {train['duplicate_candidate_keys']:,}.",
        f"- Training non-finite approved feature values: {sum(train['feature_invalid_numeric_values'].values()):,}.",
        f"- Training candidate rows per represented observation: min {train['candidate_rows_per_observation']['min']}, median {train['candidate_rows_per_observation']['median']:.1f}, mean {train['candidate_rows_per_observation']['mean']:.2f}, max {train['candidate_rows_per_observation']['max']}.",
        f"- Candidate miss rows with a nonempty candidate set: {train['ineligible_observations_with_candidates']:,} observations.",
        f"- Raw observations with no candidate rows: {train['observations_without_candidate_rows']:,}.",
        "- Feature matrix uses only the approved Phase 1 numeric columns listed above. Preprocessing is fitted within each training fold.",
        "",
    ])
    return "\n".join(rows)


def _implementation_markdown(
    schema: dict,
    comparison: pd.DataFrame,
    selected: dict,
    optimization: dict,
    tuned_vs_untuned: pd.DataFrame,
    student_metrics: dict,
    artifact_paths: dict,
    test_result: dict,
) -> str:
    metric_cols = [
        "model", "mean_macro_f1", "mean_top1_accuracy", "mean_log_loss",
        "candidate_miss_rate", "route_continuity", "cv_training_time_s", "cv_inference_time_s",
    ]
    table = comparison.rename(columns={
        "mean_macro_f1": "CV Macro F1",
        "mean_candidate_positive_precision": "Candidate Positive Precision",
        "mean_candidate_binary_accuracy": "Candidate Binary Accuracy",
        "mean_top1_accuracy": "Top-1 Accuracy",
        "mean_log_loss": "Log Loss",
        "candidate_miss_rate": "Candidate Miss Rate",
        "route_continuity": "Route Continuity",
        "cv_training_time_s": "CV Training Time (s)",
        "cv_inference_time_s": "CV Inference Time (s)",
    })
    display_cols = [
        "model", "CV Macro F1", "Candidate Positive Precision", "Candidate Binary Accuracy", "Top-1 Accuracy",
        "Log Loss", "Candidate Miss Rate", "Route Continuity", "CV Training Time (s)", "CV Inference Time (s)",
    ]
    ranking_cols = [
        "model", "pooled_macro_f1", "pooled_candidate_positive_precision", "pooled_candidate_binary_accuracy",
        "pooled_top1_accuracy", "macro_f1_rank", "candidate_precision_rank", "candidate_accuracy_rank", "top1_accuracy_rank",
    ]
    ranking_table = comparison[ranking_cols].rename(columns={
        "pooled_macro_f1": "Pooled OOF Macro F1",
        "pooled_candidate_positive_precision": "Pooled OOF Positive Precision",
        "pooled_candidate_binary_accuracy": "Pooled OOF Candidate Accuracy",
        "pooled_top1_accuracy": "Pooled OOF Top-1 Accuracy",
        "macro_f1_rank": "Macro F1 Rank",
        "candidate_precision_rank": "Precision Rank",
        "candidate_accuracy_rank": "Candidate Accuracy Rank",
        "top1_accuracy_rank": "Top-1 Accuracy Rank",
    })
    tuned_table = tuned_vs_untuned.copy()
    tuned_cols = [c for c in [
        "configuration", "macro_f1_mean", "candidate_positive_precision_mean", "candidate_binary_accuracy_mean",
        "top1_accuracy_mean", "log_loss_mean", "route_continuity_pooled", "training_time_s", "inference_time_s",
    ] if c in tuned_table]
    train = schema["training"]
    test = schema["student_test"]
    features = ", ".join(f"`{c}`" for c in train["feature_columns"])
    lines = [
        "# Phase 2 ML Implementation",
        "",
        "## ML problem and inputs",
        "",
        "The system is a candidate-level binary classifier. Each row is one GPS observation paired with one candidate road arc. `label=1` means the candidate arc equals the Phase 1 aligned reference arc; `label=0` means it does not. The model estimates binary candidate probabilities. HMM export normalizes those scores over each observation's candidate set.",
        "",
        f"Training data: {train['rows']:,} candidate rows, {train['trip_count']} trips, {train['raw_observation_count']:,} raw GPS observations. Student-test data: {test['rows']:,} candidate rows, {test['trip_count']} trips, {test['raw_observation_count']:,} observations.",
        "",
        "## Schema and feature selection",
        "",
        "Exact field types and data-quality counts are in [PHASE2_DATASET_SCHEMA.md](PHASE2_DATASET_SCHEMA.md) and `results/dataset_schema.json`. Identifier columns `trip_id`, `obs_index`, and `arc_id`, target `label`, and evaluation flag `is_gt_in_candidates` are never predictive features.",
        "",
        f"Approved Phase 1 feature columns: {features}.",
        "",
        "All 14 approved features are numeric. Infinite values are converted to missing values; median imputation is fitted separately inside each fold. Logistic Regression and the shallow MLP also use a fold-fitted StandardScaler. No learned preprocessing is fitted on validation or student-test rows.",
        "",
        "## Observation filtering",
        "",
        f"Initial classifier training retains observations where `is_gt_in_candidates == True`: {train['eligible_observation_count']:,} eligible observations and {train['candidate_rows_after_filtering']:,} candidate rows. It excludes {train['excluded_observation_count']:,} observations and {train['rows'] - train['candidate_rows_after_filtering']:,} candidate rows. Filtering is at the observation level and preserves all candidate rows in each eligible observation.",
        "",
        f"Candidate miss rate uses all {train['raw_observation_count']:,} raw training observations. There are {train['candidate_miss_count']:,} misses ({train['candidate_miss_rate']:.3%}), including {train['observations_without_candidate_rows']:,} observations with an empty candidate set. This is kept separate from classifier accuracy, which is evaluated only where the true arc exists in the candidate set.",
        "",
        "## Models and grouped validation",
        "",
        "The five implemented models use the same approved features, target, eligible training observations, random seed, and five folds: LightGBM Classifier, Random Forest Classifier, Logistic Regression, shallow MLP Classifier, and Gaussian Naive Bayes.",
        "",
        "The shallow MLP reached its configured 35-iteration limit on each baseline fold and emitted scikit-learn ConvergenceWarnings; the warning was retained and the measured last-iteration predictions were used.",
        "",
        "`GroupKFold(n_splits=5)` assigns trips to folds before candidate rows are filtered. Every row from a held-out trip stays in validation; only eligible rows from training trips are fit. Automated assertions verify that training and validation trip ID sets do not overlap. The exact IDs and row counts are recorded in `results/cv_results.csv`.",
        "",
        "Candidate Macro F1 is binary macro-averaged F1 over eligible candidate rows at a 0.5 probability threshold. Candidate Positive Precision measures the share of predicted positive candidates that are correct at that threshold; Candidate Binary Accuracy measures all eligible candidate rows at the same threshold. Top-1 Accuracy separately selects the highest-scoring candidate for each eligible observation and checks whether it is the aligned arc. Fold means and standard deviations are reported; `cv_results.csv` also has per-trip and pooled out-of-fold rows. `model_comparison.csv` includes 1-based ranks for Macro F1, positive precision, binary candidate accuracy, and top-1 accuracy; higher values rank first.",
        "",
        "Candidate Miss Rate is the fraction of all raw observations with no ground-truth candidate. It includes observations with no candidate rows. Route Continuity is connected top-1 directed arc pairs divided by directly consecutive observation pairs where both observations have candidates; a pair connects when the first arc's `to_node` equals the next arc's `from_node` in that trip's graph. Arc IDs remain scoped by trip.",
        "",
        "Training and inference times in the comparison are cumulative seconds over the five fold fits and validations. Per-fold timings are also in `results/cv_results.csv`.",
        "",
        "## Model comparison and selection",
        "",
        _format_table(table, display_cols),
        "",
        "### Metric rankings",
        "",
        "Ranks are based on pooled out-of-fold (OOF) predictions from the same five trip-grouped folds; higher metrics rank first. Positive precision and candidate binary accuracy use a 0.5 probability threshold. Top-1 accuracy selects the highest-scoring candidate per eligible observation.",
        "",
        _format_table(ranking_table, list(ranking_table.columns)),
        "",
        f"Selected baseline: **{selected['selected_model']}**. Selection uses a weighted mean of percentile ranks: Top-1 Accuracy (0.23), candidate Macro F1 (0.23), route continuity (0.18), candidate Log Loss (0.14), Macro F1 fold stability (0.10), five-fold training time (0.07), and inference time (0.05). Lower Log Loss, variability, and both time measures rank higher. The selection details and per-metric ranks are saved in `results/model_comparison.json`.",
        "",
        "## Bayesian optimization",
        "",
        f"Optuna TPESampler ran {optimization['n_trials']} trials on the selected model with seed {optimization['seed']} ({optimization['tpe_startup_trials']} startup trials followed by {optimization['n_trials'] - optimization['tpe_startup_trials']} adaptive suggestions). The objective is mean five-fold GroupKFold candidate-level Log Loss on eligible validation observations; each trial also records Top-1 Accuracy. Best CV Log Loss: {optimization['best_cv_log_loss']:.6f}; best trial: {optimization['best_trial']}; Top-1 on best trial: {optimization.get('best_trial_top1_accuracy')!s}.",
        "",
        "The complete trial history is in `results/optimization_trials.csv`; best parameters are in `results/best_hyperparameters.json`. `results/tuned_vs_untuned.csv` compares baseline and tuned CV performance on the same folds.",
        "",
        _format_table(tuned_table, tuned_cols) if tuned_cols else "(no tuned comparison table)",
        "",
        "## Probability quality and HMM interface",
        "",
        "Probability quality is checked using held-out candidate Log Loss, Brier Score, and 10-bin expected calibration error. No post-hoc calibration model was fit. The HMM output normalization is a within-observation score normalization, not a calibration fit; it is applied independently to each trip/observation candidate group. Invalid or negative probabilities are set to zero; if all candidate scores are unusable, the group receives a uniform distribution. Empty candidate sets are represented by `candidates: []` in JSON.",
        "",
        "Student-test output files are `results/prediction_probabilities/student_test_candidate_probabilities.parquet` and `.json`. Each Parquet row has `trip_id`, `observation_id`, `arc_id`, and `probability`. JSON contains one observation record in chronological order, with candidate probabilities that sum to one within nonempty sets. Trip ID is always part of the grouping key, so trip-specific arc IDs are not merged.",
        "",
        "```python",
        "import joblib, pandas as pd",
        "model = joblib.load('artifacts/best_model.pkl')",
        "probabilities = pd.read_parquet('results/prediction_probabilities/student_test_candidate_probabilities.parquet')",
        "# Pass probability as the HMM emission score for its (trip_id, observation_id, arc_id).",
        "```",
        "",
        "## Student-test evaluation",
        "",
        "The selected tuned estimator was fitted only on eligible training observations from the 80 training trips. Student-test labels were used once for the requested final evaluation and not for model selection or optimization.",
        "",
        f"Metrics: Top-1 Accuracy {student_metrics.get('top1_accuracy')!s}; candidate Macro F1 {student_metrics.get('macro_f1')!s}; candidate positive precision {student_metrics.get('candidate_positive_precision')!s}; candidate binary accuracy {student_metrics.get('candidate_binary_accuracy')!s}; candidate Log Loss {student_metrics.get('log_loss')!s}; candidate miss rate {student_metrics.get('candidate_miss_rate')!s}; route continuity {student_metrics.get('route_continuity')!s}; Brier Score {student_metrics.get('brier_score')!s}; expected calibration error {student_metrics.get('expected_calibration_error')!s}; final-model training time {test_result['final_training_time_s']:.3f}s; student-test inference time {test_result['student_test_inference_time_s']:.3f}s.",
        "",
        "## Saved artifacts",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in artifact_paths.items())
    lines.extend([
        "",
        "## Limitations",
        "",
        "The classifier ranks individual candidates and does not enforce route continuity itself; the HMM stage should apply transition constraints. Candidate misses cannot be fixed by the classifier and remain reported separately. The supplied student-test set has only 10 trips, so its metrics are descriptive and do not replace trip-grouped cross-validation. Probability normalization creates a per-observation distribution but does not guarantee calibration under distribution shift.",
        "",
        "## Reproduction",
        "",
        "From the repository root, using Windows PowerShell:",
        "",
        "```powershell",
        "py -3.12 -m venv .venv",
        ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
        "Expand-Archive -LiteralPath \"$env:USERPROFILE\\Downloads\\Student folder.zip\" -DestinationPath \"data/raw\"",
        ".\\.venv\\Scripts\\python.exe -m src.models.run_phase2 --trials 5 --seed 42",
        ".\\.venv\\Scripts\\python.exe -m pytest tests/ -v",
        "```",
        "",
    ])
    return "\n".join(lines)


def _comparison_row(result: dict) -> dict:
    pooled = result["pooled_metrics"]
    return {
        "model": result["model"],
        "mean_macro_f1": result.get("mean_macro_f1"),
        "std_macro_f1": result.get("std_macro_f1"),
        "pooled_macro_f1": pooled.get("macro_f1"),
        "mean_candidate_positive_precision": result.get("mean_candidate_positive_precision"),
        "std_candidate_positive_precision": result.get("std_candidate_positive_precision"),
        "pooled_candidate_positive_precision": pooled.get("candidate_positive_precision"),
        "mean_candidate_binary_accuracy": result.get("mean_candidate_binary_accuracy"),
        "std_candidate_binary_accuracy": result.get("std_candidate_binary_accuracy"),
        "pooled_candidate_binary_accuracy": pooled.get("candidate_binary_accuracy"),
        "mean_top1_accuracy": result.get("mean_top1_accuracy"),
        "std_top1_accuracy": result.get("std_top1_accuracy"),
        "pooled_top1_accuracy": pooled.get("top1_accuracy"),
        "mean_log_loss": result.get("mean_log_loss"),
        "std_log_loss": result.get("std_log_loss"),
        "pooled_log_loss": pooled.get("log_loss"),
        "brier_score": pooled.get("brier_score"),
        "expected_calibration_error": pooled.get("expected_calibration_error"),
        "candidate_miss_rate": pooled.get("candidate_miss_rate"),
        "route_continuity": pooled.get("route_continuity"),
        "mean_route_continuity": result.get("mean_route_continuity"),
        "std_route_continuity": result.get("std_route_continuity"),
        "cv_training_time_s": result["cv_training_time_s"],
        "cv_inference_time_s": result["cv_inference_time_s"],
        "normalized_observation_log_loss": pooled.get("normalized_observation_log_loss"),
    }


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("PATHSNAP_N_JOBS", str(args.threads))
    warnings.filterwarnings("default", category=UserWarning)

    train_path = Path(args.train_parquet)
    test_path = Path(args.student_test_parquet)
    raw_root = Path(args.raw_root)
    results_root = Path(args.results_dir)
    artifacts_root = Path(args.artifacts_dir)
    plots_dir = results_root / "plots"
    predictions_dir = results_root / "prediction_probabilities"

    train = load_candidate_data(train_path)
    student_test = load_candidate_data(test_path)
    features = get_feature_columns(train)
    if features != APPROVED_FEATURE_COLUMNS:
        raise AssertionError("Feature order differs from the approved Phase 1 schema")
    if get_feature_columns(student_test) != features:
        raise ValueError("Training and student-test feature schemas do not match")

    raw_train_ids = discover_trips(str(raw_root / "train"))
    raw_test_ids = discover_trips(str(raw_root / "student_test"))
    train_ids = sorted(train[TRIP_COLUMN].astype(str).unique().tolist())
    test_ids = sorted(student_test[TRIP_COLUMN].astype(str).unique().tolist())
    if raw_train_ids != train_ids:
        raise ValueError(f"Raw training trip IDs do not match Parquet. Raw-only={sorted(set(raw_train_ids)-set(train_ids))}, Parquet-only={sorted(set(train_ids)-set(raw_train_ids))}")
    if raw_test_ids != test_ids:
        raise ValueError(f"Raw student-test trip IDs do not match Parquet. Raw-only={sorted(set(raw_test_ids)-set(test_ids))}, Parquet-only={sorted(set(test_ids)-set(raw_test_ids))}")

    print("Loading raw observation counts and trip-specific arc connectivity…", flush=True)
    train_context = load_raw_context(raw_root, "train", raw_train_ids)
    test_context = load_raw_context(raw_root, "student_test", raw_test_ids)
    train_schema = dataset_summary(train, train_path, {k: v["observation_count"] for k, v in train_context.items()})
    test_schema = dataset_summary(student_test, test_path, {k: v["observation_count"] for k, v in test_context.items()})
    schema = {
        "training": train_schema,
        "student_test": test_schema,
        "identifiers": ["trip_id", "obs_index", "arc_id"],
        "target": TARGET_COLUMN,
        "evaluation_only_columns": ["is_gt_in_candidates"],
        "approved_feature_columns": features,
        "candidate_geometry_fields": [],
        "connectivity_source": "trip-specific raw .arcs files; arc_id indexes that trip's directed arc list",
        "filtering": {
            "condition": "is_gt_in_candidates == True, applied per (trip_id, obs_index)",
            "training_observations_before": train_schema["raw_observation_count"],
            "training_observations_eligible": train_schema["eligible_observation_count"],
            "training_observations_excluded": train_schema["excluded_observation_count"],
            "training_candidate_rows_before": train_schema["candidate_rows_before_filtering"],
            "training_candidate_rows_after": train_schema["candidate_rows_after_filtering"],
        },
        "cross_validation": {"method": "GroupKFold", "n_splits": 5, "group_column": "trip_id", "seed": args.seed},
    }
    write_json(results_root / "dataset_schema.json", schema)
    (REPO_ROOT / "docs" / "PHASE2_DATASET_SCHEMA.md").write_text(_schema_markdown(schema), encoding="utf-8")

    eligible_train = filter_eligible_observations(train)
    if len(eligible_train) != train_schema["candidate_rows_after_filtering"]:
        raise AssertionError("Filtered training row count differs from schema summary")
    folds = make_grouped_folds(train, train_ids, n_splits=5)
    print(
        f"Training rows: {len(eligible_train):,}/{len(train):,}; eligible observations "
        f"{train_schema['eligible_observation_count']:,}; misses "
        f"{train_schema['candidate_miss_count']:,}/{train_schema['raw_observation_count']:,} "
        f"({train_schema['candidate_miss_rate']:.3%}).",
        flush=True,
    )

    model_results: dict[str, dict] = {}
    cv_csv_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for model_name in MODEL_NAMES:
        print(f"\n=== Cross-validation: {model_name} ===", flush=True)
        result = cross_validate_model(
            model_name, train, features, folds, train_context, seed=args.seed,
            store_oof=True, progress=True,
        )
        model_results[model_name] = result
        for row in result["fold_metrics"]:
            cv_csv_rows.append({**row, "configuration": "baseline"})
        for row in result["per_trip_metrics"]:
            cv_csv_rows.append({**row, "configuration": "baseline"})
        cv_csv_rows.append({
            "model": model_name,
            "row_type": "pooled_oof",
            "fold": None,
            "trip_id": None,
            "training_trip_ids": None,
            "validation_trip_ids": json.dumps(train_ids),
            "training_row_count": int(len(eligible_train)),
            "validation_row_count": int(len(train)),
            "training_time_s": result["cv_training_time_s"],
            "inference_time_s": result["cv_inference_time_s"],
            **result["pooled_metrics"],
            "configuration": "baseline",
        })
        comparison_rows.append(_comparison_row(result))
        print(
            f"  pooled OOF: Macro F1={result['pooled_metrics']['macro_f1']:.4f}, "
            f"Precision={result['pooled_metrics']['candidate_positive_precision']:.4f}, "
            f"Candidate Accuracy={result['pooled_metrics']['candidate_binary_accuracy']:.4f}, "
            f"Top-1={result['pooled_metrics']['top1_accuracy']:.4f}, "
            f"LogLoss={result['pooled_metrics']['log_loss']:.4f}, "
            f"continuity={result['pooled_metrics']['route_continuity']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows)
    for rank_column, metric_column in [
        ("macro_f1_rank", "pooled_macro_f1"),
        ("candidate_precision_rank", "pooled_candidate_positive_precision"),
        ("candidate_accuracy_rank", "pooled_candidate_binary_accuracy"),
        ("top1_accuracy_rank", "pooled_top1_accuracy"),
    ]:
        comparison[rank_column] = comparison[metric_column].rank(method="min", ascending=False).astype("Int64")
    selected_name, selection = select_model(comparison)
    comparison["selection_score"] = comparison["model"].map(selection["model_scores"])
    comparison.to_csv(results_root / "model_comparison.csv", index=False)
    comparison_plots(comparison, plots_dir)
    model_details = {}
    for name, result in model_results.items():
        model_details[name] = {
            key: value for key, value in result.items() if key != "oof_probabilities"
        }
    write_json(results_root / "model_comparison.json", {
        "selection": selection,
        "comparison": comparison.to_dict("records"),
        "models": model_details,
    })

    # Optuna uses the exact same five grouped folds and does not see student-test rows.
    print(f"\n=== Bayesian optimization: {selected_name} ({args.trials} trials) ===", flush=True)
    study, optimization, trial_history = optimize_selected_model(
        selected_name, train, features, folds, train_context,
        n_trials=args.trials, seed=args.seed, progress=True,
    )
    trial_history.to_csv(results_root / "optimization_trials.csv", index=False)
    write_json(results_root / "best_hyperparameters.json", optimization)
    optimization_figure(study, plots_dir / "optimization_history.png")

    print("\n=== Re-evaluating tuned model on common folds ===", flush=True)
    tuned_cv = cross_validate_model(
        selected_name, train, features, folds, train_context,
        params=optimization["best_params"], seed=args.seed, store_oof=True, progress=True,
    )
    for row in tuned_cv["fold_metrics"]:
        cv_csv_rows.append({**row, "configuration": "tuned"})
    for row in tuned_cv["per_trip_metrics"]:
        cv_csv_rows.append({**row, "configuration": "tuned"})
    cv_csv_rows.append({
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
    pd.DataFrame(cv_csv_rows).to_csv(results_root / "cv_results.csv", index=False)

    baseline_result = model_results[selected_name]
    tuned_vs_untuned = pd.DataFrame([
        {
            "configuration": "untuned_baseline",
            "macro_f1_mean": baseline_result["mean_macro_f1"],
            "macro_f1_std": baseline_result["std_macro_f1"],
            "candidate_positive_precision_mean": baseline_result["mean_candidate_positive_precision"],
            "candidate_binary_accuracy_mean": baseline_result["mean_candidate_binary_accuracy"],
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
            "candidate_positive_precision_mean": tuned_cv["mean_candidate_positive_precision"],
            "candidate_binary_accuracy_mean": tuned_cv["mean_candidate_binary_accuracy"],
            "top1_accuracy_mean": tuned_cv["mean_top1_accuracy"],
            "log_loss_mean": tuned_cv["mean_log_loss"],
            "route_continuity_pooled": tuned_cv["pooled_metrics"]["route_continuity"],
            "training_time_s": tuned_cv["cv_training_time_s"],
            "inference_time_s": tuned_cv["cv_inference_time_s"],
            "params": json.dumps(optimization["best_params"], sort_keys=True),
        },
    ])
    tuned_vs_untuned.to_csv(results_root / "tuned_vs_untuned.csv", index=False)

    print("\n=== Fitting final model and evaluating student-test trips ===", flush=True)
    final_model = make_model_pipeline(selected_name, optimization["best_params"], args.seed)
    x_final = eligible_train[features].replace([np.inf, -np.inf], np.nan)
    y_final = eligible_train[TARGET_COLUMN].astype(np.int8).to_numpy()
    start = time.perf_counter()
    final_model.fit(x_final, y_final)
    final_training_seconds = time.perf_counter() - start
    observation_index = pd.DataFrame([
        {TRIP_COLUMN: trip_id, OBSERVATION_COLUMN: obs_idx}
        for trip_id, context in test_context.items()
        for obs_idx in range(int(context["observation_count"]))
    ])
    start = time.perf_counter()
    test_probabilities, hmm_records = predict_candidate_probabilities(
        final_model, student_test, features, observations=observation_index
    )
    student_test_inference_seconds = time.perf_counter() - start

    # Verify all nonempty candidate groups normalize and empty groups stay empty.
    probability_sums = test_probabilities.groupby([TRIP_COLUMN, OBSERVATION_COLUMN])["probability"].sum()
    if len(probability_sums) and not np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-6):
        raise AssertionError("HMM candidate probabilities do not sum to one within every observation")
    if not np.isfinite(test_probabilities["probability"].to_numpy()).all() or (test_probabilities["probability"] < 0).any():
        raise AssertionError("HMM probabilities must be finite and non-negative")
    student_metrics = __import__("src.models.phase2", fromlist=["evaluate_predictions"]).evaluate_predictions(
        student_test, test_probabilities["raw_probability"].to_numpy(), test_context, test_ids
    )
    output_parquet, output_json = save_probability_outputs(
        test_probabilities, hmm_records, predictions_dir, "student_test_candidate_probabilities"
    )

    metadata = {
        "selected_model": selected_name,
        "feature_columns": features,
        "target_column": TARGET_COLUMN,
        "identifier_columns": ["trip_id", "obs_index", "arc_id"],
        "random_seed": int(args.seed),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "optuna": __import__("optuna").__version__,
            "joblib": joblib.__version__,
        },
        "training_partition": {
            "source": str(train_path),
            "trip_count": len(train_ids),
            "trip_ids": train_ids,
            "eligible_observation_count": train_schema["eligible_observation_count"],
            "candidate_rows_after_filtering": len(eligible_train),
        },
        "cross_validation": {
            "method": "GroupKFold(n_splits=5)",
            "group_column": TRIP_COLUMN,
            "baseline_selected_model": baseline_result["pooled_metrics"],
            "tuned_selected_model": tuned_cv["pooled_metrics"],
            "fold_metrics": tuned_cv["fold_metrics"],
            "mean_metrics": {k: v for k, v in tuned_cv.items() if k.startswith(("mean_", "std_"))},
        },
        "student_test": {
            "source": str(test_path),
            "trip_count": len(test_ids),
            "trip_ids": test_ids,
            "metrics": student_metrics,
            "final_model_training_time_s": final_training_seconds,
            "inference_time_s": student_test_inference_seconds,
            "prediction_parquet": str(output_parquet),
            "prediction_json": str(output_json),
        },
        "hyperparameters": optimization["best_params"],
        "bayesian_optimization": optimization,
        "preprocessing": "median imputation fitted within each training fold; StandardScaler for LogisticRegression and MLP",
        "calibration_method": "none; probability quality checked with held-out Brier score, ECE, and Log Loss; HMM scores normalized within candidate sets",
        "candidate_probability_normalization": "clip invalid or negative scores to zero, then divide by the within-observation sum; uniform fallback when all scores are unusable",
        "final_training_time_s": final_training_seconds,
        "student_test_inference_time_s": student_test_inference_seconds,
    }
    artifact_paths = save_model_artifacts(
        final_model, features, selected_name, optimization["best_params"], metadata, artifacts_root
    )

    results_payload = {
        "selection": selection,
        "comparison": comparison.to_dict("records"),
        "optimization": optimization,
        "tuned_cv_metrics": tuned_cv["pooled_metrics"],
        "student_test_metrics": student_metrics,
        "artifact_paths": artifact_paths,
        "training_schema": train_schema,
        "student_test_schema": test_schema,
    }
    write_json(results_root / "model_comparison.json", {
        **results_payload,
        "baseline_models": model_details,
    })
    (REPO_ROOT / "docs" / "PHASE2_ML_IMPLEMENTATION.md").write_text(
        _implementation_markdown(
            schema, comparison, selection, optimization, tuned_vs_untuned,
            student_metrics, artifact_paths,
            {"final_training_time_s": final_training_seconds, "student_test_inference_time_s": student_test_inference_seconds},
        ),
        encoding="utf-8",
    )

    print("\n=== Final results ===", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"Selected: {selected_name}; tuned CV Log Loss={optimization['best_cv_log_loss']:.6f}", flush=True)
    print(f"Student test metrics: {json.dumps(json_safe(student_metrics), sort_keys=True)}", flush=True)
    print(f"Probability outputs: {output_parquet}, {output_json}", flush=True)
    print(f"Artifacts: {json.dumps(artifact_paths, sort_keys=True)}", flush=True)
    return results_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", default=str(REPO_ROOT / "data" / "processed" / "train_candidates.parquet"))
    parser.add_argument("--student-test-parquet", default=str(REPO_ROOT / "data" / "processed" / "student_test_candidates.parquet"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "data" / "raw" / "Student folder"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    parser.add_argument("--trials", type=int, default=5, help="Optuna TPE trials (default: 5)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
