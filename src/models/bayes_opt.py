"""Phase 2 Bayesian optimisation: tune the selected model and produce final artifacts."""

import os
import sys
import json
import logging
import time
import warnings
import numpy as np
import pandas as pd
import joblib

import optuna
from optuna.samplers import TPESampler

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from src.models.data_prep import load_and_prepare, FEATURE_COLUMNS
from src.models.evaluation import compute_top1_accuracy, evaluate_on_test

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "experiments")
ARTIFACTS_MODELS_DIR = os.path.join(REPO_ROOT, "artifacts", "models")
ARTIFACTS_CONFIGS_DIR = os.path.join(REPO_ROOT, "artifacts", "configs")
DATA_PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

for d in (RESULTS_DIR, ARTIFACTS_MODELS_DIR, ARTIFACTS_CONFIGS_DIR):
    os.makedirs(d, exist_ok=True)

TRAIN_PARQUET = os.path.join(DATA_PROCESSED_DIR, "train_candidates.parquet")
TEST_PARQUET = os.path.join(DATA_PROCESSED_DIR, "student_test_candidates.parquet")


# ---------------------------------------------------------------------------
# Model constructors from trial params
# ---------------------------------------------------------------------------

def _build_lgbm(trial):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _build_rf(trial):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _build_lr(trial):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 100.0, log=True),
        solver=trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )


def _build_mlp(trial):
    from sklearn.neural_network import MLPClassifier
    hidden_choices = [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)]
    hidden = trial.suggest_categorical(
        "hidden_layer_sizes",
        [str(h) for h in hidden_choices]
    )
    # Convert string representation back to tuple
    hidden = eval(hidden)  # safe: values come from our controlled list
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )


MODEL_BUILDERS = {
    "LightGBM": _build_lgbm,
    "RandomForest": _build_rf,
    "LogisticRegression": _build_lr,
    "MLP": _build_mlp,
}


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(model_name, X, y, groups, meta_df, n_splits=5):
    """Return an Optuna objective function for the given model type."""
    gkf = GroupKFold(n_splits=n_splits)
    builder = MODEL_BUILDERS[model_name]

    trial_rows = []  # will be populated outside via closure reference

    def objective(trial):
        t_start = time.perf_counter()
        model = builder(trial)

        top1_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            meta_val = meta_df.iloc[val_idx].reset_index(drop=True)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            m = clone(model)
            m.fit(X_tr_s, y_tr)

            if hasattr(m, "predict_proba"):
                scores = m.predict_proba(X_val_s)[:, 1]
            elif hasattr(m, "decision_function"):
                scores = m.decision_function(X_val_s)
            else:
                scores = m.predict(X_val_s).astype(float)

            top1_scores.append(compute_top1_accuracy(y_val, scores, meta_val))

        mean_top1 = float(np.mean(top1_scores))
        duration = time.perf_counter() - t_start

        trial_rows.append({
            "trial_number": trial.number,
            "model_name": model_name,
            "params": json.dumps(trial.params),
            "cv_top1_acc": mean_top1,
            "duration_s": duration,
        })

        return mean_top1

    return objective, trial_rows


# ---------------------------------------------------------------------------
# Main Bayes opt routine
# ---------------------------------------------------------------------------

def run_bayes_opt(n_trials: int = 30):
    logger.info("=" * 60)
    logger.info("Phase 2 Bayesian Optimisation")
    logger.info("=" * 60)

    # 1. Load data
    data = load_and_prepare()
    X_train = data["X_train"]
    y_train = data["y_train"]
    groups_train = data["groups_train"]
    meta_train = data["meta_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    meta_test = data["meta_test"]

    # 2. Read selected model name
    selection_path = os.path.join(RESULTS_DIR, "selected_model.json")
    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            f"selected_model.json not found at {selection_path}. "
            "Run train_models.py first."
        )
    with open(selection_path) as fh:
        selection = json.load(fh)
    model_name = selection["model_name"]
    logger.info("Selected model for optimisation: %s", model_name)

    # 3. Load baseline (full-train) model for comparison metrics
    baseline_model_path = os.path.join(ARTIFACTS_MODELS_DIR, f"model_{model_name}.joblib")
    if os.path.exists(baseline_model_path):
        baseline_model = joblib.load(baseline_model_path)
    else:
        baseline_model = None
        logger.warning("Baseline model artifact not found: %s", baseline_model_path)

    # Baseline scaler (refit on train)
    baseline_scaler = StandardScaler()
    baseline_scaler.fit(X_train)

    baseline_test_metrics = None
    if baseline_model is not None:
        baseline_test_metrics = evaluate_on_test(
            baseline_model, baseline_scaler, X_test, y_test, meta_test
        )

    # 4. GNB shortcut
    if model_name == "GaussianNB":
        logger.info("GaussianNB: skipping optimisation (already optimal).")
        final_model = baseline_model
        final_scaler = baseline_scaler
        best_params = {}
        study = None

        # Still produce required output files
        _save_configs_and_predictions(
            final_model, final_scaler, model_name, best_params,
            X_train, y_train, baseline_test_metrics, baseline_test_metrics,
            selection.get("cv_top1_acc", float("nan")),
            bayes_opt_rows=[],
            study=None,
        )
        return

    # 5. Optimise
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"No search space defined for model: {model_name}")

    objective_fn, trial_rows = _make_objective(
        model_name, X_train, y_train, groups_train, meta_train
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    logger.info("Starting Optuna study: %d trials ...", n_trials)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best trial: value=%.5f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Save Optuna study
    study_path = os.path.join(RESULTS_DIR, f"optuna_study_{model_name}.pkl")
    joblib.dump(study, study_path)
    logger.info("Saved Optuna study: %s", study_path)

    # Save bayes_opt_results CSV
    bayes_df = pd.DataFrame(trial_rows)
    bayes_csv = os.path.join(RESULTS_DIR, "bayes_opt_results.csv")
    bayes_df.to_csv(bayes_csv, index=False)
    logger.info("Saved bayes_opt_results.csv: %s", bayes_csv)

    best_params = study.best_params

    # 6. Refit tuned model on full training set
    logger.info("Refitting tuned model on full training set ...")
    best_trial = study.best_trial
    tuned_model = MODEL_BUILDERS[model_name](best_trial)
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train)
    tuned_model.fit(X_train_scaled, y_train)

    # Save final model and scaler
    final_model_path = os.path.join(ARTIFACTS_MODELS_DIR, "final_model.joblib")
    final_scaler_path = os.path.join(ARTIFACTS_MODELS_DIR, "scaler.joblib")
    joblib.dump(tuned_model, final_model_path)
    joblib.dump(final_scaler, final_scaler_path)
    logger.info("Saved final_model.joblib and scaler.joblib")

    # 7. Evaluate tuned model on student_test
    tuned_test_metrics = evaluate_on_test(tuned_model, final_scaler, X_test, y_test, meta_test)
    logger.info(
        "Tuned test | top1=%.4f | macro_f1=%.4f | log_loss=%.4f",
        tuned_test_metrics["top1_acc"],
        tuned_test_metrics["macro_f1"],
        tuned_test_metrics["log_loss"] if not np.isnan(tuned_test_metrics["log_loss"]) else -1,
    )

    # 8. Save all configs and predictions
    _save_configs_and_predictions(
        tuned_model, final_scaler, model_name, best_params,
        X_train, y_train, baseline_test_metrics, tuned_test_metrics,
        study.best_value,
        bayes_opt_rows=trial_rows,
        study=study,
    )


def _save_configs_and_predictions(
    final_model, final_scaler, model_name, best_params,
    X_train, y_train, baseline_test_metrics, tuned_test_metrics,
    cv_top1_tuned,
    bayes_opt_rows, study,
):
    """Save all configs, comparison CSV, and HMM-ready predictions."""

    # --- Best params JSON ---
    best_params_path = os.path.join(ARTIFACTS_CONFIGS_DIR, "best_params.json")
    with open(best_params_path, "w") as fh:
        # Convert any non-serialisable types (e.g. tuples)
        serialisable_params = {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in best_params.items()}
        json.dump({"model_name": model_name, "params": serialisable_params}, fh, indent=2)
    logger.info("Saved best_params.json")

    # --- feature_columns.json ---
    feature_cols_path = os.path.join(ARTIFACTS_CONFIGS_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)
    logger.info("Saved feature_columns.json")

    # --- training_config.json ---
    training_config = {
        "radius_m": 50,
        "filter": "is_gt_in_candidates_true",
        "cv_strategy": "GroupKFold_n5",
        "random_seed": 42,
        "n_trials_bayes": 30,
    }
    training_config_path = os.path.join(ARTIFACTS_CONFIGS_DIR, "training_config.json")
    with open(training_config_path, "w") as fh:
        json.dump(training_config, fh, indent=2)
    logger.info("Saved training_config.json")

    # --- tuning_comparison CSV ---
    cv_top1_baseline = float("nan")
    if baseline_test_metrics is not None:
        # We don't have baseline CV top1 stored here; use NaN
        pass

    tuning_rows = []
    if baseline_test_metrics is not None:
        infer_ms_baseline = baseline_test_metrics["inference_time_s"] * 1000
        tuning_rows.append({
            "model_name": model_name,
            "config": "baseline",
            "cv_top1_acc": float("nan"),  # baseline CV not recomputed here
            "student_test_top1_acc": baseline_test_metrics["top1_acc"],
            "student_test_macro_f1": baseline_test_metrics["macro_f1"],
            "student_test_log_loss": baseline_test_metrics["log_loss"],
            "inference_time_ms": infer_ms_baseline,
        })

    if tuned_test_metrics is not None:
        infer_ms_tuned = tuned_test_metrics["inference_time_s"] * 1000
        tuning_rows.append({
            "model_name": model_name,
            "config": "tuned",
            "cv_top1_acc": cv_top1_tuned,
            "student_test_top1_acc": tuned_test_metrics["top1_acc"],
            "student_test_macro_f1": tuned_test_metrics["macro_f1"],
            "student_test_log_loss": tuned_test_metrics["log_loss"],
            "inference_time_ms": infer_ms_tuned,
        })

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_csv = os.path.join(RESULTS_DIR, "tuning_comparison.csv")
    tuning_df.to_csv(tuning_csv, index=False)
    logger.info("Saved tuning_comparison.csv")

    # --- HMM-ready predictions ---
    _generate_hmm_predictions(final_model, final_scaler)


def _generate_hmm_predictions(final_model, final_scaler):
    """Generate prob_positive for every candidate row (all rows, not filtered).

    For rows where is_gt_in_candidates==True: use predict_proba.
    For rows where is_gt_in_candidates==False: assign 1 / avg_candidates_per_obs.
    """
    logger.info("Generating HMM-ready predictions ...")

    # Compute avg_candidates_per_obs from filtered training data
    train_raw = pd.read_parquet(TRAIN_PARQUET)
    train_filtered = train_raw[train_raw["is_gt_in_candidates"] == True]
    avg_cands = (
        train_filtered.groupby(["trip_id", "obs_index"])
        .size()
        .mean()
    )
    fallback_prob = float(1.0 / avg_cands)
    logger.info("avg_candidates_per_obs (train filtered): %.2f  → fallback_prob=%.5f",
                avg_cands, fallback_prob)

    for parquet_path, out_name in [
        (TRAIN_PARQUET, "train_predictions.parquet"),
        (TEST_PARQUET, "student_test_predictions.parquet"),
    ]:
        df = pd.read_parquet(parquet_path)

        prob_positive = np.full(len(df), float("nan"), dtype=np.float32)

        # Rows with GT in candidates: predict
        mask_in = df["is_gt_in_candidates"].values == True
        if mask_in.any():
            X_in = df.loc[mask_in, FEATURE_COLUMNS].values.astype(np.float64)
            X_in_scaled = final_scaler.transform(X_in)
            if hasattr(final_model, "predict_proba"):
                probs = final_model.predict_proba(X_in_scaled)[:, 1]
            elif hasattr(final_model, "decision_function"):
                raw = final_model.decision_function(X_in_scaled)
                # Sigmoid transform to [0,1]
                probs = 1.0 / (1.0 + np.exp(-raw))
            else:
                probs = final_model.predict(X_in_scaled).astype(np.float32)
            prob_positive[mask_in] = probs.astype(np.float32)

        # Rows where GT is NOT in candidates: fallback
        mask_out = ~mask_in
        prob_positive[mask_out] = np.float32(fallback_prob)

        # Clip to [0, 1]
        prob_positive = np.clip(prob_positive, 0.0, 1.0).astype(np.float32)

        out_df = pd.DataFrame({
            "trip_id": df["trip_id"].values,
            "obs_index": df["obs_index"].values,
            "arc_id": df["arc_id"].values,
            "prob_positive": prob_positive,
            "is_gt_in_candidates": df["is_gt_in_candidates"].values,
        })

        out_path = os.path.join(DATA_PROCESSED_DIR, out_name)
        out_df.to_parquet(out_path, index=False)
        logger.info("Saved %s (%d rows)", out_path, len(out_df))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    import argparse

    parser = argparse.ArgumentParser(description="Bayesian optimisation for Phase 2")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()

    run_bayes_opt(n_trials=args.n_trials)
