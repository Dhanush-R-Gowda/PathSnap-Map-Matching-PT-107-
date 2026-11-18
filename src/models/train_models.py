"""Phase 2 model training: fit five classifiers, compare, select best, save artifacts."""

import os
import sys
import json
import logging
import warnings
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier

from src.models.data_prep import load_and_prepare, FEATURE_COLUMNS
from src.models.evaluation import cross_validate_model, evaluate_on_test

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models():
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "GaussianNB": GaussianNB(),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_training():
    # 1. Load data
    logger.info("=" * 60)
    logger.info("Phase 2 Model Training")
    logger.info("=" * 60)
    data = load_and_prepare()

    X_train = data["X_train"]
    y_train = data["y_train"]
    groups_train = data["groups_train"]
    meta_train = data["meta_train"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    groups_test = data["groups_test"]
    meta_test = data["meta_test"]

    # 2. Define models (LightGBM first)
    models = _build_models()

    # 3. Cross-validate each model
    cv_results = {}
    for model_name, model in models.items():
        logger.info("=" * 50)
        logger.info("CV: %s", model_name)
        try:
            result = cross_validate_model(
                model_name, model, X_train, y_train, groups_train, meta_train
            )
        except Exception as exc:
            logger.error("CV failed entirely for %s: %s", model_name, exc, exc_info=True)
            result = {
                "fold_metrics": [],
                "mean_macro_f1": float("nan"),
                "std_macro_f1": float("nan"),
                "mean_top1_acc": float("nan"),
                "std_top1_acc": float("nan"),
                "mean_log_loss": float("nan"),
                "std_log_loss": float("nan"),
                "mean_fit_time": float("nan"),
                "mean_inference_time": float("nan"),
                "model_name": model_name,
            }
        cv_results[model_name] = result
        logger.info(
            "  CV summary | top1=%.4f±%.4f | macro_f1=%.4f±%.4f | log_loss=%.4f±%.4f",
            result["mean_top1_acc"], result["std_top1_acc"],
            result["mean_macro_f1"], result["std_macro_f1"],
            result["mean_log_loss"] if not np.isnan(result["mean_log_loss"]) else -1,
            result["std_log_loss"] if not np.isnan(result["std_log_loss"]) else 0,
        )

    # 4. Refit each model on the full training set + evaluate on student_test
    full_scaler = StandardScaler()
    X_train_scaled = full_scaler.fit_transform(X_train)

    fitted_models = {}
    test_results = {}

    for model_name, model in models.items():
        logger.info("Refitting %s on full training set ...", model_name)
        from sklearn.base import clone
        full_model = clone(model)
        try:
            full_model.fit(X_train_scaled, y_train)
            fitted_models[model_name] = full_model

            # Evaluate on student_test
            test_scaler = StandardScaler()
            test_scaler.fit(X_train)  # always fit scaler on train only
            t_res = evaluate_on_test(full_model, test_scaler, X_test, y_test, meta_test)
            test_results[model_name] = t_res
            logger.info(
                "  Test | top1=%.4f | macro_f1=%.4f | log_loss=%.4f",
                t_res["top1_acc"], t_res["macro_f1"],
                t_res["log_loss"] if not np.isnan(t_res["log_loss"]) else -1,
            )
        except Exception as exc:
            logger.error("Full refit failed for %s: %s", model_name, exc, exc_info=True)
            fitted_models[model_name] = None
            test_results[model_name] = {
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                "log_loss": float("nan"),
                "top1_acc": float("nan"),
                "fit_time_s": float("nan"),
                "inference_time_s": float("nan"),
            }

        # Save individual model artifact
        if fitted_models[model_name] is not None:
            model_path = os.path.join(ARTIFACTS_MODELS_DIR, f"model_{model_name}.joblib")
            joblib.dump(fitted_models[model_name], model_path)
            logger.info("  Saved model artifact: %s", model_path)

    # 5. Build comparison DataFrame
    rows = []
    for model_name in models:
        cv = cv_results[model_name]
        te = test_results[model_name]
        rows.append({
            "model_name": model_name,
            "cv_top1_acc_mean": cv["mean_top1_acc"],
            "cv_top1_acc_std": cv["std_top1_acc"],
            "cv_macro_f1_mean": cv["mean_macro_f1"],
            "cv_macro_f1_std": cv["std_macro_f1"],
            "cv_log_loss_mean": cv["mean_log_loss"],
            "cv_log_loss_std": cv["std_log_loss"],
            "student_test_top1_acc": te["top1_acc"],
            "student_test_macro_f1": te["macro_f1"],
            "student_test_log_loss": te["log_loss"],
            "mean_fit_time_s": cv["mean_fit_time"],
            "mean_inference_time_s": cv["mean_inference_time"],
        })

    comparison_df = pd.DataFrame(rows)

    # Save CSVs
    results_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    artifacts_csv = os.path.join(ARTIFACTS_MODELS_DIR, "model_comparison.csv")
    comparison_df.to_csv(results_csv, index=False)
    comparison_df.to_csv(artifacts_csv, index=False)
    logger.info("Saved model_comparison.csv to results/ and artifacts/")

    # 6. Print comparison table
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))

    # 7. Select best model based on CV per-observation Top-1 accuracy
    valid_rows = comparison_df.dropna(subset=["cv_top1_acc_mean"])
    if valid_rows.empty:
        logger.error("All models have NaN CV top1 accuracy; cannot select best model.")
        return comparison_df, None, None

    best_idx = valid_rows["cv_top1_acc_mean"].idxmax()
    best_model_name = valid_rows.loc[best_idx, "model_name"]
    best_top1 = valid_rows.loc[best_idx, "cv_top1_acc_mean"]

    # Tiebreaker: if any other model is within 0.5% of the best, use log loss
    threshold = 0.005
    close_models = valid_rows[
        (best_top1 - valid_rows["cv_top1_acc_mean"]) <= threshold
    ].copy()

    if len(close_models) > 1:
        # Use log loss as tiebreaker (lower is better); drop models where log_loss is NaN
        close_with_ll = close_models.dropna(subset=["cv_log_loss_mean"])
        if not close_with_ll.empty:
            best_idx_ll = close_with_ll["cv_log_loss_mean"].idxmin()
            winner_name = close_with_ll.loc[best_idx_ll, "model_name"]
            runner_names = [n for n in close_models["model_name"] if n != winner_name]
            rationale = (
                f"Models within 0.5% CV top-1: {list(close_models['model_name'].values)}. "
                f"Tiebreaker: lowest CV log loss. "
                f"'{winner_name}' wins with log_loss="
                f"{close_with_ll.loc[best_idx_ll, 'cv_log_loss_mean']:.5f} over "
                f"{runner_names}."
            )
            best_model_name = winner_name
        else:
            # All close models lack log_loss; keep highest top1
            rationale = (
                f"'{best_model_name}' selected with CV top-1 accuracy={best_top1:.4f}. "
                f"Tiebreaker candidates lacked log_loss; kept highest top-1."
            )
    else:
        rationale = (
            f"'{best_model_name}' selected with highest CV per-observation "
            f"Top-1 accuracy={best_top1:.4f}. No model within 0.5% required tiebreaker."
        )

    logger.info("\n%s", rationale)
    print(f"\nSelected model: {best_model_name}")
    print(f"Rationale: {rationale}")

    # Save selection JSON
    selection = {"model_name": best_model_name, "rationale": rationale}
    selection_path = os.path.join(RESULTS_DIR, "selected_model.json")
    with open(selection_path, "w") as fh:
        json.dump(selection, fh, indent=2)
    logger.info("Saved selected_model.json: %s", selection_path)

    return comparison_df, best_model_name, fitted_models


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_training()
