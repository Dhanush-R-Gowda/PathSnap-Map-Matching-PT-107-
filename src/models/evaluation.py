"""Phase 2 evaluation utilities: cross-validation, top-1 accuracy, test evaluation."""

import logging
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, log_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: per-observation Top-1 accuracy
# ---------------------------------------------------------------------------

def compute_top1_accuracy(
    y_true: np.ndarray,
    scores: np.ndarray,
    meta_df: pd.DataFrame,
) -> float:
    """Compute per-observation Top-1 accuracy.

    For each (trip_id, obs_index) group, pick the candidate with the highest
    score and check if that candidate has label == 1.  Returns the fraction
    of observations where the top-scoring candidate is the GT arc.

    Parameters
    ----------
    y_true : 1-D int array aligned with meta_df rows
    scores : 1-D float array aligned with meta_df rows (higher = more positive)
    meta_df : DataFrame with columns trip_id, obs_index (index reset to 0..n-1)

    Returns
    -------
    float — per-observation Top-1 accuracy in [0, 1]
    """
    df = meta_df[["trip_id", "obs_index"]].copy()
    df["y_true"] = y_true
    df["score"] = scores

    # For each observation group find the row with the highest score
    idx_best = df.groupby(["trip_id", "obs_index"])["score"].idxmax()
    best_labels = df.loc[idx_best, "y_true"].values
    return float(np.mean(best_labels == 1))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model_name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """Run GroupKFold CV and return aggregated metrics.

    For each fold a fresh StandardScaler is fitted on X_train_fold only.

    Parameters
    ----------
    model_name : str
    model : sklearn-compatible estimator (cloned each fold)
    X : feature matrix (n_samples, n_features)
    y : label array (n_samples,)
    groups : trip_id array (n_samples,)
    meta_df : DataFrame aligned to X/y with columns trip_id, obs_index, arc_id
    n_splits : number of CV folds

    Returns
    -------
    dict with keys:
        fold_metrics, mean_macro_f1, std_macro_f1,
        mean_top1_acc, std_top1_acc,
        mean_log_loss, std_log_loss,
        mean_fit_time, mean_inference_time,
        model_name
    """
    from sklearn.base import clone

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    has_proba = hasattr(model, "predict_proba")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        try:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            meta_val = meta_df.iloc[val_idx].reset_index(drop=True)

            # Scaler fitted on train fold only
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            # Fit model
            fold_model = clone(model)
            t0 = time.perf_counter()
            fold_model.fit(X_tr_scaled, y_tr)
            fit_time = time.perf_counter() - t0

            # MLP fallback: if fit took more than 10 minutes, reduce to (32,)
            if fit_time > 600 and hasattr(fold_model, "hidden_layer_sizes"):
                logger.warning(
                    "Fold %d: MLP fit time %.1fs > 600s, retrying with hidden_layer_sizes=(32,)",
                    fold_idx, fit_time
                )
                from sklearn.neural_network import MLPClassifier
                fold_model = MLPClassifier(
                    hidden_layer_sizes=(32,),
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                )
                t0 = time.perf_counter()
                fold_model.fit(X_tr_scaled, y_tr)
                fit_time = time.perf_counter() - t0

            # Predict
            t0 = time.perf_counter()
            y_pred = fold_model.predict(X_val_scaled)
            infer_time = time.perf_counter() - t0

            # Scores for Top-1 (probability of positive class if available)
            if has_proba and hasattr(fold_model, "predict_proba"):
                scores = fold_model.predict_proba(X_val_scaled)[:, 1]
                ll = log_loss(y_val, scores)
            elif hasattr(fold_model, "decision_function"):
                scores = fold_model.decision_function(X_val_scaled)
                ll = float("nan")
            else:
                # Fall back to prediction as 0/1 score
                scores = y_pred.astype(float)
                ll = float("nan")

            macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_val, y_pred)
            top1 = compute_top1_accuracy(y_val, scores, meta_val)

            fold_metrics.append({
                "fold": fold_idx,
                "model_name": model_name,
                "macro_f1": macro_f1,
                "accuracy": acc,
                "log_loss": ll,
                "top1_acc": top1,
                "fit_time_s": fit_time,
                "inference_time_s": infer_time,
            })
            logger.info(
                "Fold %d | macro_f1=%.4f | top1=%.4f | log_loss=%.4f | fit=%.1fs",
                fold_idx, macro_f1, top1, ll, fit_time
            )

        except Exception as exc:
            logger.error("Fold %d failed for %s: %s", fold_idx, model_name, exc, exc_info=True)
            fold_metrics.append({
                "fold": fold_idx,
                "model_name": model_name,
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                "log_loss": float("nan"),
                "top1_acc": float("nan"),
                "fit_time_s": float("nan"),
                "inference_time_s": float("nan"),
            })

    if not fold_metrics:
        logger.error("All folds failed for model %s", model_name)
        return {
            "fold_metrics": fold_metrics,
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

    # Aggregate (ignoring NaN folds)
    def _nanmean(key):
        vals = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _nanstd(key):
        vals = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    return {
        "fold_metrics": fold_metrics,
        "mean_macro_f1": _nanmean("macro_f1"),
        "std_macro_f1": _nanstd("macro_f1"),
        "mean_top1_acc": _nanmean("top1_acc"),
        "std_top1_acc": _nanstd("top1_acc"),
        "mean_log_loss": _nanmean("log_loss"),
        "std_log_loss": _nanstd("log_loss"),
        "mean_fit_time": _nanmean("fit_time_s"),
        "mean_inference_time": _nanmean("inference_time_s"),
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    model,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
) -> dict:
    """Evaluate a fitted model on the test set.

    The model and scaler must already be fitted. This function only applies
    scaler.transform and runs inference.

    Returns
    -------
    dict with the same keys as the per-fold dict inside fold_metrics:
        macro_f1, accuracy, log_loss, top1_acc, fit_time_s, inference_time_s
    """
    X_scaled = scaler.transform(X_test)

    t0 = time.perf_counter()
    y_pred = model.predict(X_scaled)
    infer_time = time.perf_counter() - t0

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        scores = model.predict_proba(X_scaled)[:, 1]
        ll = log_loss(y_test, scores)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_scaled)
        ll = float("nan")
    else:
        scores = y_pred.astype(float)
        ll = float("nan")

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    top1 = compute_top1_accuracy(y_test, scores, meta_test)

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "log_loss": ll,
        "top1_acc": top1,
        "fit_time_s": 0.0,
        "inference_time_s": infer_time,
    }
