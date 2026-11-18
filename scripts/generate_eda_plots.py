"""Generate EDA plots and confusion matrices for the final technical report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.loader import load_trip
from src.utils.geometry import build_projection, project

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_eda_kinematics_and_trajectory():
    trip_id = "00000052"
    trip_path = REPO_ROOT / "data" / "raw" / "Student folder" / "student_test" / trip_id
    d = load_trip(str(trip_path))
    track = d["track"]
    proj = build_projection(track)

    coords_xy = [project(lon, lat, proj) for lon, lat, _ in track]
    xs, ys = zip(*coords_xy)
    ts = np.array([t for _, _, t in track])
    dt = np.diff(ts)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Trajectory in projected local metric plane
    axes[0].plot(xs, ys, "b.-", alpha=0.6, markersize=3, label="GPS trajectory")
    axes[0].scatter([xs[0]], [ys[0]], color="green", s=60, zorder=5, label="Start")
    axes[0].scatter([xs[-1]], [ys[-1]], color="red", s=60, zorder=5, label="End")
    axes[0].set_title(f"Trip {trip_id}: GPS Trajectory (Metric Plane)")
    axes[0].set_xlabel("Local X (meters)")
    axes[0].set_ylabel("Local Y (meters)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 2. Sampling interval distribution
    axes[1].hist(dt, bins=np.arange(0, 10, 0.5), color="#3478f6", edgecolor="black", alpha=0.7)
    axes[1].axvline(1.0, color="red", linestyle="--", label="Nominal 1.0s (97%)")
    axes[1].set_title(f"Sampling Time Delta (dt) Distribution")
    axes[1].set_xlabel("dt (seconds)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # 3. GPS Speeds
    dx = np.diff(xs)
    dy = np.diff(ys)
    speeds = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-3)
    speeds_kmh = speeds * 3.6
    axes[2].plot(ts[1:], speeds_kmh, color="#e08a00", linewidth=1.5)
    axes[2].set_title(f"Instantaneous Speed Profile")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Speed (km/h)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = PLOTS_DIR / "eda_trajectory_and_kinematics.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved", out)


def plot_eda_road_network():
    trip_id = "00000052"
    trip_path = REPO_ROOT / "data" / "raw" / "Student folder" / "student_test" / trip_id
    d = load_trip(str(trip_path))
    nodes = d["nodes"]
    arcs = d["arcs"]
    track = d["track"]
    proj = build_projection(track)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot road network arcs
    for from_id, to_id in arcs:
        lon1, lat1 = nodes[from_id]
        lon2, lat2 = nodes[to_id]
        x1, y1 = project(lon1, lat1, proj)
        x2, y2 = project(lon2, lat2, proj)
        ax.plot([x1, x2], [y1, y2], color="#aaaaaa", linewidth=0.8, alpha=0.7)

    # Plot ground-truth route
    route_arcs = d["route"]
    for arc_id in route_arcs:
        from_id, to_id = arcs[arc_id]
        lon1, lat1 = nodes[from_id]
        lon2, lat2 = nodes[to_id]
        x1, y1 = project(lon1, lat1, proj)
        x2, y2 = project(lon2, lat2, proj)
        ax.plot([x1, x2], [y1, y2], color="#2ca02c", linewidth=2.5, alpha=0.85)

    # Plot GPS observations
    gps_xy = [project(lon, lat, proj) for lon, lat, _ in track]
    gx, gy = zip(*gps_xy)
    ax.scatter(gx, gy, color="#1f77b4", s=8, alpha=0.8, label="GPS observations")
    ax.plot([], [], color="#aaaaaa", label="Road network arcs")
    ax.plot([], [], color="#2ca02c", linewidth=2.5, label="Reference route")

    ax.set_title(f"Trip {trip_id}: Road Network & Reference Route Traversed")
    ax.set_xlabel("Local X (meters)")
    ax.set_ylabel("Local Y (meters)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = PLOTS_DIR / "eda_road_network_overlay.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved", out)


def plot_confusion_matrix():
    preds_parquet = REPO_ROOT / "results" / "prediction_probabilities" / "student_test_candidate_probabilities.parquet"
    test_parquet = REPO_ROOT / "data" / "processed" / "student_test_candidates.parquet"
    if not preds_parquet.is_file() or not test_parquet.is_file():
        print("Missing parquets for confusion matrix")
        return

    df_test = pd.read_parquet(test_parquet)
    df_pred = pd.read_parquet(preds_parquet)
    if "observation_id" in df_pred.columns:
        df_pred = df_pred.rename(columns={"observation_id": "obs_index"})
    merged = pd.merge(df_test, df_pred, on=["trip_id", "obs_index", "arc_id"])
    y_true = merged["label"].values
    y_pred = (merged["probability"].values >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-matching (0)", "Matched Arc (1)"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Student Test Candidate Classification Confusion Matrix")
    plt.tight_layout()
    out = PLOTS_DIR / "candidate_confusion_matrix.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved", out)


def plot_classifier_vs_hmm():
    eval_json = REPO_ROOT / "results" / "pipeline_eval.json"
    if not eval_json.is_file():
        print("Missing pipeline_eval.json")
        return

    data = json.loads(eval_json.read_text())
    per_trip = data["per_trip"]
    trip_labels = [r["trip_id"][-4:] for r in per_trip]
    ml_top1 = [r["ml_top1"] * 100 for r in per_trip]
    hmm_top1 = [r["hmm_top1"] * 100 for r in per_trip]
    ml_cont = [r["ml_continuity"] * 100 for r in per_trip]
    hmm_cont = [r["hmm_continuity"] * 100 for r in per_trip]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(trip_labels))
    width = 0.38

    # Accuracy
    axes[0].bar(x - width/2, ml_top1, width, label="Classifier-only Top-1", color="#ff7f0e", alpha=0.85)
    axes[0].bar(x + width/2, hmm_top1, width, label="Classifier + HMM", color="#2ca02c", alpha=0.85)
    axes[0].set_ylabel("Pointwise Accuracy (%)")
    axes[0].set_title("Pointwise Accuracy: Classifier vs. HMM Decoder")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(trip_labels, rotation=45)
    axes[0].set_ylim(50, 105)
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Route continuity
    axes[1].bar(x - width/2, ml_cont, width, label="Classifier-only", color="#ff7f0e", alpha=0.85)
    axes[1].bar(x + width/2, hmm_cont, width, label="Classifier + HMM", color="#2ca02c", alpha=0.85)
    axes[1].set_ylabel("Route Continuity (%)")
    axes[1].set_title("Sequence Continuity: Classifier vs. HMM Decoder")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(trip_labels, rotation=45)
    axes[1].set_ylim(0, 110)
    axes[1].legend()
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = PLOTS_DIR / "classifier_vs_hmm_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved", out)


if __name__ == "__main__":
    plot_eda_kinematics_and_trajectory()
    plot_eda_road_network()
    plot_confusion_matrix()
    plot_classifier_vs_hmm()
