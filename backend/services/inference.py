"""Model-backed trip loading, route reconstruction, and map generation."""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folium
import joblib
import numpy as np
import pandas as pd

from hmm.integrated_decoder import decode_candidate_routes
from src.data.loader import load_arcs, load_nodes, load_track
from src.features.build_features import build_trip_inference_features
from src.models.phase2 import (
    APPROVED_FEATURE_COLUMNS,
    ARC_COLUMN,
    OBSERVATION_COLUMN,
    TRIP_COLUMN,
    predict_candidate_probabilities,
)
from src.utils.geometry import arc_length, build_projection


TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_NETWORK_ARCS_IN_RESPONSE = 600
MAX_CANDIDATE_ARCS_IN_MAP = 1000


@dataclass(frozen=True)
class BackendSettings:
    repo_root: Path
    raw_root: Path
    artifact_dir: Path
    uploads_dir: Path
    maps_dir: Path
    demo_partition: str = "student_test"
    search_radius_m: float = 50.0
    reset_gap_seconds: float = 120.0
    transition_beta_m: float = 50.0


def default_settings() -> BackendSettings:
    root = Path(__file__).resolve().parents[2]
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass
    return BackendSettings(
        repo_root=root,
        raw_root=Path(__import__("os").environ.get("PATHSNAP_RAW_ROOT", root / "data" / "raw" / "Student folder")),
        artifact_dir=Path(__import__("os").environ.get("PATHSNAP_ARTIFACT_DIR", root / "artifacts")),
        uploads_dir=Path(__import__("os").environ.get("PATHSNAP_UPLOADS_DIR", root / "backend" / "uploads")),
        maps_dir=Path(__import__("os").environ.get("PATHSNAP_MAPS_DIR", root / "visualizations" / "maps")),
        demo_partition=__import__("os").environ.get("PATHSNAP_DEMO_PARTITION", "student_test"),
    )


def _route_lines(sequence: list[int | None], nodes: list[tuple[float, float]], arcs: list[tuple[int, int]]) -> list[list[list[float]]]:
    lines: list[list[list[float]]] = []
    current: list[list[float]] = []
    last_arc: int | None = None
    for arc_id in sequence:
        if arc_id is None:
            if len(current) >= 2:
                lines.append(current)
            current = []
            last_arc = None
            continue
        arc_id = int(arc_id)
        if arc_id == last_arc:
            continue
        from_node, to_node = arcs[arc_id]
        start = [float(nodes[from_node][1]), float(nodes[from_node][0])]
        end = [float(nodes[to_node][1]), float(nodes[to_node][0])]
        if current and current[-1] == start:
            current.append(end)
        else:
            if len(current) >= 2:
                lines.append(current)
            current = [start, end]
        last_arc = arc_id
    if len(current) >= 2:
        lines.append(current)
    return lines


def _continuity(sequence: list[int | None], arcs: list[tuple[int, int]], track: list[tuple[float, float, float]], max_gap: float) -> dict[str, Any]:
    evaluated = connected = 0
    for index in range(1, len(sequence)):
        previous, current = sequence[index - 1], sequence[index]
        if previous is None or current is None:
            continue
        interval = track[index][2] - track[index - 1][2]
        if interval <= 0 or interval > max_gap:
            continue
        evaluated += 1
        if previous == current or arcs[previous][1] == arcs[current][0]:
            connected += 1
    return {
        "connected_pairs": connected,
        "evaluated_pairs": evaluated,
        "route_continuity": connected / evaluated if evaluated else None,
    }


def _route_distance(sequence: list[int | None], lengths: list[float]) -> float:
    total = 0.0
    previous = None
    for arc_id in sequence:
        if arc_id is None:
            previous = None
        elif arc_id != previous:
            total += lengths[int(arc_id)]
            previous = int(arc_id)
    return total


class PathSnapInferenceService:
    """Loads the existing model once and serves route-free inference trips."""

    def __init__(self, settings: BackendSettings):
        self.settings = settings
        self.results: dict[str, dict[str, Any]] = {}
        self.model, self.preprocessing, self.feature_columns, self.metadata, self.hyperparameters = self._load_artifacts()
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.settings.maps_dir.mkdir(parents=True, exist_ok=True)

    def _load_artifacts(self) -> tuple[Any, Any, list[str], dict[str, Any], dict[str, Any]]:
        model_path = self.settings.artifact_dir / "best_model.pkl"
        preprocessing_path = self.settings.artifact_dir / "preprocessing_pipeline.pkl"
        schema_path = self.settings.artifact_dir / "feature_schema.json"
        metadata_path = self.settings.artifact_dir / "model_metadata.json"
        hyperparameters_path = self.settings.artifact_dir / "best_hyperparameters.json"
        missing = [str(path) for path in (model_path, preprocessing_path, schema_path, metadata_path, hyperparameters_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required Phase 2 artifacts are missing: " + ", ".join(missing))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        feature_columns = list(schema.get("feature_columns", []))
        if feature_columns != APPROVED_FEATURE_COLUMNS:
            raise ValueError("Saved model feature order does not match the approved Phase 1 schema")
        model = joblib.load(model_path)
        preprocessing = joblib.load(preprocessing_path)
        if not callable(getattr(preprocessing, "transform", None)):
            raise ValueError("Saved preprocessing artifact does not expose transform()")
        model_preprocessing = getattr(model, "named_steps", {}).get("preprocess")
        if model_preprocessing is None:
            raise ValueError("Saved estimator does not include its fitted preprocessing pipeline")
        preprocess_feature_count = getattr(preprocessing, "n_features_in_", len(feature_columns))
        if int(preprocess_feature_count) != len(feature_columns):
            raise ValueError("Saved preprocessing artifact has an incompatible feature count")
        model_feature_count = getattr(model, "n_features_in_", len(feature_columns))
        if int(model_feature_count) != len(feature_columns):
            raise ValueError(f"Saved estimator expects {model_feature_count} features; schema provides {len(feature_columns)}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
        metadata_features = metadata.get("feature_columns")
        if metadata_features is not None and list(metadata_features) != feature_columns:
            raise ValueError("Model metadata feature order does not match feature_schema.json")
        return model, preprocessing, feature_columns, metadata, hyperparameters

    @property
    def model_info(self) -> dict[str, Any]:
        hyperparameters = self.hyperparameters.get("hyperparameters", self.hyperparameters)
        return {
            "selected_model": self.metadata.get("selected_model", self.hyperparameters.get("model", "unknown")),
            "model_version": self.metadata.get("model_version", "phase2-selected"),
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "random_seed": self.metadata.get("random_seed"),
            "hyperparameters": hyperparameters,
            "score_note": "Candidate scores are normalized within each observation for HMM use; they are not calibrated probabilities.",
        }

    def _safe_trip_id(self, trip_id: str) -> str:
        if not TRIP_ID_PATTERN.fullmatch(trip_id):
            raise ValueError("Trip ID must contain only letters, digits, underscore, or hyphen")
        return trip_id

    def trip_dir(self, trip_id: str) -> Path:
        trip_id = self._safe_trip_id(trip_id)
        upload_path = self.settings.uploads_dir / trip_id
        if upload_path.is_dir() and self._has_required_files(upload_path, trip_id):
            return upload_path
        demo_path = self.settings.raw_root / self.settings.demo_partition / trip_id
        if demo_path.is_dir() and self._has_required_files(demo_path, trip_id):
            return demo_path
        raise FileNotFoundError(f"Unknown trip ID: {trip_id}")

    @staticmethod
    def _has_required_files(directory: Path, trip_id: str) -> bool:
        return all((directory / f"{trip_id}{suffix}").is_file() for suffix in (".track", ".nodes", ".arcs"))

    def list_trips(self) -> list[dict[str, Any]]:
        trips: list[dict[str, Any]] = []
        demo_dir = self.settings.raw_root / self.settings.demo_partition
        if demo_dir.is_dir():
            for child in sorted(demo_dir.iterdir()):
                if child.is_dir() and TRIP_ID_PATTERN.fullmatch(child.name) and self._has_required_files(child, child.name):
                    trips.append({"trip_id": child.name, "source": self.settings.demo_partition, "uploaded": False})
        if self.settings.uploads_dir.is_dir():
            for child in sorted(self.settings.uploads_dir.iterdir()):
                if child.is_dir() and child.name.startswith("upload-") and self._has_required_files(child, child.name):
                    trips.append({"trip_id": child.name, "source": "upload", "uploaded": True})
        return trips

    def load_trip(self, trip_id: str) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]], list[tuple[int, int]], list[str]]:
        directory = self.trip_dir(trip_id)
        parser_warnings: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            track = load_track(str(directory))
            nodes = load_nodes(str(directory))
            arcs = load_arcs(str(directory))
        parser_warnings = [str(item.message) for item in caught[:8]]
        if not track:
            raise ValueError("The trip has no valid GPS observations")
        if not nodes or not arcs:
            raise ValueError("The trip must contain at least one road node and arc")
        if any(not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat, _ in track):
            raise ValueError("Track contains invalid coordinates")
        if any(track[index][2] <= track[index - 1][2] for index in range(1, len(track))):
            raise ValueError("Track timestamps must be strictly increasing")
        if any(not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in nodes):
            raise ValueError("Road nodes contain invalid coordinates")
        for arc_id, (from_node, to_node) in enumerate(arcs):
            if not 0 <= from_node < len(nodes) or not 0 <= to_node < len(nodes):
                raise ValueError(f"Arc {arc_id} references a node outside the .nodes file")
        return track, nodes, arcs, parser_warnings

    def preview_trip(self, trip_id: str) -> dict[str, Any]:
        trip_id = self._safe_trip_id(trip_id)
        track, nodes, arcs, parser_warnings = self.load_trip(trip_id)
        return {
            "trip_id": trip_id,
            "source": "upload" if trip_id.startswith("upload-") else self.settings.demo_partition,
            "observation_count": len(track),
            "node_count": len(nodes),
            "arc_count": len(arcs),
            "start_time": track[0][2],
            "end_time": track[-1][2],
            "gps_coordinates": [{"lat": float(lat), "lon": float(lon), "timestamp": float(ts)} for lon, lat, ts in track],
            "parser_warnings": parser_warnings,
        }

    def _write_folium_map(
        self,
        trip_id: str,
        track: list[tuple[float, float, float]],
        nodes: list[tuple[float, float]],
        arcs: list[tuple[int, int]],
        candidates: pd.DataFrame,
        classifier_sequence: list[int | None],
        hmm_sequence: list[int | None],
    ) -> Path:
        centre = [float(np.mean([point[1] for point in track])), float(np.mean([point[0] for point in track]))]
        map_view = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

        gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
        gps_locations = [[float(lat), float(lon)] for lon, lat, _ in track]
        if gps_locations:
            folium.PolyLine(gps_locations, color="#3577f6", weight=2, opacity=0.55, tooltip="GPS trace").add_to(gps_layer)
        marker_step = max(1, len(track) // 400)
        for obs_id in range(0, len(track), marker_step):
            lon, lat, ts = track[obs_id]
            folium.CircleMarker([lat, lon], radius=2, color="#3577f6", fill=True, fill_opacity=0.7, tooltip=f"Observation {obs_id} · {ts:.1f}s").add_to(gps_layer)
        gps_layer.add_to(map_view)

        candidate_layer = folium.FeatureGroup(name="Candidate road arcs (sample)", show=False)
        if not candidates.empty:
            for arc_id in candidates[ARC_COLUMN].drop_duplicates().head(MAX_CANDIDATE_ARCS_IN_MAP).astype(int):
                from_node, to_node = arcs[arc_id]
                folium.PolyLine(
                    [[nodes[from_node][1], nodes[from_node][0]], [nodes[to_node][1], nodes[to_node][0]]],
                    color="#8390a5", weight=1, opacity=0.45, tooltip=f"Candidate arc {arc_id}",
                ).add_to(candidate_layer)
        candidate_layer.add_to(map_view)

        for name, sequence, color, shown in (
            ("Classifier top-1", classifier_sequence, "#ef8b2c", False),
            ("HMM / Viterbi route", hmm_sequence, "#148c67", True),
        ):
            route_layer = folium.FeatureGroup(name=name, show=shown)
            for line in _route_lines(sequence, nodes, arcs):
                folium.PolyLine(line, color=color, weight=5, opacity=0.9, tooltip=name).add_to(route_layer)
            route_layer.add_to(map_view)

        folium.LayerControl(collapsed=False).add_to(map_view)
        map_view.fit_bounds(gps_locations)
        path = self.settings.maps_dir / f"reconstruction_{trip_id}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        map_view.save(str(path))
        return path

    def map_match(self, trip_id: str) -> dict[str, Any]:
        trip_id = self._safe_trip_id(trip_id)
        directory = self.trip_dir(trip_id)
        track, nodes, arcs, parser_warnings = self.load_trip(trip_id)
        started = time.perf_counter()
        candidates = build_trip_inference_features(trip_id, str(directory), radius=self.settings.search_radius_m)
        expected_observations = pd.DataFrame({TRIP_COLUMN: trip_id, OBSERVATION_COLUMN: np.arange(len(track), dtype=np.int32)})
        probabilities, _records = predict_candidate_probabilities(
            self.model, candidates, self.feature_columns, expected_observations
        )
        if not probabilities.empty:
            if not np.isfinite(probabilities["probability"].to_numpy()).all() or (probabilities["probability"] < 0).any():
                raise ValueError("The model produced non-finite or negative candidate scores")
            sums = probabilities.groupby([TRIP_COLUMN, OBSERVATION_COLUMN], sort=False)["probability"].sum()
            if not np.allclose(sums.to_numpy(), 1.0, atol=1e-8):
                raise ValueError("Candidate scores did not normalize to one within each observation")

        decoded = decode_candidate_routes(
            track,
            nodes,
            arcs,
            probabilities,
            beta_m=self.settings.transition_beta_m,
            reset_gap_seconds=self.settings.reset_gap_seconds,
        )
        classifier_sequence = decoded["classifier_arc_ids"]
        hmm_sequence = decoded["hmm_arc_ids"]
        projection = build_projection(track)
        lengths = [arc_length(nodes, start, end, projection) for start, end in arcs]
        classifier_continuity = _continuity(classifier_sequence, arcs, track, self.settings.reset_gap_seconds)
        hmm_continuity = _continuity(hmm_sequence, arcs, track, self.settings.reset_gap_seconds)
        candidate_counts = probabilities.groupby(OBSERVATION_COLUMN, sort=False).size() if not probabilities.empty else pd.Series(dtype=int)
        matched_observations = sum(value is not None for value in hmm_sequence)
        elapsed = time.perf_counter() - started

        map_path = self._write_folium_map(
            trip_id, track, nodes, arcs, probabilities, classifier_sequence, hmm_sequence
        )
        result = {
            "trip_id": trip_id,
            "status": "completed" if matched_observations else "completed_no_candidates",
            "decoder_status": decoded["decoder_status"],
            "observations": [{"observation_id": index, "arc_id": arc_id} for index, arc_id in enumerate(classifier_sequence)],
            "gps_coordinates": [{"lat": float(lat), "lon": float(lon), "timestamp": float(ts)} for lon, lat, ts in track],
            "classifier_route": {"arc_ids_by_observation": classifier_sequence, "segments": _route_lines(classifier_sequence, nodes, arcs)},
            "hmm_route": {"arc_ids_by_observation": hmm_sequence, "segments": _route_lines(hmm_sequence, nodes, arcs)},
            "candidate_road_segments": self._candidate_road_segments(probabilities, nodes, arcs, classifier_sequence, hmm_sequence),
            "metrics": {
                "candidate_count": int(len(probabilities)),
                "observations_with_candidates": int(candidate_counts.size),
                "observations_without_candidates": int(len(track) - candidate_counts.size),
                "hmm_matched_observations": int(matched_observations),
                "classifier_route_continuity": classifier_continuity["route_continuity"],
                "hmm_route_continuity": hmm_continuity["route_continuity"],
                "classifier_connected_pairs": classifier_continuity["connected_pairs"],
                "hmm_connected_pairs": hmm_continuity["connected_pairs"],
                "evaluated_pairs": max(classifier_continuity["evaluated_pairs"], hmm_continuity["evaluated_pairs"]),
                "classifier_route_distance_m": _route_distance(classifier_sequence, lengths),
                "hmm_route_distance_m": _route_distance(hmm_sequence, lengths),
                "processing_time_seconds": elapsed,
            },
            "model": self.model_info,
            "warnings": parser_warnings + decoded["warnings"],
            "folium_map_url": f"/api/trips/{trip_id}/map",
        }
        self.results[trip_id] = result
        return result

    @staticmethod
    def _candidate_road_segments(
        candidates: pd.DataFrame,
        nodes: list[tuple[float, float]],
        arcs: list[tuple[int, int]],
        classifier_sequence: list[int | None],
        hmm_sequence: list[int | None],
    ) -> list[dict[str, Any]]:
        if candidates.empty:
            selected: list[int] = []
        else:
            means = candidates.groupby(ARC_COLUMN, sort=False)["probability"].mean().sort_values(ascending=False)
            selected = list(map(int, means.head(MAX_NETWORK_ARCS_IN_RESPONSE).index))
        segments = []
        for arc_id in selected:
            from_node, to_node = arcs[arc_id]
            segments.append({
                "arc_id": arc_id,
                "coordinates": [[float(nodes[from_node][1]), float(nodes[from_node][0])], [float(nodes[to_node][1]), float(nodes[to_node][0])]],
            })
        return segments

    def save_uploaded_trip(self, trip_id: str, files: dict[str, bytes]) -> Path:
        self._safe_trip_id(trip_id)
        if set(files) != {".track", ".nodes", ".arcs"}:
            raise ValueError("Upload exactly one .track, one .nodes, and one .arcs file")
        directory = self.settings.uploads_dir / trip_id
        if directory.exists():
            raise FileExistsError(f"Trip {trip_id} already exists")
        directory.mkdir(parents=True)
        try:
            for suffix, content in files.items():
                if not content:
                    raise ValueError(f"Uploaded {suffix} file is empty")
                (directory / f"{trip_id}{suffix}").write_bytes(content)
            self.load_trip(trip_id)
        except Exception:
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()
            raise
        return directory

    def folium_map_path(self, trip_id: str) -> Path | None:
        self._safe_trip_id(trip_id)
        path = self.settings.maps_dir / f"reconstruction_{trip_id}.html"
        return path if path.is_file() else None
