"""End-to-end route-free inference: GPS + road network -> ML probabilities -> HMM/Viterbi."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.decoder.graph import RoadGraph
from src.decoder.hmm_matcher import HMMConfig, decode, route_continuity
from src.features.build_features import build_trip_features_inference
from src.models.phase2 import load_model, predict_candidate_probabilities

REPO_ROOT = Path(__file__).resolve().parents[1]


class PathSnapMatcher:
    """Loads the model once; ``match`` never needs ground truth (.route)."""

    def __init__(self, artifact_dir: str | Path = REPO_ROOT / "artifacts", cfg: HMMConfig | None = None):
        artifact_dir = Path(artifact_dir)
        self.model = load_model(artifact_dir / "best_model.pkl")
        schema = json.loads((artifact_dir / "feature_schema.json").read_text())
        self.feature_columns = schema["feature_columns"]
        meta = artifact_dir / "model_metadata.json"
        self.metadata = json.loads(meta.read_text()) if meta.exists() else {}
        self.cfg = cfg or HMMConfig()

    def candidate_probabilities(self, trip_id, track, nodes, arcs, radius=50.0):
        feats = build_trip_features_inference(trip_id, track, nodes, arcs, radius=radius)
        if feats.empty:
            return feats.assign(probability=[]), feats
        frame, _ = predict_candidate_probabilities(self.model, feats, self.feature_columns)
        return frame, feats

    def match(self, trip_id, track, nodes, arcs, radius=50.0, use_hmm=True):
        t0 = time.perf_counter()
        graph = RoadGraph(nodes, arcs, track=track)
        frame, _ = self.candidate_probabilities(trip_id, track, nodes, arcs, radius)
        t_ml = time.perf_counter()
        cands = [[] for _ in track]
        for r in frame.itertuples(index=False):
            cands[int(r.obs_index)].append((int(r.arc_id), float(r.snap_t), float(r.probability)))
        res = decode(graph, track, cands, self.cfg)
        t1 = time.perf_counter()
        top1 = [max(c, key=lambda x: x[2])[0] if c else None for c in cands]
        return {
            "trip_id": trip_id,
            "graph": graph,
            "candidates": cands,
            "ml_top1": top1,
            "result": res,
            "continuity": route_continuity(graph, res.arc_sequence),
            "timing_s": {"features_and_ml": t_ml - t0, "hmm": t1 - t_ml, "total": t1 - t0},
        }
