"""Integration checks for the generated Phase 2 inference artifacts.

Skipped when the artifacts have not been generated yet (they are gitignored by
design). Run `python -m src.models.run_phase2 --trials 5 --seed 42` first.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
TRIP = ROOT / "data/raw/Student folder/student_test/00000052"

REQUIRED = (
    "best_model.pkl",
    "preprocessing_pipeline.pkl",
    "feature_schema.json",
    "model_metadata.json",
    "best_hyperparameters.json",
)

pytestmark = pytest.mark.skipif(
    not all((ART / name).exists() for name in REQUIRED),
    reason="Phase 2 artifacts not generated",
)


def test_required_artifacts_exist():
    for name in REQUIRED:
        assert (ART / name).is_file(), f"missing artifact: {ART / name}"


def test_feature_schema_matches_approved_features():
    from src.models.phase2 import APPROVED_FEATURE_COLUMNS

    schema = json.loads((ART / "feature_schema.json").read_text())
    assert schema["feature_columns"] == APPROVED_FEATURE_COLUMNS
    assert len(schema["feature_columns"]) == 14


def test_loaded_model_is_a_pipeline_with_preprocessing():
    from sklearn.pipeline import Pipeline

    from src.models.phase2 import load_model

    model = load_model(ART / "best_model.pkl")
    assert isinstance(model, Pipeline)
    assert "preprocess" in model.named_steps
    assert "classifier" in model.named_steps


def test_metadata_matches_selected_model_and_hyperparameters():
    meta = json.loads((ART / "model_metadata.json").read_text())
    params = json.loads((ART / "best_hyperparameters.json").read_text())
    assert meta["selected_model"]
    assert meta["hyperparameters"] == params["hyperparameters"]


@pytest.mark.skipif(not TRIP.is_dir(), reason="raw data not available")
def test_pipeline_end_to_end_uses_artifact_schema():
    from src.data.loader import load_trip_for_inference
    from src.pipeline import PathSnapMatcher

    matcher = PathSnapMatcher(ART)
    assert matcher.feature_columns == json.loads(
        (ART / "feature_schema.json").read_text()
    )["feature_columns"]

    d = load_trip_for_inference(str(TRIP))
    out = matcher.match("00000052", d["track"], d["nodes"], d["arcs"])
    assert len(out["result"].obs_arc) == len(d["track"])
    assert out["result"].arc_sequence
    assert 0.0 <= out["continuity"] <= 1.0
