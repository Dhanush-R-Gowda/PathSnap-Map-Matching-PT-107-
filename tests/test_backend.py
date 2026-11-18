import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.inference import BackendSettings


@pytest.fixture
def app_settings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    for filename in (
        "best_model.pkl",
        "preprocessing_pipeline.pkl",
        "feature_schema.json",
        "model_metadata.json",
        "best_hyperparameters.json",
    ):
        shutil.copy2(root / "artifacts" / filename, artifact_dir / filename)

    raw_root = tmp_path / "raw"
    trip_dir = raw_root / "student_test" / "00000001"
    trip_dir.mkdir(parents=True)
    (trip_dir / "00000001.track").write_text("0.0001 0.0001 0\n0.0009 0.0001 1\n", encoding="utf-8")
    (trip_dir / "00000001.nodes").write_text("0.0 0.0001\n0.0005 0.0001\n0.001 0.0001\n", encoding="utf-8")
    (trip_dir / "00000001.arcs").write_text("0 1\n1 2\n", encoding="utf-8")
    # There intentionally is no .route file in this inference fixture.

    settings = BackendSettings(
        repo_root=tmp_path,
        raw_root=raw_root,
        artifact_dir=artifact_dir,
        uploads_dir=tmp_path / "uploads",
        maps_dir=tmp_path / "maps",
        demo_partition="student_test",
    )
    return settings


def test_route_free_api_flow_loads_model_and_generates_folium_map(app_settings):
    app = create_app(app_settings)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        trips = client.get("/api/trips")
        assert trips.status_code == 200
        assert trips.json()["trips"][0]["trip_id"] == "00000001"

        preview = client.get("/api/trips/00000001")
        assert preview.status_code == 200
        assert preview.json()["observation_count"] == 2
        assert "route" not in json.dumps(preview.json()).lower()

        result = client.post("/api/map-match", json={"trip_id": "00000001"})
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["decoder_status"] in {"completed", "completed_with_resets", "no_candidates"}
        assert len(payload["gps_coordinates"]) == 2
        assert len(payload["classifier_route"]["arc_ids_by_observation"]) == 2
        assert len(payload["hmm_route"]["arc_ids_by_observation"]) == 2
        assert payload["model"]["feature_count"] == 14
        assert not (app_settings.raw_root / "student_test" / "00000001" / "00000001.route").exists()

        saved = client.get("/api/trips/00000001/results")
        assert saved.status_code == 200
        folium_map = client.get("/api/trips/00000001/map")
        assert folium_map.status_code == 200
        assert "leaflet" in folium_map.text.lower()


def test_upload_requires_three_matching_files(app_settings):
    app = create_app(app_settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/trips/upload",
            files={"track_file": ("trip.track", b"0.0001 0.0001 0\n", "text/plain")},
        )
    assert response.status_code == 422


def test_upload_and_predict_trip_without_ground_truth(app_settings):
    app = create_app(app_settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/trips/upload",
            files={
                "track_file": ("new-trip.track", b"0.0001 0.0001 0\n0.0009 0.0001 1\n", "text/plain"),
                "nodes_file": ("new-trip.nodes", b"0.0 0.0001\n0.0005 0.0001\n0.001 0.0001\n", "text/plain"),
                "arcs_file": ("new-trip.arcs", b"0 1\n1 2\n", "text/plain"),
            },
        )
        assert response.status_code == 201, response.text
        trip_id = response.json()["trip_id"]
        assert trip_id.startswith("upload-")
        assert client.get(f"/api/trips/{trip_id}").status_code == 200
        matched = client.post("/api/map-match", json={"trip_id": trip_id})
        assert matched.status_code == 200, matched.text
        assert (app_settings.uploads_dir / trip_id / f"{trip_id}.route").exists() is False
