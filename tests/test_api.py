"""API tests (use real trained artifacts; skipped if none are available)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
TRIP = ROOT / "data/raw/Student folder/student_test/00000052"

pytestmark = pytest.mark.skipif(
    not (ART / "best_model.pkl").exists() or not TRIP.is_dir(),
    reason="trained artifacts or raw data not available",
)


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    with TestClient(app) as c:  # triggers lifespan: model loaded once
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_upload_without_route_and_predict(client):
    files = {e: (f"00000052.{e}", (TRIP / f"00000052.{e}").read_bytes()) for e in ("track", "nodes", "arcs")}
    r = client.post("/api/v1/trips/upload", files=files)
    assert r.status_code == 200, r.text
    tid = r.json()["trip_id"]
    assert client.get(f"/api/v1/trips/{tid}/results").status_code == 409  # not run yet
    assert client.post(f"/api/v1/trips/{tid}/predict").status_code == 200
    out = client.get(f"/api/v1/trips/{tid}/results").json()
    assert out["status"] == "completed"
    assert out["metrics"]["route_continuity"] > 0.95
    assert len(out["original_gps"]) == out["metrics"]["observations"]
    assert out["reconstructed_route"]
    assert not any("label" in k or "ground" in k for k in out)
    assert any(t["trip_id"] == tid for t in client.get("/api/v1/trips").json())


def test_missing_file_and_unknown_trip(client):
    r = client.post("/api/v1/trips/upload", files={"track": ("a.track", b"1 2 3\n")})
    assert r.status_code == 422
    assert client.post("/api/v1/trips/nope/predict").status_code == 404
    assert client.get("/api/v1/trips/nope/results").status_code == 404


def test_invalid_content_rejected(client):
    bad = {e: (f"x.{e}", b"garbage\n") for e in ("track", "nodes", "arcs")}
    assert client.post("/api/v1/trips/upload", files=bad).status_code == 422


def test_cors_headers(client):
    r = client.options("/health", headers={"Origin": "http://192.168.1.5:5173",
                                            "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") in ("*", "http://192.168.1.5:5173")
