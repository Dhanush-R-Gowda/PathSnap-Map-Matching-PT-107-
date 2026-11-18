"""PathSnap FastAPI backend (route-free inference; never needs the .route ground truth).

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.data.loader import discover_trips, load_trip_for_inference
from src.pipeline import PathSnapMatcher

log = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("PATHSNAP_ARTIFACTS", REPO_ROOT / "artifacts"))
SAMPLE_ROOT = Path(os.environ.get("PATHSNAP_SAMPLE_ROOT", REPO_ROOT / "data" / "raw" / "Student folder"))
SAMPLE_PARTITIONS = ("evaluator_test", "student_test", "train")
REQUIRED = ("track", "nodes", "arcs")

STATE: dict = {"matcher": None}
TRIPS: dict[str, dict] = {}  # in-memory: trip_id -> {status, data, result}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    STATE["matcher"] = PathSnapMatcher(ARTIFACT_DIR)  # loaded once, reused per request
    log.info("model loaded from %s in %.2fs", ARTIFACT_DIR, time.perf_counter() - t0)
    yield


app = FastAPI(title="PathSnap API", version="1.0.0",
              description="Map matching: ML candidate probabilities + HMM/Viterbi route reconstruction.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _matcher() -> PathSnapMatcher:
    if STATE["matcher"] is None:
        raise HTTPException(503, "Model not loaded")
    return STATE["matcher"]


def _build_result(trip_id: str, data: dict) -> dict:
    t0 = time.perf_counter()
    track, nodes, arcs = data["track"], data["nodes"], data["arcs"]
    out = _matcher().match(trip_id, track, nodes, arcs)
    graph, res = out["graph"], out["result"]

    # Route geometry, split into continuous segments at breaks / non-adjacent arcs.
    segments, cur, prev = [], [], None
    for a in res.arc_sequence:
        if prev is not None and graph.arc_to[prev] != graph.arc_from[a]:
            segments.append(cur)
            cur = []
        pts = graph.arc_polyline(a)
        if not cur:
            cur.append(pts[0])
        cur.append(pts[1])
        prev = a
    if cur:
        segments.append(cur)
    to_ll = lambda seg: [{"lat": lat, "lon": lon} for lon, lat in seg]

    n_obs = len(track)
    matched = [i for i, a in enumerate(res.obs_arc) if a is not None]
    elapsed = time.perf_counter() - t0

    # Calculate server-side reference accuracy if reference route is available
    # (Reference route is never passed to or seen by the prediction pipeline)
    ref_route = data.get("route")
    if ref_route is None:
        for part in SAMPLE_PARTITIONS:
            rf = SAMPLE_ROOT / part / trip_id / f"{trip_id}.route"
            if rf.is_file():
                try:
                    from src.data.loader import load_route
                    ref_route = load_route(str(SAMPLE_ROOT / part / trip_id))
                    break
                except Exception:
                    pass

    ref_accuracy = None
    ml_top1_acc = None
    if ref_route is not None and len(ref_route) > 0:
        try:
            from src.utils.geometry import build_projection
            from src.data.alignment import align_observations
            proj = build_projection(track)
            _, gt = align_observations(track, nodes, arcs, ref_route, proj)
            if len(gt) == n_obs:
                correct_hmm = sum(1 for p, g in zip(res.obs_arc, gt) if p == g and g is not None)
                ref_accuracy = float(correct_hmm / n_obs)
                correct_ml = sum(1 for m, g in zip(out["ml_top1"], gt) if m == g and g is not None)
                ml_top1_acc = float(correct_ml / n_obs)
        except Exception as exc:
            log.warning("could not compute reference accuracy for %s: %s", trip_id, exc)

    return {
        "trip_id": trip_id,
        "status": "completed",
        "original_gps": [{"lat": lat, "lon": lon} for lon, lat, _ in track],
        "reconstructed_route": [p for seg in map(to_ll, segments) for p in seg],
        "route_segments": [to_ll(s) for s in segments],
        "matched_arcs": [
            {"obs_index": i, "arc_id": res.obs_arc[i],
             "polyline": to_ll(graph.arc_polyline(res.obs_arc[i]))} for i in matched
        ][:: max(1, len(matched) // 400)],
        "metrics": {
            "observations": n_obs,
            "matched_observations": len(matched),
            "unmatched_observations": len(res.unmatched),
            "route_arcs": len(res.arc_sequence),
            "route_distance_m": float(sum(graph.arc_len[a] for a in res.arc_sequence)),
            "route_continuity": out["continuity"],
            "chain_breaks": len(res.breaks),
            "mean_candidates_per_obs": sum(len(c) for c in out["candidates"]) / max(n_obs, 1),
            "processing_time_seconds": elapsed,
            "stage_seconds": out["timing_s"],
            "reference_accuracy": ref_accuracy,
            "ml_top1_accuracy": ml_top1_acc,
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "PathSnap API", "model_loaded": STATE["matcher"] is not None}


def _parse_upload(trip_id: str, files: dict[str, UploadFile]) -> dict:
    missing = [e for e in REQUIRED if e not in files]
    if missing:
        raise HTTPException(422, f"Missing required file(s): {', '.join(missing)}")
    with tempfile.TemporaryDirectory() as tmp:
        trip_dir = Path(tmp) / trip_id  # loader derives file names from the directory name
        trip_dir.mkdir()
        for ext, up in files.items():
            (trip_dir / f"{trip_id}.{ext}").write_bytes(up.file.read())
        try:
            data = load_trip_for_inference(str(trip_dir))
        except Exception as exc:  # loader raises on malformed rows
            raise HTTPException(422, f"Invalid trip data: {exc}") from exc
    if not data["track"] or not data["arcs"] or not data["nodes"]:
        raise HTTPException(422, "Trip files must be non-empty")
    return data


@app.post("/api/v1/trips/upload")
def upload_trip(track: UploadFile = File(...), nodes: UploadFile = File(...),
                arcs: UploadFile = File(...), trip_id: str | None = Form(None)):
    """Upload .track/.nodes/.arcs (no .route needed)."""
    tid = trip_id or Path(track.filename or "").stem or uuid.uuid4().hex[:8]
    tid = "".join(ch for ch in tid if ch.isalnum() or ch in "-_")[:40] or uuid.uuid4().hex[:8]
    data = _parse_upload(tid, {"track": track, "nodes": nodes, "arcs": arcs})
    TRIPS[tid] = {"status": "uploaded", "data": data, "result": None}
    log.info("uploaded trip %s (%d obs, %d arcs)", tid, len(data["track"]), len(data["arcs"]))
    return {"trip_id": tid, "status": "uploaded", "message": "Trip files uploaded successfully"}


@app.post("/api/v1/trips/{trip_id}/predict")
def predict(trip_id: str):
    trip = TRIPS.get(trip_id)
    if trip is None:
        raise HTTPException(404, f"Unknown trip_id {trip_id}")
    try:
        trip["status"] = "processing"
        trip["result"] = _build_result(trip_id, trip["data"])
        trip["status"] = "completed"
    except HTTPException:
        raise
    except Exception as exc:
        trip["status"] = "failed"
        log.exception("reconstruction failed for %s", trip_id)
        raise HTTPException(500, f"Route reconstruction failed: {exc}") from exc
    log.info("trip %s done in %.2fs", trip_id, trip["result"]["metrics"]["processing_time_seconds"])
    return {"trip_id": trip_id, "status": "completed", "message": "Route reconstruction completed"}


@app.get("/api/v1/trips/{trip_id}/results")
def results(trip_id: str):
    trip = TRIPS.get(trip_id)
    if trip is None:
        raise HTTPException(404, f"Unknown trip_id {trip_id}")
    if trip["result"] is None:
        raise HTTPException(409, f"Trip {trip_id} is '{trip['status']}'; run predict first")
    return trip["result"]


@app.get("/api/v1/trips")
def list_trips():
    return [{"trip_id": k, "status": v["status"],
             "observations": len(v["data"]["track"]),
             "route_distance_m": (v["result"] or {}).get("metrics", {}).get("route_distance_m")}
            for k, v in TRIPS.items()]


@app.get("/api/v1/samples")
def list_samples():
    """Demo trips available on the backend machine (track/nodes/arcs only)."""
    out = {}
    for part in SAMPLE_PARTITIONS:
        d = SAMPLE_ROOT / part
        out[part] = discover_trips(str(d)) if d.is_dir() else []
    return out


@app.post("/api/v1/samples/{partition}/{trip_id}/load")
def load_sample(partition: str, trip_id: str):
    if partition not in SAMPLE_PARTITIONS or not trip_id.isdigit():
        raise HTTPException(404, "Unknown sample")
    d = SAMPLE_ROOT / partition / trip_id
    if not d.is_dir():
        raise HTTPException(404, "Unknown sample")
    TRIPS[trip_id] = {"status": "uploaded", "data": load_trip_for_inference(str(d)), "result": None}
    return {"trip_id": trip_id, "status": "uploaded", "message": "Sample trip loaded"}
