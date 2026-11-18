"""FastAPI service for route-free PathSnap map matching.

Integrates ML candidate scoring, HMM/Viterbi route reconstruction,
evaluator reference accuracy calculation, and React/Leaflet dashboard APIs.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services.inference import (
    MAX_UPLOAD_BYTES,
    BackendSettings,
    PathSnapInferenceService,
    default_settings,
)
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


class MapMatchRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=80)


def get_service(request: Request) -> PathSnapInferenceService:
    instance = getattr(request.app.state, "service", None)
    if instance is None:
        detail = (
            getattr(request.app.state, "service_error", None)
            or "PathSnap inference service is not initialized"
        )
        raise HTTPException(status_code=503, detail=detail)
    return instance


def _matcher() -> PathSnapMatcher:
    if STATE["matcher"] is None:
        raise HTTPException(503, "Model not loaded")
    return STATE["matcher"]


def _build_result(trip_id: str, data: dict, matcher: PathSnapMatcher | None = None) -> dict:
    t0 = time.perf_counter()
    m_instance = matcher or _matcher()
    track, nodes, arcs = data["track"], data["nodes"], data["arcs"]
    out = m_instance.match(trip_id, track, nodes, arcs)
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


def create_app(settings: BackendSettings | None = None) -> FastAPI:
    settings = settings or default_settings()
    artifact_dir = settings.artifact_dir if settings else ARTIFACT_DIR

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # 1. Initialize PathSnapMatcher (for /api/v1 routes)
        t0 = time.perf_counter()
        try:
            STATE["matcher"] = PathSnapMatcher(artifact_dir)
            log.info("PathSnapMatcher loaded from %s in %.2fs", artifact_dir, time.perf_counter() - t0)
        except Exception as exc:
            STATE["matcher"] = None
            log.warning("PathSnapMatcher not loaded: %s", exc)

        # 2. Initialize PathSnapInferenceService (for /api/model, /api/trips routes)
        try:
            application.state.service = PathSnapInferenceService(settings)
            application.state.service_error = None
        except Exception as exc:
            application.state.service = None
            application.state.service_error = f"{type(exc).__name__}: {exc}"
            log.warning("PathSnapInferenceService error: %s", exc)

        yield

    app = FastAPI(
        title="PathSnap API",
        version="1.0.0",
        description="Candidate scoring and directed HMM/Viterbi route reconstruction.",
        lifespan=lifespan,
    )

    cors_origins = [
        origin.strip()
        for origin in os.environ.get(
            "PATHSNAP_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,*",
        ).split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=(
            r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|"
            r"10\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?"
        ),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Root & Health Endpoints
    # ------------------------------------------------------------------
    @app.get("/")
    def root() -> dict:
        return {
            "service": "PathSnap API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health(request: Request) -> dict:
        service_instance = getattr(request.app.state, "service", None)
        matcher_instance = STATE.get("matcher")
        model_loaded = matcher_instance is not None or service_instance is not None
        trips_path = settings.raw_root / settings.demo_partition

        return {
            "status": "healthy" if model_loaded else "degraded",
            "service": "PathSnap API",
            "model_loaded": model_loaded,
            "hmm_available": model_loaded,
            "dataset_available": (trips_path.is_dir() or settings.uploads_dir.is_dir()),
            "model_error": getattr(request.app.state, "service_error", None),
        }

    # ------------------------------------------------------------------
    # /api/ Endpoints (Team / Codex Service API)
    # ------------------------------------------------------------------
    @app.get("/api/model")
    def model_info(
        instance: Annotated[PathSnapInferenceService, Depends(get_service)],
    ) -> dict:
        return instance.model_info

    @app.get("/api/trips")
    def service_list_trips(
        instance: Annotated[PathSnapInferenceService, Depends(get_service)],
    ) -> dict:
        return {"trips": instance.list_trips()}

    @app.get("/api/trips/{trip_id}")
    def get_trip(
        trip_id: str,
        instance: Annotated[PathSnapInferenceService, Depends(get_service)],
    ) -> dict:
        try:
            return instance.preview_trip(trip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/trips/upload", status_code=201)
    async def service_upload_trip(
        instance: Annotated[PathSnapInferenceService, Depends(get_service)],
        track_file: Annotated[UploadFile, File(description="GPS observations in .track format")],
        nodes_file: Annotated[UploadFile, File(description="Road nodes in .nodes format")],
        arcs_file: Annotated[UploadFile, File(description="Directed road arcs in .arcs format")],
    ) -> dict:
        uploads = {
            ".track": track_file,
            ".nodes": nodes_file,
            ".arcs": arcs_file,
        }

        contents: dict[str, bytes] = {}
        for suffix, upload in uploads.items():
            filename = (upload.filename or "").lower()
            if not filename.endswith(suffix):
                raise HTTPException(
                    status_code=422,
                    detail=f"Expected a {suffix} file for {suffix[1:]}_file",
                )
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{suffix} file exceeds the 25 MiB upload limit",
                )
            contents[suffix] = data

        trip_id = f"upload-{uuid.uuid4().hex[:12]}"
        try:
            instance.save_uploaded_trip(trip_id, contents)
            preview = instance.preview_trip(trip_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "trip_id": trip_id,
            "status": "uploaded",
            "message": "Trip files validated and stored",
            **preview,
        }

    @app.post("/api/map-match")
    def map_match(
        payload: MapMatchRequest,
        instance: Annotated[PathSnapInferenceService, Depends(get_service)],
    ) -> dict:
        try:
            return instance.map_match(payload.trip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/trips/{trip_id}/results")
    def get_service_results(
        trip_id: str,
        request: Request,
    ) -> dict:
        instance = getattr(request.app.state, "service", None)
        if instance is not None and trip_id in instance.results:
            return instance.results[trip_id]
        trip = TRIPS.get(trip_id)
        if trip is not None and trip.get("result") is not None:
            return trip["result"]
        raise HTTPException(
            status_code=404,
            detail="No route result is available for this trip in the current server session",
        )

    @app.get("/api/trips/{trip_id}/map")
    def get_folium_map(
        trip_id: str,
        instance: Annotated[PathSnapInferenceService, Depends(get_service)],
    ) -> FileResponse:
        try:
            path = instance.folium_map_path(trip_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if path is None:
            raise HTTPException(
                status_code=404,
                detail="Run map matching before opening the Folium map",
            )

        return FileResponse(
            path,
            media_type="text/html",
            filename=path.name,
        )

    # ------------------------------------------------------------------
    # /api/v1/ Endpoints (Feature Branch & Evaluator Dashboard API)
    # ------------------------------------------------------------------
    @app.post("/api/v1/trips/upload")
    def upload_trip_v1(
        track: UploadFile = File(...),
        nodes: UploadFile = File(...),
        arcs: UploadFile = File(...),
        trip_id: str | None = Form(None),
    ):
        """Upload .track/.nodes/.arcs (no .route needed)."""
        tid = trip_id or Path(track.filename or "").stem or uuid.uuid4().hex[:8]
        tid = "".join(ch for ch in tid if ch.isalnum() or ch in "-_")[:40] or uuid.uuid4().hex[:8]
        data = _parse_upload(tid, {"track": track, "nodes": nodes, "arcs": arcs})
        TRIPS[tid] = {"status": "uploaded", "data": data, "result": None}
        log.info("uploaded trip %s (%d obs, %d arcs)", tid, len(data["track"]), len(data["arcs"]))
        return {"trip_id": tid, "status": "uploaded", "message": "Trip files uploaded successfully"}

    @app.post("/api/v1/trips/{trip_id}/predict")
    def predict_v1(trip_id: str, request: Request):
        trip = TRIPS.get(trip_id)
        if trip is not None:
            try:
                trip["status"] = "processing"
                trip["result"] = _build_result(trip_id, trip["data"])
                trip["status"] = "completed"
                log.info("trip %s done in %.2fs", trip_id, trip["result"]["metrics"]["processing_time_seconds"])
                return {"trip_id": trip_id, "status": "completed", "message": "Route reconstruction completed"}
            except HTTPException:
                raise
            except Exception as exc:
                trip["status"] = "failed"
                log.exception("reconstruction failed for %s", trip_id)
                raise HTTPException(500, f"Route reconstruction failed: {exc}") from exc

        # Fallback to inference service if trip exists there
        service = getattr(request.app.state, "service", None)
        if service is not None:
            try:
                return service.map_match(trip_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        raise HTTPException(404, f"Unknown trip_id {trip_id}")

    @app.get("/api/v1/trips/{trip_id}/results")
    def results_v1(trip_id: str, request: Request):
        trip = TRIPS.get(trip_id)
        if trip is not None:
            if trip["result"] is None:
                raise HTTPException(409, f"Trip {trip_id} is '{trip['status']}'; run predict first")
            return trip["result"]
        service = getattr(request.app.state, "service", None)
        if service is not None and trip_id in service.results:
            return service.results[trip_id]
        raise HTTPException(404, f"Unknown trip_id {trip_id}")

    @app.get("/api/v1/trips")
    def list_trips_v1():
        return [
            {
                "trip_id": k,
                "status": v["status"],
                "observations": len(v["data"]["track"]),
                "route_distance_m": (v["result"] or {}).get("metrics", {}).get("route_distance_m"),
            }
            for k, v in TRIPS.items()
        ]

    @app.get("/api/v1/samples")
    def list_samples_v1():
        """Demo trips available on the backend machine (track/nodes/arcs only)."""
        out = {}
        for part in SAMPLE_PARTITIONS:
            d = SAMPLE_ROOT / part
            out[part] = discover_trips(str(d)) if d.is_dir() else []
        return out

    @app.post("/api/v1/samples/{partition}/{trip_id}/load")
    def load_sample_v1(partition: str, trip_id: str):
        if partition not in SAMPLE_PARTITIONS or not trip_id.isdigit():
            raise HTTPException(404, "Unknown sample")
        d = SAMPLE_ROOT / partition / trip_id
        if not d.is_dir():
            raise HTTPException(404, "Unknown sample")
        TRIPS[trip_id] = {"status": "uploaded", "data": load_trip_for_inference(str(d)), "result": None}
        return {"trip_id": trip_id, "status": "uploaded", "message": "Sample trip loaded"}

    return app


app = create_app()
