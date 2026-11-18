
"""FastAPI service for route-free PathSnap map matching."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services.inference import (
    MAX_UPLOAD_BYTES,
    BackendSettings,
    PathSnapInferenceService,
    default_settings,
)


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


def create_app(settings: BackendSettings | None = None) -> FastAPI:
    settings = settings or default_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.service = PathSnapInferenceService(settings)
            application.state.service_error = None
        except Exception as exc:
            # Report a readiness error and keep health queryable.
            application.state.service = None
            application.state.service_error = f"{type(exc).__name__}: {exc}"

        yield

    app = FastAPI(
        title="PathSnap Map Matching API",
        version="1.0.0",
        description="Candidate scoring and directed HMM/Viterbi route reconstruction.",
        lifespan=lifespan,
    )

    cors_origins = [
        origin.strip()
        for origin in os.environ.get(
            "PATHSNAP_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=(
            r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|"
            r"10\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?"
        ),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": "PathSnap API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    def health(request: Request) -> dict:
        instance = getattr(request.app.state, "service", None)
        trips_path = settings.raw_root / settings.demo_partition

        return {
            "status": "healthy" if instance is not None else "degraded",
            "service": "PathSnap API",
            "model_loaded": instance is not None,
            "hmm_available": instance is not None,
            "dataset_available": (
                trips_path.is_dir() or settings.uploads_dir.is_dir()
            ),
            "model_error": getattr(request.app.state, "service_error", None),
        }

    @app.get("/api/model")
    def model_info(
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
    ) -> dict:
        return instance.model_info

    @app.get("/api/trips")
    def list_trips(
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
    ) -> dict:
        return {"trips": instance.list_trips()}

    @app.get("/api/trips/{trip_id}")
    def get_trip(
        trip_id: str,
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
    ) -> dict:
        try:
            return instance.preview_trip(trip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/trips/upload", status_code=201)
    async def upload_trip(
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
        track_file: Annotated[
            UploadFile, File(description="GPS observations in .track format")
        ],
        nodes_file: Annotated[
            UploadFile, File(description="Road nodes in .nodes format")
        ],
        arcs_file: Annotated[
            UploadFile, File(description="Directed road arcs in .arcs format")
        ],
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
                    detail=(
                        f"Expected a {suffix} file for "
                        f"{suffix[1:]}_file"
                    ),
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
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
    ) -> dict:
        try:
            return instance.map_match(payload.trip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/trips/{trip_id}/predict")
    def predict_trip(
        trip_id: str,
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
    ) -> dict:
        try:
            return instance.map_match(trip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/trips/{trip_id}/results")
    def get_results(
        trip_id: str,
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
    ) -> dict:
        if trip_id not in instance.results:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No route result is available for this trip "
                    "in the current server session"
                ),
            )

        return instance.results[trip_id]

    @app.get("/api/trips/{trip_id}/map")
    def get_folium_map(
        trip_id: str,
        instance: Annotated[
            PathSnapInferenceService, Depends(get_service)
        ],
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

    return app


app = create_app()
