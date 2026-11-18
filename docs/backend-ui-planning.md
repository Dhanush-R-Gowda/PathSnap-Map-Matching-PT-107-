
# PathSnap - Backend and UI Design Notes

**Responsible:** Baby (Person 4)  
**Project:** Distributed Map-Matching & Route Reconstruction

> This document preserves the original design plan for context. The current implementation is summarized in Section 10 and specified in `docs/api-specification.md` and `README.md`.

## 1. Objective

Develop a user-friendly web application that accepts GPS trip data, communicates with the ML and HMM components, and displays reconstructed vehicle routes.

## 2. Technology Stack

- Backend: FastAPI (Python)
- Frontend: React
- Map Visualization: Leaflet
- Communication: REST API
- Data Format: JSON
- Deployment: Two laptops on the same network

## 3. Proposed Backend Architecture

```text
backend/
├── main.py
├── routes/
│   ├── health.py
│   ├── upload.py
│   ├── prediction.py
│   └── history.py
├── services/
│   ├── file_service.py
│   ├── prediction_service.py
│   └── route_service.py
├── schemas/
│   └── request_response.py
└── requirements.txt
```

## 4. UI Wireframe

The evaluator-facing dashboard will contain:

1. GPS Trip Upload
2. Interactive Route Map
3. Prediction Results
4. Original GPS vs Reconstructed Route Comparison
5. Previous Trip History

Design: Modern professional dashboard with a clean layout and blue accents.

## 5. Proposed API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Check backend status |
| POST | /api/v1/trips/upload | Upload GPS trip files |
| POST | /api/v1/trips/{trip_id}/predict | Start reconstruction |
| GET | /api/v1/trips/{trip_id}/results | Retrieve results |
| GET | /api/v1/trips | Retrieve trip history |

## 6. API Request and Response

### Upload Request

POST /api/v1/trips/upload

Content-Type: multipart/form-data

Files:
- track_file
- nodes_file
- arcs_file

### Upload Response

```json
{
  "trip_id": "trip_001",
  "status": "uploaded",
  "message": "Trip uploaded successfully"
}
```

### Prediction Result Response

```json
{
  "trip_id": "trip_001",
  "status": "completed",
  "predicted_segments": [101, 102, 103],
  "route_coordinates": [
    [12.9710, 77.5940],
    [12.9720, 77.5950],
    [12.9730, 77.5960]
  ]
}
```

These are proposed examples, not actual prediction results. Final API formats will be agreed upon with teammates.

## 7. Frontend–Backend Communication

1. Evaluator uploads GPS trip files.
2. React sends files to FastAPI.
3. FastAPI validates and processes the request.
4. ML model generates candidate probabilities.
5. HMM decoder reconstructs the most likely route.
6. Backend returns reconstructed route information.
7. Frontend displays the route using Leaflet.

## 8. Two-Laptop Deployment

- Laptop 1: FastAPI backend (Port 8000).
- Laptop 2: React frontend (Port 3000).
- Both laptops communicate through the same Wi-Fi network.
- React accesses FastAPI using Laptop 1's local IP address.

## 9. Integration Dependencies

- Dhanush: Preprocessing and feature-generation interface.
- Chethan: Trained ML model and prediction probabilities.
- Chandra Raj: HMM decoder and reconstructed route output.

## 10. Current Status

The integration is implemented in `backend/main.py`, `backend/services/inference.py`, and `frontend/src/App.jsx`.

- The backend loads the saved model and schema on startup, lists route-free demo trips, validates `.track`/`.nodes`/`.arcs` uploads, runs candidate inference and HMM/Viterbi decoding, and serves result and Folium-map routes.
- The React UI loads trip and model metadata from the API, accepts all three trip files, runs map matching, and displays GPS, candidate-road, classifier, and HMM layers plus route diagnostics.
- The current decoder preserves explicit resets for missing candidates, long time gaps, and disconnected candidate sets. It reports warnings instead of fabricating road connectors.
- Two-laptop commands and CORS settings are documented in the repository `README.md`.

Automated backend and integration coverage is in `tests/test_backend.py` and `tests/test_hmm_integration.py`. A manual interactive browser workflow and two-laptop LAN connectivity still need deployment-machine verification. The local browser automation bridge was unavailable during this review.
