# PathSnap API Specification

## 1. Overview

PathSnap is a distributed map-matching and route-reconstruction system that reconstructs realistic vehicle routes from noisy GPS trajectories.

The backend uses FastAPI to receive trip data, coordinate the map-matching pipeline, and return reconstructed routes and evaluation results.

## 2. Technology Stack

- Backend: Python, FastAPI
- Frontend: React
- Map visualization: Leaflet
- Data format: JSON
- Communication: REST API


## 3. API Endpoints

The following endpoints are proposed for the initial backend design.

### 3.1 Health Check

- **Method:** GET
- **Endpoint:** `/health`
- **Purpose:** Check whether the backend server is running.

**Proposed response:**

{
  "status": "healthy",
  "service": "PathSnap API"
}

### 3.2 Upload Trip Files

- **Method:** POST
- **Endpoint:** `/api/v1/trips/upload`
- **Purpose:** Upload the files required to process a trip.

**Proposed input files:**
- `.track` – GPS trajectory observations.
- `.nodes` – Road network nodes.
- `.arcs` – Road network arcs.

The `.route` file is ground-truth data and must not be required for normal evaluator inference.

**Proposed response:**

{
  "trip_id": "trip_001",
  "status": "uploaded",
  "message": "Trip files uploaded successfully"
}

### 3.3 Predict Reconstructed Route

- **Method:** POST
- **Endpoint:** `/api/v1/trips/{trip_id}/predict`
- **Purpose:** Start the map-matching and route-reconstruction pipeline.

**Proposed response:**

{
  "trip_id": "trip_001",
  "status": "processing",
  "message": "Route reconstruction started"
}

### 3.4 Get Trip Results

- **Method:** GET
- **Endpoint:** `/api/v1/trips/{trip_id}/results`
- **Purpose:** Retrieve the reconstructed route and processing results.

**Proposed response:**

{
  "trip_id": "trip_001",
  "status": "completed",
  "reconstructed_route": [],
  "processing_time_seconds": 0.0
}

### 3.5 List Trips

- **Method:** GET
- **Endpoint:** `/api/v1/trips`
- **Purpose:** Retrieve the list of uploaded and processed trips.

## 4. Backend Processing Workflow

After receiving the trip files, the backend will coordinate the following stages:

1. **File Validation:** Check that the required `.track`, `.nodes`, and `.arcs` files are available and valid.
2. **Data Preprocessing:** Prepare GPS observations and road-network information.
3. **Candidate Generation:** Generate possible road arcs for each GPS observation.
4. **Feature Extraction:** Calculate the features required by the trained ML model.
5. **ML Prediction:** Obtain candidate-level prediction scores or probabilities.
6. **HMM Decoding:** Use the prediction scores to reconstruct a connected road route.
7. **Result Preparation:** Prepare the reconstructed route and relevant metrics for the frontend.
8. **Map Visualization:** Display the original GPS trajectory and reconstructed route.

## 5. Training Data vs. Inference Data

The backend must distinguish between training and evaluator inference.

### Training
- Uses training trip data and ground-truth `.route` files.
- Ground-truth information is used for generating labels and evaluating models.

### Evaluator Inference
- Uses only the available `.track`, `.nodes`, and `.arcs` files.
- Must not depend on the ground-truth `.route` file.
- Returns the reconstructed route produced by the pipeline.

## 6. Integration Status

All interfaces are implemented and wired end to end:

- `src.features.build_features.build_trip_features_inference` generates candidates
  and the 14 model features without any ground-truth `.route` file.
- `src.models.phase2.load_model` / `predict_candidate_probabilities` load the
  saved pipeline and return per-observation-normalised candidate probabilities.
- `src.decoder.hmm_matcher.decode` consumes those probabilities and returns the
  matched arcs, connected arc sequence and reported breaks/unmatched observations.
- `src.pipeline.PathSnapMatcher` composes the three stages; `backend/main.py`
  serialises the route as GeoJSON-style `{lat, lon}` points for the Leaflet UI.

## 7. Frontend Response Requirements

The frontend will use backend responses to display the trip-processing status and reconstructed route.

### 7.1 Trip Upload

After a successful upload, the frontend displays:
- Trip ID
- Upload status
- Success or error message

### 7.2 Processing Status

While route reconstruction is running, the frontend displays:
- Processing indicator
- Current trip ID
- Processing status

### 7.3 Reconstructed Route

After successful processing, the frontend should receive enough information to display:

- Original GPS coordinates
- Reconstructed road-route geometry
- Route distance, if available
- Processing time
- Prediction status

**Proposed response structure:**

{
  "trip_id": "trip_001",
  "status": "completed",
  "original_gps": [
    {"lat": 12.9716, "lon": 77.5946}
  ],
  "reconstructed_route": [
    {"lat": 12.9720, "lon": 77.5950}
  ],
  "metrics": {
    "route_distance_m": null,
    "processing_time_seconds": 0.0
  }
}

The actual geometry and metric fields must be finalized after integration with the data-processing and HMM modules.

## 8. Frontend Features

The React dashboard is planned to include:

1. Trip file upload.
2. Upload and processing status.
3. Interactive map using Leaflet.
4. Original GPS trajectory visualization.
5. Reconstructed route visualization.
6. Route result and metric summary.
7. Previous trip history.

## 9. Error Handling

The backend should return meaningful error responses for:

- Missing required files.
- Unsupported file formats.
- Invalid trip data.
- Unknown trip IDs.
- Prediction or reconstruction failures.
- Internal server errors.

The frontend should display understandable error messages rather than raw backend tracebacks.