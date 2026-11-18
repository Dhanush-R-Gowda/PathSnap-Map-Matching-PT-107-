# PathSnap: Distributed Map-Matching & Route Reconstruction (PT-107)

PathSnap reconstructs a coherent vehicle route from noisy GPS observations by
matching each observation to directed road-network arcs. It is a four-stage
pipeline: Phase 1 builds candidate arcs and features, Phase 2 scores candidates
with a supervised classifier, Phase 3 reconstructs a route with an HMM/Viterbi
decoder, and Phase 4 exposes the result through a FastAPI backend and a React
dashboard.

## 1. Project overview

The unit of prediction is a **candidate**: one GPS observation paired with one
road arc within a 50 m search radius. A classifier estimates P(candidate is the
travelled arc), and the HMM uses the directed road network to turn those local
scores into a connected route. Ground-truth `.route` files are used **only** for
training labels and offline evaluation — inference (API/UI) never reads them.

## 2. System architecture

```mermaid
flowchart LR
  U[React + Leaflet evaluator UI] -->|REST / JSON| A[FastAPI]
  A --> D[Route-free Phase 1 candidate generation]
  D --> F[14-feature schema]
  F --> M[Saved Random Forest pipeline]
  M --> P[Candidate score normalization]
  P --> H[Directed HMM + Viterbi]
  H --> R[Classifier and decoded route results]
  R --> L[Leaflet map and Folium HTML map]
```

```text
.track/.nodes/.arcs (+ .route for training)
        │
        ▼
Phase 1  src/data/*, src/features/*, src/utils/geometry.py
         KD-tree candidate generation, point-to-segment projection,
         banded dynamic-programming alignment, 14 features, miss-rate report
        │  data/processed/{train,student_test}_candidates.parquet
        ▼
Phase 2  src/models/phase2.py + run_phase2.py
         5 models, trip-grouped 5-fold CV, Optuna TPE tuning, final fit,
         per-observation probability normalisation
        │  artifacts/{best_model.pkl,feature_schema.json,...}
        ▼
Phase 3  src/decoder/graph.py + hmm_matcher.py, src/pipeline.py
         directed graph + bounded Dijkstra transitions, log-space Viterbi,
         chain-break handling, route polyline assembly
        │
        ▼
Phase 4  backend/main.py (FastAPI) + frontend/ (React/Vite/Leaflet)
         upload/load demo trip → predict → visualise GPS + reconstructed route
```

`src/pipeline.py::PathSnapMatcher` is the integration seam: it loads the model
and feature schema once, generates route-free features, predicts candidate
probabilities and decodes the route. The backend and the offline evaluation
script both use it.

**Canonical implementations.** The active pipeline is the consolidated one
described above. The repository also contains an earlier parallel HMM package
(`hmm/`) and an earlier split-module Phase 2 implementation that lived on the
`feature/data-features` branch (`data_prep.py`, `train_models.py`,
`evaluation.py`, `bayes_opt.py`). Their functionality is fully represented in
`src/decoder/` and `src/models/phase2.py`; they are not imported by the running
application and are retained only for provenance. See the final integration
report for how the divergent branches were reconciled.

## 3. Technology stack

- Python 3.10+ (developed and verified on Python 3.12 and 3.14)
- NumPy, pandas, PyArrow, SciPy
- scikit-learn, LightGBM, Optuna, joblib
- FastAPI, Uvicorn, httpx
- React 19, Vite, Leaflet / react-leaflet, Oxlint
- pytest, matplotlib, folium

Pinned compatible ranges are in `requirements.txt`.

## 4. Repository structure

| Path | Contents |
|---|---|
| `src/data/` | Loaders, spatial-index candidate generation, DP alignment, miss-rate analysis |
| `src/features/` | Training and route-free inference feature builders |
| `src/utils/geometry.py` | Projection, point-to-segment distance, bearing, arc length |
| `src/models/phase2.py` | Models, CV, metrics, optimisation, inference and artifact helpers |
| `src/models/run_phase2.py` | Full training/comparison/tuning/evaluation runner |
| `src/models/resume_phase2.py` | Resume from an existing baseline comparison |
| `src/decoder/` | Directed road graph and HMM/Viterbi decoder |
| `src/pipeline.py` | End-to-end route-free matcher used by backend and scripts |
| `backend/` | FastAPI application (`main.py` and `services/inference.py`) |
| `frontend/` | Evaluator dashboard (React + Vite + Leaflet) |
| `hmm/` | Earlier HMM implementation and integrated decoder |
| `scripts/` | Offline pipeline evaluation and reconstruction-map generation |
| `data/raw/`, `data/processed/` | Input trips and generated candidate Parquet files |
| `artifacts/` | Trained model, preprocessing, schema and metadata (generated) |
| `results/` | Metrics, plots, fold details, probability exports (generated) |
| `report/` | Technical report and PDF build scripts |
| `docs/` | Dataset schema, ML implementation, API spec, deployment guide |
| `tests/` | Phase 1, Phase 2, HMM, inference, backend, and API tests |

## 5. Dataset requirements

Each trip directory contains four space-separated text files named
`<trip_id>.track` (lon lat timestamp), `.nodes` (lon lat), `.arcs`
(from_node to_node, zero-based; the row index is the trip-local `arc_id`) and
`.route` (ground-truth arc IDs, training/evaluation only).

The supplied dataset is expected under `data/raw/`. Unpack the course-supplied
`Student folder.zip` so that:

```text
data/raw/Student folder/
  train/<8-digit-trip-id>/<id>.track, <id>.nodes, <id>.arcs, <id>.route
  student_test/<8-digit-trip-id>/<id>.track, <id>.nodes, <id>.arcs, <id>.route
  evaluator_test/<8-digit-trip-id>/<id>.track, <id>.nodes, <id>.arcs, <id>.route
```

The demo UI lists trips but reads only `.track`, `.nodes`, and `.arcs` for
inference. Training and test labels remain outside the inference response.

## 6. Installation and environment setup

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Environment variables (all optional):

| Variable | Purpose | Default |
|---|---|---|
| `PATHSNAP_ARTIFACTS` | Artifact directory used by backend/scripts | `<repo>/artifacts` |
| `PATHSNAP_SAMPLE_ROOT` | Root of demo trips | `<repo>/data/raw/Student folder` |
| `PATHSNAP_CORS_ORIGINS` | Comma-separated CORS origin allowlist | `*` (open for LAN/dev) |
| `VITE_API_URL` / `VITE_API_BASE_URL` | Backend URL baked into frontend build | current host on port 8000 |

## 7. Data processing pipeline (Phase 1)

From the repository root:

```bash
python -m src.features.build_features --partition both --radius 50
```

This writes `data/processed/train_candidates.parquet` and
`student_test_candidates.parquet` plus `results/experiments/candidate_miss_rates.csv`.
The pipeline uses a KD-tree over arc midpoints, exact point-to-segment
projection, a default DP alignment band of 100 (adaptively expanded after GPS
gaps ≥ 5 s), and emits 14 numeric features:

`snap_distance_m, snap_t, arc_length_m, bearing_diff_rad,
abs_bearing_diff_rad, gps_speed_mps, gps_heading_rad, dt_prev_s, dt_next_s,
out_degree, in_degree, is_gap_prev, is_gap_next, heading_reliable`.

Measured from candidate data: 1,870,717 training rows / 80 trips, 209,756
student-test rows / 10 trips, a 6.13% training candidate miss rate (11,654 /
190,188 observations represented in the miss-rate report) and a 0.14%
student-test miss rate.

## 8. Model training and hyperparameter optimisation (Phase 2)

```bash
python -m src.models.run_phase2 --trials 5 --seed 42
```

The runner validates the Parquet schema, applies observation-level filtering
(keep every candidate of an observation whose ground truth is present), builds
five fixed `GroupKFold` splits over **trips**, compares LightGBM, RandomForest,
LogisticRegression, a shallow MLP and Gaussian Naive Bayes, selects a model by a
weighted rank of accuracy/F1/continuity/log-loss/stability/time, tunes it with
Optuna `TPESampler` on mean five-fold candidate log loss, retrains the final
model on all eligible training rows, evaluates on student test, and saves
the artifacts and reports. `--trials N` controls the search budget (default 5).
All learned preprocessing (median imputation; StandardScaler for LR/MLP) lives
inside the saved sklearn pipeline, so training and inference are consistent.

Exact measured metrics, selected model and hyperparameters are written to
`docs/PHASE2_ML_IMPLEMENTATION.md`, `results/model_comparison.json`,
`results/best_hyperparameters.json` and `artifacts/model_metadata.json`.

## 9. HMM and Viterbi decoding (Phase 3)

`src/decoder/graph.py` builds a directed graph from each trip's `.arcs`, with
metric arc lengths and bounded Dijkstra shortest paths. Emissions are the
per-observation-normalised classifier probabilities. Transitions follow the
Newson–Krumm form `exp(-|d_net - d_gps| / beta)`, row-normalised over reachable
candidates; unreachable transitions are `-inf`. Decoding is in log space with
Viterbi backtracking; observations with no candidates are recorded as
unmatched, and disconnected jumps restart the chain and are reported as
`breaks` rather than silently bridged. `src/pipeline.py` converts the decoded
arc sequence into route polylines for the UI.

## 10. Backend startup

Generate artifacts first (section 8), then:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The model is loaded once at startup; confirm with `GET /health`.

## 11. Frontend startup

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

Enter the backend URL in the dashboard's *Backend* box (or set `VITE_API_URL` /
`VITE_API_BASE_URL`). No `.route` file is sent or needed.

## 12. Two-laptop deployment

Both laptops must be on the same trusted Wi-Fi/LAN.

**Laptop 1 — backend**

1. Find its LAN IPv4 address (e.g. `ipconfig` on Windows or `ifconfig` on macOS).
2. Start the API on all interfaces:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

3. Ensure port 8000 is open in the local firewall.

**Laptop 2 — frontend**

1. In `frontend/.env.local` or the UI connection box, set the backend address:

```text
VITE_API_BASE_URL=http://192.168.1.20:8000
```

2. Start Vite on the LAN interface:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

3. Open the frontend laptop's LAN URL shown by Vite (`http://<ip>:5173`).

## 13. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API/model/HMM readiness |
| `GET` | `/api/model` | Model metadata, feature schema, hyperparameters |
| `GET` | `/api/trips` | Available demo and uploaded trips |
| `GET` | `/api/trips/{trip_id}` | Trip preview (GPS observations and counts) |
| `POST` | `/api/trips/upload` | Multipart upload (`track_file`, `nodes_file`, `arcs_file`) |
| `POST` | `/api/map-match` | JSON map-match request (`{"trip_id": ...}`) |
| `GET` | `/api/trips/{trip_id}/results` | Return latest route result |
| `GET` | `/api/trips/{trip_id}/map` | Return generated Folium HTML map |
| `POST` | `/api/v1/trips/upload` | Multipart upload (`track`, `nodes`, `arcs`) |
| `POST` | `/api/v1/trips/{trip_id}/predict` | Path-based prediction endpoint |
| `GET` | `/api/v1/trips/{trip_id}/results` | Return completed route reconstruction and metrics |
| `GET` | `/api/v1/trips` | Trips held by server session |
| `GET` | `/api/v1/samples` | List sample trips across partitions |
| `POST` | `/api/v1/samples/{partition}/{trip_id}/load` | Load sample trip into session |

Interactive Swagger documentation: `http://localhost:8000/docs`. Full
contract in `docs/api-specification.md`; deployment guide in
`docs/DEPLOYMENT.md`.

## 14. Testing

```bash
# Full test suite
PYTHONPATH=. .venv/bin/pytest tests/ -v

# Frontend lint and production build
cd frontend && npm run lint && npm run build
```

The test suite covers:
- Phase 1 data loading, geometry, KD-tree candidates, DP alignment
- Phase 2 ML models, GroupKFold CV, Optuna optimization, artifact persistence
- Phase 3 HMM graph, bounded Dijkstra, Newson-Krumm transition, Viterbi decoder
- Backend API endpoints, CORS, route-free inference, Folium map generation
- Route-free inference pipeline without `.route` files

## 15. Expected artifacts

Generated by `run_phase2.py`:

- `artifacts/best_model.pkl` — fitted preprocessing + classifier pipeline
- `artifacts/preprocessing_pipeline.pkl` — preprocessing step alone
- `artifacts/feature_schema.json` — ordered 14-feature schema
- `artifacts/model_metadata.json` — selection, CV, test metrics, versions
- `artifacts/best_hyperparameters.json` — selected hyperparameters
- `results/prediction_probabilities/student_test_candidate_probabilities.{parquet,json}`
- `results/{dataset_schema,cv_results,model_comparison,optimization_trials,tuned_vs_untuned,best_hyperparameters}.*`
- `results/plots/*.png`

## 16. Known limitations

- Candidate misses (ground-truth arc outside the 50 m radius) cannot be fixed
  downstream; they are reported separately from classifier accuracy.
- Trip-local arc IDs are scoped to their trip and must never be merged across
  trips.
- Normalisation produces a per-observation distribution, not a calibrated
  probability model under distribution shift.
- Only 10 student-test trips are supplied, so their metrics are descriptive;
  trip-grouped CV is the reliable estimate.
- The decoder does not invent intermediate roads or bridge disconnected
  networks; disconnected gaps produce chain breaks and restart the Viterbi chain.
