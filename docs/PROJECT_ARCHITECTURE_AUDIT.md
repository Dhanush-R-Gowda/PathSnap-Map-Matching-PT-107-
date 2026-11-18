# PathSnap Architecture Audit

**Audit date:** 2026-11-18
**Repository:** `Dhanush-R-Gowda/PathSnap-Map-Matching-PT-107-`
**Audited ref:** local `codex/model-metric-rankings` at `973fda5`, with `origin/main` at `59d381d` after the metric-ranking PR merge. The working tree was clean at audit start.

## Existing implementation

| Area | Status | Evidence |
|---|---|---|
| Phase 1 data loading | Implemented | `src/data/loader.py` parses `.track`, `.nodes`, `.arcs`, `.route`; raw training and student-test data are present locally. |
| Candidate generation and features | Implemented for labelled training data | `src/data/candidates.py` and `src/features/build_features.py` generate candidate rows and the 14 model features. The current single-trip builder always loads `.route`, so evaluator inference has no route-free entry point. |
| Trained model and schema | Present locally | `artifacts/best_model.pkl`, preprocessing pipeline, schema, metadata, and hyperparameters load from the workspace. The model files are ignored/untracked, so a clean clone does not receive them. |
| Model inference utility | Implemented | `src/models/phase2.py` predicts candidate scores and normalizes within `(trip_id, obs_index)`. It accepts feature rows without target columns. |
| HMM/Viterbi | Partial, standalone modules | `hmm/` contains emission, transition, route geometry, and Viterbi examples. No service connects candidate probabilities to a route result. Some imports are script-relative and not package-safe. Viterbi rejects empty observation candidate sets. |
| FastAPI backend | Missing | `backend/` contains only `.gitkeep`; there are no app modules, schemas, services, or API tests. |
| React evaluator UI | Prototype only | `frontend/src/App.jsx` uses hardcoded trip rows, sample metrics, placeholder coordinates, and a local-only upload state. Leaflet is installed in `package.json`, but the map is not connected to the dataset or API. |
| Folium maps | Partial | `src/features/make_maps.py` creates training-data validation and network maps. It does not visualize model predictions or an HMM result. |
| Tests | Partial | Phase 1 and Phase 2 test modules exist. No backend, API, HMM integration, frontend build, or system workflow checks exist. The existing test suite previously reported 36 passes before this audit. |
| Documentation/report | Partial | Phase 1/2 and API planning docs exist. `README.md` has basic setup and ML workflow instructions; `report/` has no consolidated technical report or PDF. |
| Deployment/configuration | Missing | No `.env.example`, backend/frontend connection settings, CORS setup, or two-laptop instructions are defined. |

## Gaps to address

1. Add route-free feature generation and validate uploaded `.track`, `.nodes`, and `.arcs` inputs.
2. Load the existing model and schema once at backend startup; surface missing or incompatible artifacts clearly.
3. Connect candidate probabilities to a directed road-network transition model and segmented Viterbi decoder, with explicit gap/no-path warnings.
4. Implement FastAPI trip listing, trip metadata, upload, map matching, cached results, health, metadata, and Folium-map routes.
5. Replace placeholder React data with API-backed trip selection, upload, processing, result metrics, errors, and route layers.
6. Generate a Folium reconstruction map from actual GPS, candidate-road, classifier, and HMM output layers.
7. Add backend and integration coverage, verify the frontend build, and document exact local and LAN startup commands.
8. Provide the consolidated report with actual experiment results and explicit deployment/testing limitations.

## Data and integrity constraints

- Use the fitted Phase 2 model and exact ordered feature schema. Do not retrain as an API startup action.
- Inference must read `.track`, `.nodes`, and `.arcs` only. The `.route` file is not required or read by the inference builder.
- The UI and inference API must not return route labels or training ground truth.
- Model artifacts are available in this workspace but are not tracked by the repository; clean-clone startup therefore requires an explicit artifact provisioning step.
- Existing raw and generated local data are ignored or locally provisioned; keep them out of commits.
