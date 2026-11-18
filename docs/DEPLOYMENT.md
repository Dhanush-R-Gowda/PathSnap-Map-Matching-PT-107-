# Two-laptop deployment

| Laptop | Role | Command |
|---|---|---|
| Laptop 1 | Model + HMM + API (FastAPI) | `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000` |
| Laptop 2 | Evaluator UI (React/Vite) | `cd frontend && npm install && npm run dev` |

## Laptop 1 (backend)

1. Install Python deps: `pip install -r requirements.txt`.
2. Generate the model artifacts on Laptop 1 (they are gitignored, so they must be
   produced locally):
   `python -m src.models.run_phase2 --trials 5 --seed 42`.
   This writes `artifacts/best_model.pkl`, `artifacts/feature_schema.json`,
   `artifacts/model_metadata.json`, `artifacts/preprocessing_pipeline.pkl` and
   `artifacts/best_hyperparameters.json`. Copy the whole `artifacts/` directory
   to Laptop 1 if training was done elsewhere, or point `PATHSNAP_ARTIFACTS` at
   its location.
3. Start: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`.
   The model is loaded **once at startup** (see the `model loaded ...` log line).
4. Find its LAN IP (`ipconfig` on Windows / `ipconfig getifaddr en0` on macOS) and allow
   inbound TCP 8000 in the firewall. Both laptops must be on the same network.
5. Check from Laptop 2: `curl http://<LAPTOP1_IP>:8000/health`.

Optional env vars: `PATHSNAP_ARTIFACTS` (artifact dir), `PATHSNAP_SAMPLE_ROOT` (demo trips).

## Laptop 2 (UI)

1. `cd frontend && npm install && npm run dev` (serves on `0.0.0.0:5173`).
2. Open `http://localhost:5173`, type `http://<LAPTOP1_IP>:8000` in the **Backend** box and
   click *Connect*. The URL is remembered in the browser; alternatively build with
   `VITE_API_URL=http://<LAPTOP1_IP>:8000 npm run build`.
3. Either select the evaluator trip's `.track`, `.nodes`, `.arcs` files (the multi-select
   "all 3" input assigns them by extension) and press *Upload & reconstruct*, or pick a trip
   stored on the backend. No `.route` file is needed or accepted.

CORS is open (`*`), so the UI origin does not need to be registered.

## API summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + `model_loaded` |
| POST | `/api/v1/trips/upload` | multipart `track`, `nodes`, `arcs` (+ optional `trip_id`) |
| POST | `/api/v1/trips/{id}/predict` | candidates -> features -> model -> HMM/Viterbi |
| GET | `/api/v1/trips/{id}/results` | GPS, route geometry, metrics |
| GET | `/api/v1/trips` | trips held by the server |
| GET | `/api/v1/samples`, POST `/api/v1/samples/{partition}/{id}/load` | demo trips on the backend (train / student_test only) |

Interactive docs: `http://<LAPTOP1_IP>:8000/docs`.

## Verified here

Backend and UI were exercised on one machine over HTTP (browser -> `:5173` -> `:8000`). A real
two-laptop run still needs to be performed on the day; this requires a second physical machine.
