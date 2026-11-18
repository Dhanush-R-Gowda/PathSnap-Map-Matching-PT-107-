# PathSnap evaluator dashboard

React + Vite + Leaflet frontend for the PathSnap FastAPI backend. The dashboard
lets an evaluator connect to a backend, select demo trips or upload
`.track`/`.nodes`/`.arcs` files, run reconstruction and inspect the GPS track,
the reconstructed route and quality metrics on a map. No ground-truth `.route`
file is used or accepted.

## Setup

```bash
npm install
npm run dev            # dev server on http://localhost:5173 (host 0.0.0.0)
npm run build          # production bundle in dist/
npm run lint           # Oxlint
```

## Backend configuration

The backend URL is resolved in this order:

1. `VITE_API_URL` at build time (e.g. `VITE_API_URL=http://192.168.1.5:8000 npm run build`),
2. the URL saved in the browser from the *Backend* input (`localStorage`),
3. `http://<current-host>:8000`.

The backend must be running (`python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`)
and its model artifacts generated. See the root `README.md` and `docs/DEPLOYMENT.md`.

## API used

`GET /health`, `GET /api/v1/samples`,
`POST /api/v1/samples/{partition}/{id}/load`,
`POST /api/v1/trips/upload`, `POST /api/v1/trips/{id}/predict`,
`GET /api/v1/trips/{id}/results`, `GET /api/v1/trips`.
The request/response shapes are defined by `backend/main.py`; loading, error and
empty-result states are handled in `src/App.jsx` and `src/api.js`.
