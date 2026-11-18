import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CircleMarker,
  LayerGroup,
  LayersControl,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const EMPTY_COORDINATES = [];

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.message || `Request failed (${response.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function percentage(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function distance(value) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${value.toFixed(0)} m`;
}

function MapBounds({ coordinates, resetKey }) {
  const map = useMap();
  useEffect(() => {
    if (coordinates.length > 1) map.fitBounds(coordinates, { padding: [24, 24] });
    else if (coordinates.length === 1) map.setView(coordinates[0], 15);
  }, [map, resetKey, coordinates]);
  return null;
}

function RouteMap({ preview, result }) {
  const gps = preview?.gps_coordinates || result?.gps_coordinates || EMPTY_COORDINATES;
  const gpsPositions = useMemo(() => gps.map((point) => [point.lat, point.lon]), [gps]);
  const center = gpsPositions[0];
  const routePoints = useMemo(() => [
    ...(result?.classifier_route?.segments || []).flat(),
    ...(result?.hmm_route?.segments || []).flat(),
  ], [result]);
  const bounds = useMemo(() => [...gpsPositions, ...routePoints], [gpsPositions, routePoints]);

  if (!gpsPositions.length) return <div className="map-frame map-empty">Select a trip to preview its GPS trace.</div>;

  return (
    <div className="map-frame">
      <MapContainer center={center} zoom={13} scrollWheelZoom className="route-map">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapBounds coordinates={bounds} resetKey={`${preview?.trip_id || ""}-${result?.status || "preview"}`} />
        <LayersControl position="topright">
          <LayersControl.Overlay checked name="GPS observations">
            <LayerGroup>
              {gpsPositions.length > 1 && <Polyline positions={gpsPositions} pathOptions={{ color: "#3978f6", weight: 3, opacity: 0.8, dashArray: "7 5" }} />}
              {gpsPositions.filter((_, index) => index % Math.max(1, Math.ceil(gpsPositions.length / 250)) === 0).map((point, index) => (
                <CircleMarker key={`gps-${index}`} center={point} radius={3} pathOptions={{ color: "#3978f6", fillColor: "#ffffff", fillOpacity: 0.9 }}>
                  <Popup>GPS observation</Popup>
                </CircleMarker>
              ))}
            </LayerGroup>
          </LayersControl.Overlay>
          {result && (
            <LayersControl.Overlay name="Candidate road arcs">
              <LayerGroup>
                {(result.candidate_road_segments || []).map((segment) => (
                  <Polyline key={`candidate-${segment.arc_id}`} positions={segment.coordinates} pathOptions={{ color: "#7d8899", weight: 2, opacity: 0.38 }} />
                ))}
              </LayerGroup>
            </LayersControl.Overlay>
          )}
          {result && (
            <LayersControl.Overlay name="Classifier top-1" checked>
              <LayerGroup>
                {(result.classifier_route?.segments || []).map((line, index) => (
                  <Polyline key={`classifier-${index}`} positions={line} pathOptions={{ color: "#ed8b2c", weight: 5, opacity: 0.9 }} />
                ))}
              </LayerGroup>
            </LayersControl.Overlay>
          )}
          {result && (
            <LayersControl.Overlay name="HMM / Viterbi route" checked>
              <LayerGroup>
                {(result.hmm_route?.segments || []).map((line, index) => (
                  <Polyline key={`hmm-${index}`} positions={line} pathOptions={{ color: "#148c67", weight: 6, opacity: 0.95 }} />
                ))}
              </LayerGroup>
            </LayersControl.Overlay>
          )}
        </LayersControl>
      </MapContainer>
    </div>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [trips, setTrips] = useState([]);
  const [selectedTripId, setSelectedTripId] = useState("");
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loadingTrips, setLoadingTrips] = useState(true);
  const [matching, setMatching] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadFiles, setUploadFiles] = useState({});
  const [apiError, setApiError] = useState("");
  const [actionError, setActionError] = useState("");
  const [active, setActive] = useState("Dashboard");
  const previewRequestId = useRef(0);

  const selectTrip = useCallback((tripId) => {
    const requestId = ++previewRequestId.current;
    setSelectedTripId(tripId);
    setResult(null);
    setActionError("");
    setPreview(null);
    if (!tripId) return;
    request(`/api/trips/${encodeURIComponent(tripId)}`)
      .then((payload) => { if (requestId === previewRequestId.current) setPreview(payload); })
      .catch((error) => { if (requestId === previewRequestId.current) setActionError(error.message); });
  }, []);

  const loadTrips = useCallback(async (selectFirst = false) => {
    setLoadingTrips(true);
    setApiError("");
    try {
      const healthData = await request("/health");
      setHealth(healthData);
      const tripData = await request("/api/trips");
      setTrips(tripData.trips || []);
      request("/api/model").then(setModelInfo).catch(() => setModelInfo(null));
      if (selectFirst && tripData.trips?.length) selectTrip(tripData.trips[0].trip_id);
    } catch (error) {
      setApiError(error.message);
      setHealth((current) => current ? { ...current, status: "unavailable", model_loaded: false } : null);
      setModelInfo(null);
      setTrips([]);
    } finally {
      setLoadingTrips(false);
    }
  }, [selectTrip]);

  // Start the initial API load; loadTrips synchronizes React state with the backend response.
  // oxlint-disable-next-line react/set-state-in-effect
  useEffect(() => { loadTrips(true); }, [loadTrips]);

  const activePreview = preview?.trip_id === selectedTripId ? preview : null;
  const activeResult = result?.trip_id === selectedTripId ? result : null;
  const loadingPreview = Boolean(selectedTripId && !activePreview && !actionError);

  const onFilesSelected = (event) => {
    const next = {};
    for (const file of Array.from(event.target.files || [])) {
      const name = file.name.toLowerCase();
      if (name.endsWith(".track")) next.track_file = file;
      if (name.endsWith(".nodes")) next.nodes_file = file;
      if (name.endsWith(".arcs")) next.arcs_file = file;
    }
    setUploadFiles(next);
    setActionError("");
  };

  const uploadTrip = async () => {
    if (!uploadFiles.track_file || !uploadFiles.nodes_file || !uploadFiles.arcs_file) {
      setActionError("Choose one .track, one .nodes, and one .arcs file together.");
      return;
    }
    setUploading(true);
    setActionError("");
    try {
      const form = new FormData();
      Object.entries(uploadFiles).forEach(([key, file]) => form.append(key, file));
      const uploaded = await request("/api/trips/upload", { method: "POST", body: form });
      await loadTrips(false);
      selectTrip(uploaded.trip_id);
      setUploadFiles({});
    } catch (error) {
      setActionError(error.message);
    } finally {
      setUploading(false);
    }
  };

  const runMatching = async () => {
    if (!selectedTripId) return;
    setMatching(true);
    setActionError("");
    setResult(null);
    try {
      const output = await request("/api/map-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: selectedTripId }),
      });
      setResult(output);
      setHealth((current) => current || { status: "healthy", model_loaded: true });
    } catch (error) {
      setActionError(error.message);
    } finally {
      setMatching(false);
    }
  };

  const overview = useMemo(() => ({
    total: trips.length,
    uploaded: trips.filter((trip) => trip.uploaded).length,
    observations: activePreview?.observation_count ?? 0,
  }), [trips, activePreview]);

  const navigate = (name) => {
    setActive(name);
    const target = document.getElementById(name === "Upload Trip" ? "upload" : name === "Trip History" ? "history" : "workspace");
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup"><div className="brand-mark">P</div><div><strong>PathSnap</strong><span>MAP MATCHING</span></div></div>
        <p className="sidebar-label">WORKSPACE</p>
        <nav className="side-nav" aria-label="Main navigation">
          {["Dashboard", "Upload Trip", "Route Map", "Trip History"].map((name, index) => (
            <button key={name} className={active === name ? "nav-link active" : "nav-link"} onClick={() => navigate(name)}>
              <span className="nav-glyph">{["◫", "↑", "⌖", "▤"][index]}</span>{name}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className={`status-indicator ${health?.model_loaded ? "online" : "offline"}`} />
          <span>{health?.model_loaded ? "Model service ready" : "Checking service"}</span>
          <small>PathSnap · distributed route reconstruction</small>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div><span className="muted">Workspace / </span><strong>{active}</strong></div><div className="topbar-tag">Evaluator workspace</div></header>
        <div className="page-content">
          <section className="hero-section">
            <div><p className="eyebrow">DISTRIBUTED MAP MATCHING & ROUTE RECONSTRUCTION</p><h1>Turn GPS traces into road routes.</h1><p className="hero-copy">Compare candidate scoring with a globally decoded, road-connected route.</p></div>
            <div className="service-pill"><span className={`status-indicator ${health?.model_loaded ? "online" : "offline"}`} />{health?.model_loaded ? "API and model online" : "API unavailable"}</div>
          </section>

          {apiError && <div className="alert error" role="alert"><strong>Backend unavailable</strong><span>{apiError}</span><small>Start FastAPI at {API_BASE} and refresh.</small></div>}
          {health && !health.model_loaded && <div className="alert error" role="alert"><strong>Model not ready</strong><span>{health.model_error || "Required model artifacts were not loaded."}</span></div>}

          <section className="summary-grid" aria-label="Trip overview">
            <article className="summary-card"><span className="summary-icon blue">↗</span><div><span>Available trips</span><strong>{loadingTrips ? "…" : overview.total}</strong></div><small>Demo and uploaded data</small></article>
            <article className="summary-card"><span className="summary-icon green">↑</span><div><span>Uploaded here</span><strong>{overview.uploaded}</strong></div><small>Validated trip file sets</small></article>
            <article className="summary-card"><span className="summary-icon purple">◎</span><div><span>Selected observations</span><strong>{loadingPreview ? "…" : overview.observations.toLocaleString()}</strong></div><small>GPS points in selected trip</small></article>
            <article className="summary-card"><span className="summary-icon orange">◈</span><div><span>Selected model</span><strong className="model-name">{activeResult?.model?.selected_model || modelInfo?.selected_model || "—"}</strong></div><small>{activeResult?.model?.feature_count || modelInfo?.feature_count || "—"} approved features</small></article>
          </section>

          <section className="work-grid" id="workspace">
            <article className="panel trip-panel" id="upload">
              <div className="panel-heading"><div><p className="eyebrow">INPUT</p><h2>Choose a trip</h2><p>Use a packaged student-test trip or upload a route-free trip bundle.</p></div><span className="step-number">01</span></div>
              <label className="field-label" htmlFor="trip-picker">Available trip IDs</label>
              <select id="trip-picker" className="trip-select" value={selectedTripId} onChange={(event) => selectTrip(event.target.value)} disabled={loadingTrips || trips.length === 0}>
                {trips.length === 0 && <option value="">No trips available</option>}
                {trips.map((trip) => <option key={trip.trip_id} value={trip.trip_id}>{trip.trip_id}{trip.uploaded ? " · uploaded" : " · demo"}</option>)}
              </select>
              {activePreview && <div className="trip-meta"><span>{activePreview.observation_count.toLocaleString()} observations</span><span>{activePreview.node_count.toLocaleString()} nodes</span><span>{activePreview.arc_count.toLocaleString()} directed arcs</span></div>}

              <div className="divider"><span>or upload files</span></div>
              <label className="upload-zone">
                <input type="file" accept=".track,.nodes,.arcs" multiple onChange={onFilesSelected} />
                <span className="upload-symbol">↑</span><strong>Select trip bundle</strong><span>Choose all three files in one selection</span><small>.track · .nodes · .arcs</small>
              </label>
              <div className="selected-files">
                {["track_file", "nodes_file", "arcs_file"].map((key) => <span key={key} className={uploadFiles[key] ? "file-chip ready" : "file-chip"}>{uploadFiles[key]?.name || key.replace("_file", "")}</span>)}
              </div>
              <button className="secondary-button full-width" onClick={uploadTrip} disabled={uploading || Object.keys(uploadFiles).length === 0}>{uploading ? "Validating files…" : "Upload trip bundle"}</button>
              <p className="privacy-note">Inference uses only GPS, nodes, and arcs. Ground-truth `.route` files are never requested.</p>
              <button className="primary-button full-width run-button" onClick={runMatching} disabled={!selectedTripId || matching || !health?.model_loaded}>
                {matching ? <><span className="spinner" /> Reconstructing route…</> : "Run map matching →"}
              </button>
              {actionError && <div className="inline-error" role="alert">{actionError}</div>}
            </article>

            <article className="panel map-panel" id="map-panel">
              <div className="panel-heading map-heading"><div><p className="eyebrow">GEOGRAPHIC VIEW</p><h2>{activeResult ? "Reconstructed route" : "GPS trace preview"}</h2><p>{selectedTripId || "Select a trip to show its trace"}</p></div>
                {activeResult && <a className="text-link" href={`${API_BASE}${activeResult.folium_map_url}`} target="_blank" rel="noreferrer">Open Folium map ↗</a>}
              </div>
              <RouteMap preview={activePreview} result={activeResult} />
              <div className="legend"><span><i className="legend-dot gps" />GPS trace</span>{activeResult && <><span><i className="legend-dot classifier" />Classifier top-1</span><span><i className="legend-dot hmm" />HMM / Viterbi</span></>}</div>
            </article>
          </section>

          {matching && <div className="processing-banner"><span className="spinner" />Generating candidates, scoring arcs, and decoding a connected route…</div>}
          {activeResult && (
            <section className="results-section" aria-live="polite">
              <div className="section-heading"><div><p className="eyebrow">RESULTS</p><h2>Route reconstruction summary</h2><p>Model scores are normalized per observation for decoding; they are not calibrated probabilities.</p></div><span className={`decoder-badge ${activeResult.decoder_status === "completed" ? "good" : "warning"}`}>{activeResult.decoder_status.replaceAll("_", " ")}</span></div>
              <div className="metric-grid">
                <article className="metric-card"><span>Candidate coverage</span><strong>{activeResult.metrics.observations_with_candidates.toLocaleString()} / {activePreview?.observation_count.toLocaleString()}</strong><small>{activeResult.metrics.observations_without_candidates.toLocaleString()} observations unmatched</small></article>
                <article className="metric-card"><span>Classifier continuity</span><strong>{percentage(activeResult.metrics.classifier_route_continuity)}</strong><small>{activeResult.metrics.classifier_connected_pairs} connected pairs</small></article>
                <article className="metric-card"><span>HMM continuity</span><strong>{percentage(activeResult.metrics.hmm_route_continuity)}</strong><small>{activeResult.metrics.hmm_connected_pairs} connected pairs</small></article>
                <article className="metric-card"><span>HMM route length</span><strong>{distance(activeResult.metrics.hmm_route_distance_m)}</strong><small>Sum of distinct selected arcs</small></article>
                <article className="metric-card"><span>Processing time</span><strong>{activeResult.metrics.processing_time_seconds.toFixed(2)} s</strong><small>{activeResult.metrics.candidate_count.toLocaleString()} candidate rows scored</small></article>
              </div>
              {activeResult.warnings?.length > 0 && <details className="warning-list"><summary>{activeResult.warnings.length} route or data warning(s)</summary><ul>{activeResult.warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul></details>}
              <div className="model-caption"><strong>{activeResult.model.selected_model}</strong><span>{activeResult.model.feature_count} features</span><span>seed {activeResult.model.random_seed ?? "—"}</span><span>{activeResult.model.model_version}</span></div>
            </section>
          )}

          <section className="panel history-panel" id="history">
            <div className="panel-heading"><div><p className="eyebrow">DATASET</p><h2>Available trips</h2><p>Only trips with `.track`, `.nodes`, and `.arcs` are listed.</p></div><button className="refresh-button" onClick={() => loadTrips(false)} disabled={loadingTrips}>{loadingTrips ? "Refreshing…" : "Refresh list"}</button></div>
            <div className="table-wrap"><table><thead><tr><th>TRIP ID</th><th>SOURCE</th><th>OBSERVATIONS</th><th>NODES</th><th>ARCS</th><th /></tr></thead>
              <tbody>{trips.map((trip) => <tr key={trip.trip_id} className={selectedTripId === trip.trip_id ? "selected-row" : ""}><td><strong>{trip.trip_id}</strong></td><td><span className="badge">{trip.uploaded ? "Uploaded" : "Demo"}</span></td><td>{selectedTripId === trip.trip_id ? activePreview?.observation_count?.toLocaleString() ?? "Loading" : "—"}</td><td>{selectedTripId === trip.trip_id ? activePreview?.node_count?.toLocaleString() ?? "—" : "—"}</td><td>{selectedTripId === trip.trip_id ? activePreview?.arc_count?.toLocaleString() ?? "—" : "—"}</td><td><button className="text-button" onClick={() => { selectTrip(trip.trip_id); navigate("Route Map"); }}>Select</button></td></tr>)}
                {!trips.length && !loadingTrips && <tr><td colSpan="6" className="empty-row">No trips are available from the configured data folder.</td></tr>}
              </tbody></table></div>
          </section>
          <footer className="page-footer">PathSnap · Distributed Map Matching & Route Reconstruction</footer>
        </div>
      </main>
    </div>
  );
}

export default App;
