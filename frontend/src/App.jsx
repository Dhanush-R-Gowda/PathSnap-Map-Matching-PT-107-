import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { api, getBackendUrl, setBackendUrl } from "./api";

function FitBounds({ points }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) map.fitBounds(points, { padding: [24, 24] });
  }, [points, map]);
  return null;
}

const km = (m) => (m == null ? "--" : `${(m / 1000).toFixed(2)} km`);

function App() {
  const [backend, setBackend] = useState(getBackendUrl());
  const [online, setOnline] = useState(null);
  const [files, setFiles] = useState({});
  const [samples, setSamples] = useState({});
  const [sample, setSample] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [showGps, setShowGps] = useState(true);
  const [showArcs, setShowArcs] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const h = await api.health(backend);
      setOnline(h.model_loaded);
      setError((e) => (e.startsWith("Cannot reach") ? "" : e));
      setSamples(await api.samples(backend));
      setHistory(await api.trips(backend));
    } catch (e) {
      setOnline(false);
      setError(e.message);
    }
  }, [backend]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function saveBackend(e) {
    e.preventDefault();
    setBackendUrl(backend);
    refresh();
  }

  function pick(ext, file) {
    setFiles((f) => ({ ...f, [ext]: file }));
  }

  // Select all three files at once; they are assigned by extension.
  function pickMany(list) {
    for (const f of Array.from(list)) {
      const ext = f.name.split(".").pop().toLowerCase();
      if (["track", "nodes", "arcs"].includes(ext)) pick(ext, f);
    }
  }

  async function run(prepare) {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const { trip_id } = await prepare();
      await api.predict(backend, trip_id);
      setResult(await api.results(backend, trip_id));
      setHistory(await api.trips(backend));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const runUpload = () => run(() => api.upload(backend, files));
  const runSample = () => {
    const [part, id] = sample.split("/");
    return run(() => api.loadSample(backend, part, id));
  };
  async function openTrip(id) {
    setBusy(true);
    setError("");
    try {
      setResult(await api.results(backend, id));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const gps = useMemo(() => (result ? result.original_gps.map((p) => [p.lat, p.lon]) : []), [result]);
  const segments = useMemo(
    () => (result ? result.route_segments.map((s) => s.map((p) => [p.lat, p.lon])) : []),
    [result]
  );
  const allFiles = ["track", "nodes", "arcs"].every((e) => files[e]);
  const m = result?.metrics;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">P</div>
          <div>
            <h2>PathSnap</h2>
            <span>MAP MATCHING SYSTEM</span>
          </div>
        </div>
        <p className="nav-heading">BACKEND</p>
        <form onSubmit={saveBackend} className="backend-form">
          <input value={backend} onChange={(e) => setBackend(e.target.value)} aria-label="Backend URL" />
          <button className="text-button" type="submit">Connect</button>
        </form>
        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" style={{ background: online ? "#159b68" : "#d33" }}></span>
            {online === null ? "Checking…" : online ? "Backend online" : "Backend offline"}
          </div>
          <p>ML probabilities + HMM/Viterbi</p>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div><span className="breadcrumb">Evaluator / </span><strong>Route Reconstruction</strong></div>
        </header>

        <div className="content">
          <section className="welcome">
            <div>
              <p className="eyebrow">MAP MATCHING & ROUTE RECONSTRUCTION</p>
              <h1>Reconstruct a vehicle route</h1>
              <p className="subtitle">Provide a GPS track and its road network; no ground-truth route is needed.</p>
            </div>
          </section>

          {error && <p role="alert" className="error-message">⚠ {error}</p>}

          {m && (
            <section className="stats">
              <div className="stat-card"><p>GPS observations</p><h2>{m.observations}</h2><small>{m.matched_observations} matched</small></div>
              <div className="stat-card"><p>Route distance</p><h2>{km(m.route_distance_m)}</h2><small>{m.route_arcs} road segments</small></div>
              <div className="stat-card"><p>Route continuity</p><h2>{(m.route_continuity * 100).toFixed(1)}%</h2><small>{m.chain_breaks} chain breaks</small></div>
              <div className="stat-card"><p>Processing time</p><h2>{m.processing_time_seconds.toFixed(1)} s</h2><small>{m.mean_candidates_per_obs.toFixed(1)} candidates / obs</small></div>
            </section>
          )}

          <section className="workspace-grid">
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div><h3>Trip input</h3><p>Upload files or pick a trip stored on the backend.</p></div>
              </div>

              <label className="file-row">
                <span>all 3</span>
                <input type="file" multiple accept=".track,.nodes,.arcs" onChange={(e) => pickMany(e.target.files)} />
              </label>
              {["track", "nodes", "arcs"].map((ext) => (
                <label key={ext} className="file-row">
                  <span>.{ext}</span>
                  <em>{files[ext]?.name || "not selected"}</em>
                  <input type="file" hidden accept={`.${ext}`} onChange={(e) => pick(ext, e.target.files[0])} />
                </label>
              ))}
              <button className="primary-button" disabled={!allFiles || busy} onClick={runUpload}>
                {busy ? "Reconstructing…" : "Upload & reconstruct →"}
              </button>

              <p className="upload-note">— or choose a sample trip —</p>
              <select value={sample} onChange={(e) => setSample(e.target.value)} aria-label="Sample trip">
                <option value="">Select a trip…</option>
                {Object.entries(samples).map(([part, ids]) => (
                  <optgroup key={part} label={part}>
                    {ids.map((id) => <option key={id} value={`${part}/${id}`}>{id}</option>)}
                  </optgroup>
                ))}
              </select>
              <button className="primary-button" disabled={!sample || busy} onClick={runSample}>
                {busy ? "Reconstructing…" : "Reconstruct sample →"}
              </button>
              <p className="upload-note">Ground-truth .route files are never required or exposed.</p>
            </div>

            <div className="panel map-panel">
              <div className="panel-heading">
                <div><h3>Route visualisation</h3><p>Noisy GPS (blue) and reconstructed route (green)</p></div>
                <span className="map-label">
                  <label><input type="checkbox" checked={showGps} onChange={(e) => setShowGps(e.target.checked)} /> GPS</label>{" "}
                  <label><input type="checkbox" checked={showArcs} onChange={(e) => setShowArcs(e.target.checked)} /> matched arcs</label>
                </span>
              </div>
              <div className="map">
                {busy && <div className="map-loading">Processing trip…</div>}
                <MapContainer center={[12.97, 77.59]} zoom={12} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
                  <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  {gps.length > 0 && <FitBounds points={gps} />}
                  {showGps && gps.filter((_, i) => i % Math.max(1, Math.floor(gps.length / 800)) === 0).map((p, i) => (
                    <CircleMarker key={i} center={p} radius={2} pathOptions={{ color: "#3478f6", weight: 1 }} />
                  ))}
                  {showArcs && result?.matched_arcs.map((a) => (
                    <Polyline key={a.obs_index} positions={a.polyline.map((p) => [p.lat, p.lon])} pathOptions={{ color: "#e08a00", weight: 3 }} />
                  ))}
                  {segments.map((s, i) => (
                    <Polyline key={i} positions={s} pathOptions={{ color: "#159b68", weight: 5 }} />
                  ))}
                  {gps.length > 0 && (
                    <>
                      <CircleMarker center={gps[0]} radius={7} pathOptions={{ color: "#159b68", fillOpacity: 1 }}><Popup>Start</Popup></CircleMarker>
                      <CircleMarker center={gps[gps.length - 1]} radius={7} pathOptions={{ color: "#d33", fillOpacity: 1 }}><Popup>End</Popup></CircleMarker>
                    </>
                  )}
                </MapContainer>
              </div>
              <div className="map-footer">
                <span>{result ? `Trip ${result.trip_id}` : "No trip reconstructed yet"}</span>
                <span>{m ? `${m.unmatched_observations} observations without candidates` : ""}</span>
              </div>
            </div>
          </section>

          <section className="panel history-panel">
            <div className="panel-heading">
              <div><h3>Trips on backend</h3><p>Click a processed trip to reopen it</p></div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>TRIP ID</th><th>OBSERVATIONS</th><th>STATUS</th><th>DISTANCE</th></tr></thead>
                <tbody>
                  {history.length === 0 && <tr><td colSpan="4">No trips yet</td></tr>}
                  {history.map((t) => (
                    <tr key={t.trip_id} onClick={() => t.status === "completed" && openTrip(t.trip_id)} style={{ cursor: "pointer" }}>
                      <td><strong>{t.trip_id}</strong></td>
                      <td>{t.observations}</td>
                      <td><span className={`badge ${t.status === "completed" ? "completed" : "processing"}`}>{t.status}</span></td>
                      <td>{km(t.route_distance_m)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <footer>PathSnap · Distributed Map Matching & Route Reconstruction</footer>
        </div>
      </main>
    </div>
  );
}

export default App;
