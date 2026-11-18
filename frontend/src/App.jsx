
import { useState } from "react";
import "./App.css";
import { MapContainer, TileLayer, Polyline, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const trips = [
  { id: "TRIP-001", date: "Today, 10:30 AM", status: "Completed", distance: "4.8 km" },
  { id: "TRIP-002", date: "Today, 09:15 AM", status: "Completed", distance: "7.2 km" },
  { id: "TRIP-003", date: "Yesterday, 04:20 PM", status: "Processing", distance: "--" },
];

function App() {
  const [file, setFile] = useState(null);
  const [active, setActive] = useState("Dashboard");
  const [uploaded, setUploaded] = useState(false);

  const navigation = [
    ["▦", "Dashboard"],
    ["↑", "Upload Trip"],
    ["⌖", "Route Map"],
    ["◷", "Trip History"],
  ];

  function handleUpload(event) {
    const selected = event.target.files[0];
    if (selected) {
      setFile(selected);
      setUploaded(false);
    }
  }

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

        <p className="nav-heading">WORKSPACE</p>
        <nav>
          {navigation.map(([icon, name]) => (
            <button
              key={name}
              className={`nav-item ${active === name ? "active" : ""}`}
              onClick={() => setActive(name)}
            >
              <span>{icon}</span>{name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot"></span>
            System Online
          </div>
          <p>PathSnap v1.0</p>
          <p>Distributed Map Matching</p>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace / </span>
            <strong>{active}</strong>
          </div>
          <div className="profile">
            <span className="profile-avatar">B</span>
            <span>Team Member</span>
          </div>
        </header>

        <div className="content">
          <section className="welcome">
            <div>
              <p className="eyebrow">MAP MATCHING & ROUTE RECONSTRUCTION</p>
              <h1>Welcome to PathSnap</h1>
              <p className="subtitle">
                Transform noisy GPS trajectories into accurate road routes.
              </p>
            </div>
            <div className="date-label">● Project Dashboard</div>
          </section>

          <section className="stats">
            <div className="stat-card">
              <span className="stat-icon blue">⌖</span>
              <p>Total Trips</p>
              <h2>03</h2>
              <small>Uploaded trips</small>
            </div>
            <div className="stat-card">
              <span className="stat-icon green">✓</span>
              <p>Completed</p>
              <h2>02</h2>
              <small>Successfully processed</small>
            </div>
            <div className="stat-card">
              <span className="stat-icon orange">◷</span>
              <p>Processing</p>
              <h2>01</h2>
              <small>Awaiting reconstruction</small>
            </div>
            <div className="stat-card">
              <span className="stat-icon purple">↗</span>
              <p>Road Distance</p>
              <h2>12 km</h2>
              <small>Sample data</small>
            </div>
          </section>

          <section className="workspace-grid">
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div>
                  <h3>Upload Trip Data</h3>
                  <p>Import your GPS trajectory and road network files.</p>
                </div>
                <span className="heading-icon">↑</span>
              </div>

              <label className="upload-zone">
                <span className="upload-symbol">⇧</span>
                <strong>Choose a trip file</strong>
                <span>Click to browse your computer</span>
                <small>Supported: .track, .nodes, .arcs</small>
                <input
                  type="file"
                  accept=".track,.nodes,.arcs"
                  multiple
                  onChange={handleUpload}
                />
              </label>

              <div className="file-info">
                {file ? (
                  <>
                    <span>📄 {file.name}</span>
                    <button onClick={() => { setFile(null); setUploaded(false); }}>
                      Remove
                    </button>
                  </>
                ) : (
                  <span>No file selected</span>
                )}
              </div>

              <button
                className="primary-button"
                disabled={!file}
                onClick={() => setUploaded(true)}
              >
                Upload Trip Data →
              </button>

              {uploaded && (
                <p className="success-message">
                  File selected successfully. Backend upload will be connected next.
                </p>
              )}

              <p className="upload-note">
                Upload the required trip files together when the backend is connected.
                Ground-truth .route files are not needed for evaluator inference.
              </p>
            </div>

            <div className="panel map-panel">
              <div className="panel-heading">
                <div>
                  <h3>Route Visualization</h3>
                  <p>Preview of original and reconstructed trajectories</p>
                </div>
                <span className="map-label">DEMO MAP</span>
              </div>

              
<div className="map">
  <MapContainer
    center={[12.9716, 77.5946]}
    zoom={14}
    scrollWheelZoom={true}
    style={{ height: "100%", width: "100%" }}
  >
    <TileLayer
      attribution='&copy; OpenStreetMap contributors'
      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    />

    <Polyline
      positions={[
        [12.9716, 77.5946],
        [12.9730, 77.5970],
        [12.9750, 77.5990],
        [12.9780, 77.6010],
      ]}
      pathOptions={{ color: "#3478f6", weight: 5, dashArray: "8 8" }}
    />

    <Polyline
      positions={[
        [12.9716, 77.5946],
        [12.9725, 77.5960],
        [12.9755, 77.5980],
        [12.9780, 77.6010],
      ]}
      pathOptions={{ color: "#159b68", weight: 5 }}
    />

    <Marker position={[12.9716, 77.5946]}>
      <Popup>Start point</Popup>
    </Marker>

    <Marker position={[12.9780, 77.6010]}>
      <Popup>End point</Popup>
    </Marker>
  </MapContainer>
</div>

              <div className="map-footer">
                <span>Map preview uses sample data</span>
                <span>Interactive map integration coming next</span>
              </div>
            </div>
          </section>

          <section className="panel history-panel">
            <div className="panel-heading">
              <div>
                <h3>Recent Trips</h3>
                <p>Your recently uploaded trajectories</p>
              </div>
              <button className="text-button" onClick={() => setActive("Trip History")}>
                View all →
              </button>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>TRIP ID</th>
                    <th>DATE</th>
                    <th>STATUS</th>
                    <th>DISTANCE</th>
                  </tr>
                </thead>
                <tbody>
                  {trips.map((trip) => (
                    <tr key={trip.id}>
                      <td><strong>{trip.id}</strong></td>
                      <td>{trip.date}</td>
                      <td>
                        <span className={`badge ${trip.status === "Completed" ? "completed" : "processing"}`}>
                          {trip.status}
                        </span>
                      </td>
                      <td>{trip.distance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <footer>© 2026 PathSnap · Distributed Map Matching & Route Reconstruction</footer>
        </div>
      </main>
    </div>
  );
}

export default App;
