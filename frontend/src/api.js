// Backend URL is configurable (never hard-coded to localhost): Vite env var,
// then the value saved from the UI, then the current host on port 8000.
const DEFAULT_URL =
  import.meta.env.VITE_API_URL || `http://${window.location.hostname || "localhost"}:8000`;

export function getBackendUrl() {
  try {
    return localStorage.getItem("pathsnap_backend_url") || DEFAULT_URL;
  } catch {
    return DEFAULT_URL;
  }
}

export function setBackendUrl(url) {
  try {
    localStorage.setItem("pathsnap_backend_url", url);
  } catch {
    /* storage unavailable */
  }
}

async function request(base, path, options) {
  let res;
  try {
    res = await fetch(`${base.replace(/\/$/, "")}${path}`, options);
  } catch {
    throw new Error(`Cannot reach backend at ${base}. Check the URL, network and that the server is running.`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: (b) => request(b, "/health"),
  samples: (b) => request(b, "/api/v1/samples"),
  loadSample: (b, part, id) => request(b, `/api/v1/samples/${part}/${id}/load`, { method: "POST" }),
  upload: (b, files) => {
    const form = new FormData();
    for (const ext of ["track", "nodes", "arcs"]) form.append(ext, files[ext]);
    return request(b, "/api/v1/trips/upload", { method: "POST", body: form });
  },
  predict: (b, id) => request(b, `/api/v1/trips/${id}/predict`, { method: "POST" }),
  results: (b, id) => request(b, `/api/v1/trips/${id}/results`),
  trips: (b) => request(b, "/api/v1/trips"),
};
