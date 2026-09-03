// Thin client for the FastAPI backend in ../backend. See backend/README.md
// for what it expects (model weights + data files) before it can classify
// anything.

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options) {
  let res;
  try {
    // ngrok's free tier serves an HTML warning page to the first request from any client
    // unless this header is present; harmless against any other backend host.
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: { ...options?.headers, "ngrok-skip-browser-warning": "true" },
    });
  } catch {
    throw new Error(`Could not reach the backend at ${API_BASE}. Is it running (uvicorn main:app)?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export function uploadImage(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/images", { method: "POST", body: form });
}

export function listImages(status) {
  const qs = status ? `?status=${encodeURIComponent(status)}` : "";
  return request(`/api/images${qs}`);
}

export function getImage(id) {
  return request(`/api/images/${id}`);
}

export function getDashboard() {
  return request("/api/dashboard");
}
