# Lipton SKU Classifier — frontend + backend

A React (Vite) frontend, originally a mocked-data implementation of the
`Lipton SKU Classifier.dc.html` Claude Design export, now wired to a real
FastAPI backend in `backend/` that runs the detect → embed → gate →
classify → OCR cascade from the Brand Partners Engineering Guide.

- Uploading a shelf photo on Home runs the real pipeline server-side and
  persists the result; Uploads, Rejected, and Dashboard read that data back
  from the backend instead of mock data.
- Shelf photos show the actual uploaded image, with detection boxes drawn
  at their real positions (converted from pixel bboxes to percentages by
  `frontend/src/lib/detections.js`), not a placeholder canvas.
- Per-image `approved`/`rejected` status is derived server-side from
  whether verified (EXACT_SKU) crops are at least as numerous as true
  UNKNOWN crops — see `backend/storage.py`'s `save_image_result()`.

## Structure

This is a monorepo with two independent projects:

```
frontend/   React (Vite) app
backend/    FastAPI service
```

- `frontend/src/pages/` — the four screens: Home (upload), Uploads, Rejected,
  Dashboard, plus Preview (bounding-box detail view, reused for both Uploads
  and Rejected).
- `frontend/src/components/` — Button, Badge, Avatar, Navbar, ShelfCanvas
  (real photo + bounding boxes), recreated from the Polaris UI Kit
  design-system export.
- `frontend/src/api.js` — fetch client for the backend.
- `frontend/src/lib/detections.js` — pixel-bbox → percentage-box layout and
  the decision→color/tag scheme, shared by the list, preview, and dashboard
  views.
- `frontend/src/context/BatchQueueContext.jsx` — drives the "This batch"
  upload progress list on Home from the real upload request lifecycle.
- `backend/` — FastAPI service; see `backend/README.md` for setup, the
  model/data files it expects, and the API it exposes.

## Run

Frontend:

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE if the backend isn't on :8000
npm run dev             # http://localhost:5173
npm run build            # production build to frontend/dist/
```

Backend (see `backend/README.md` for the model/data files it needs first):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Run both together to use the app — Home's upload flow, and everything
downstream of it, requires the backend to be running.
