# BP Shelf Intelligence — backend

FastAPI wrapper around the detect → embed → gate → classify → OCR → decide
cascade from the Brand Partners Engineering Guide, ported from the
Streamlit `app.py` so the React frontend in `../frontend` can call it as a JSON
API. All thresholds/model ids/crop geometry in `pipeline.py` are copied
verbatim from the guide — see that file's header before changing any of
them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Drop these files in before starting the server:

```
models/
  product_detector_best.pt              (or best_augmented.pt)
  REJECT_GATE_DINOV3_MLP_V3.pt          (optional — falls back to no reject-gate)
  BP_TARGET_GATE_DINOV3_LOGREG.joblib   (optional — falls back to no BP-target gate)
  SKU_32CLASS_DINOV3_MLP.pt             (optional — falls back to cosine similarity
                                          against data/embeddings.json)
data/
  catalog_32class.txt        (required if SKU_32CLASS_DINOV3_MLP.pt is present)
  embeddings.json            (required if SKU_32CLASS_DINOV3_MLP.pt is absent)
  class_keywords.json        (required for OCR verification, stage 5)
```

The detector and the DINOv3 backbone (downloaded from Hugging Face on first
run) are mandatory — there's no fallback for those two. Everything else is
optional and degrades the same way it did in the Streamlit app: missing
gates/head just widen what counts as UNKNOWN/NON_TARGET instead of hard
failing, but that state is **not** the guide's validated configuration and
its numbers shouldn't be reported as the guide's baseline.

## Run

```bash
uvicorn main:app --reload --port 8000
```

`image_url` in every API response is built as an **absolute** URL
(`PUBLIC_BASE_URL` + `/media/...`, default `http://localhost:8000`) because
the frontend runs on a different origin (Vite on :5173) than this API — a
root-relative `/media/...` src would resolve against the *frontend's*
origin in the browser and 404. If this API isn't reachable at
`localhost:8000` from the browser (e.g. it's deployed, or port-forwarded
under a different host/port), set `PUBLIC_BASE_URL` accordingly:

```bash
PUBLIC_BASE_URL=https://your-api-host uvicorn main:app --port 8000
```

The frontend (`npm run dev` in the repo root) expects this at
`http://localhost:8000` by default — see `VITE_API_BASE` in the frontend's
`.env` if you need to change that.

## Run on Colab GPU (no local GPU needed)

If your machine has no GPU and the cascade is too slow to test locally, `colab_run.ipynb` runs
this backend on a free Colab GPU runtime and joins it to a Tailscale network, so your local
frontend can point `VITE_API_BASE` at it instead of `localhost:8000`. We use Tailscale rather
than an HTTP tunnel (ngrok/Cloudflare) because those force-close a request after a fixed time
(~100s–300s on their free tiers), which a slow multi-crop shelf-image upload can exceed —
Tailscale is a direct private network link with no such cap, at the cost of needing Tailscale
installed on your local machine too (same tailnet as the Colab VM). Open the notebook in Colab,
follow the one-time setup notes in its first cell (Tailscale on your machine, Google Drive for
`models/`/`data/`, a Hugging Face token for the gated DINOv3 model, a Tailscale auth key), and
run all cells — the last one prints the backend's tailnet address to paste into `frontend/.env`.
`OCR_DEVICE=gpu` and a `paddlepaddle-gpu` install are handled by the notebook so the OCR stage
gets the GPU speedup too, not just detection/embedding.

## Endpoints

- `POST /api/images` — multipart upload (`file` field). Runs the full
  cascade synchronously and returns the persisted image record. Can take a
  while on CPU (two OCR passes per crop).
- `GET /api/images?status=approved|rejected` — list of processed images.
- `GET /api/images/{id}` — one image with its detections.
- `GET /api/dashboard` — aggregate stats across all processed images.
- `GET /media/...` — serves the stored shelf photos.

## What's intentionally not ported

The Streamlit app's accept/reject buttons that sorted crops into
`accepted/`/`rejected/` folders were about curating a training dataset —
that's a separate concern from this frontend (which is about browsing
pipeline results, not building the next training set) and wasn't ported
here.

## Data model note

The guide's per-crop decision is one of `EXACT_SKU / UNKNOWN / REJECT /
NON_TARGET`. This frontend only distinguishes "verified" (EXACT_SKU) vs.
"unknown" (everything else) per image, and derives a whole-image
`approved`/`rejected` status from whichever bucket has more crops — see
`save_image_result()` in `storage.py`. The specific decision per crop is
still stored in the `detections` table if you need finer detail later.
