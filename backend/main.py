"""
FastAPI wrapper around pipeline.py, exposing the detect->classify->OCR
cascade as JSON endpoints for the React frontend in ../src.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

See README.md for the model/data files this expects under models/ and data/.
"""

import asyncio
import os
import threading
import time

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

import pipeline
import storage
import timing

# The pipeline (YOLO + DINOv3 + PaddleOCR) is CPU-bound with no GPU in this
# environment and will happily use every core it can get. Running several
# uploads' pipelines at once doesn't parallelize them for free -- it just
# makes all of them (and every other request competing for the same cores,
# including simple GET /api/images / /api/dashboard reads) slower. Cap how
# many pipeline runs execute at once; extra uploads queue here instead of
# fighting the running one for CPU. Override via PIPELINE_CONCURRENCY.
PIPELINE_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("PIPELINE_CONCURRENCY", "1")))

# Real per-upload progress (percent, stage), keyed by a client-generated
# progress_id -- polled by the frontend instead of a simulated timer. Plain
# dict + lock: pipeline.process_shelf_image reports from inside a
# threadpool worker thread, while GET /api/upload-progress reads from
# another (FastAPI runs sync `def` routes in a threadpool too), so writes
# and reads need to be synchronized across threads.
_progress_lock = threading.Lock()
_progress_store = {}

FRONTEND_ORIGINS = os.environ.get(
    "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app = FastAPI(title="BP Shelf Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    storage.init_db()
    pipeline.log_startup_info()


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.post("/api/images", status_code=201)
async def upload_image(file: UploadFile = File(...), progress_id: str | None = Form(None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type {ext!r}. Use JPG or PNG.")

    image_id = storage.new_image_id()
    image_bytes = await file.read()
    t_start = time.perf_counter()

    def _on_progress(percent, stage):
        if progress_id:
            with _progress_lock:
                _progress_store[progress_id] = {"percent": percent, "stage": stage}

    if progress_id:
        # Set before acquiring the semaphore: if another upload is
        # currently running, this one genuinely is just queued at 0%,
        # not lying about progress it hasn't made yet.
        with _progress_lock:
            _progress_store[progress_id] = {"percent": 0, "stage": "queued"}

    try:
        try:
            # process_shelf_image is a synchronous, CPU/GPU-bound call
            # (detect, embed, gate, classify, OCR). Run it off the event
            # loop so a single upload doesn't block every other request
            # (GET /api/images, /api/dashboard, ...) for the whole
            # pipeline duration, and cap how many run at once so a
            # multi-file batch doesn't stall everything by saturating the
            # CPU with parallel inference.
            async with PIPELINE_SEMAPHORE:
                result = await run_in_threadpool(
                    pipeline.process_shelf_image, image_bytes, on_progress=_on_progress
                )
        except FileNotFoundError as e:
            # Model weights / reference data not dropped into models//data/ yet.
            raise HTTPException(503, str(e))
        except Exception as e:
            raise HTTPException(500, f"Pipeline error: {e}")

        t_db_start = time.perf_counter()
        record = await run_in_threadpool(
            storage.save_image_result,
            image_id=image_id,
            filename=file.filename,
            image_bytes=image_bytes,
            content_type=file.content_type or "image/jpeg",
            image_width=result["image_width"],
            image_height=result["image_height"],
            detections=result["detections"],
        )

        image_timings = result["timings"]
        image_timings.add("db_save", time.perf_counter() - t_db_start)
        image_timings.total = time.perf_counter() - t_start
        timing.log_timings(image_id, image_timings)

        return record
    finally:
        if progress_id:
            with _progress_lock:
                _progress_store.pop(progress_id, None)


@app.get("/api/upload-progress/{progress_id}")
def get_upload_progress(progress_id: str):
    with _progress_lock:
        entry = _progress_store.get(progress_id)
    return entry or {"percent": None, "stage": None}


@app.get("/api/images")
def get_images(status: str | None = None):
    if status not in (None, "approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'")
    return {"images": storage.list_images(status=status)}


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    record = storage.get_image(image_id)
    if record is None:
        raise HTTPException(404, "Image not found")
    return record


@app.get("/api/images/{image_id}/file")
def get_image_file(image_id: str):
    file = storage.get_image_file(image_id)
    if file is None:
        raise HTTPException(404, "Image not found")
    return Response(content=file["data"], media_type=file["content_type"])


@app.get("/api/dashboard")
def get_dashboard():
    return storage.dashboard_data()
