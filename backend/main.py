"""
FastAPI wrapper around pipeline.py, exposing the detect->classify->OCR
cascade as JSON endpoints for the React frontend in ../src.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

See README.md for the model/data files this expects under models/ and data/.
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pipeline
import storage

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

app.mount("/media", StaticFiles(directory=pipeline.MEDIA_DIR), name="media")


@app.on_event("startup")
def on_startup():
    storage.init_db()


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.post("/api/images", status_code=201)
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type {ext!r}. Use JPG or PNG.")

    image_id = storage.new_image_id()
    stored_name = f"{image_id}{ext}"
    stored_path = os.path.join(pipeline.UPLOAD_DIR, stored_name)

    with open(stored_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = pipeline.process_shelf_image(stored_path, image_id)
    except FileNotFoundError as e:
        # Model weights / reference data not dropped into models//data/ yet.
        os.remove(stored_path)
        raise HTTPException(503, str(e))
    except Exception as e:
        os.remove(stored_path)
        raise HTTPException(500, f"Pipeline error: {e}")

    record = storage.save_image_result(
        image_id=image_id,
        filename=file.filename,
        image_path=f"uploads/{stored_name}",
        image_width=result["image_width"],
        image_height=result["image_height"],
        detections=result["detections"],
    )
    return record


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


@app.get("/api/dashboard")
def get_dashboard():
    return storage.dashboard_data()
