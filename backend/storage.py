"""
SQLite persistence for processed shelf images + their per-crop detections,
plus the deterministic per-class color used everywhere the frontend draws a
box or a legend swatch (so a class always renders the same color across the
Uploads/Rejected/Preview/Dashboard views without the frontend needing its
own copy of this logic).
"""

import colorsys
import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")

# image_url in every API response is built from this. It must be an
# ABSOLUTE url (not "/media/...") -- the frontend is served from a
# different origin (Vite on :5173) than this API (:8000), and a
# root-relative src in an <img> tag resolves against the PAGE's origin,
# not this server's, so it'd 404 against the frontend dev server instead
# of ever reaching this backend. Override with PUBLIC_BASE_URL if the API
# isn't reachable at localhost:8000 from the browser (e.g. deployed).
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000")
DEFAULT_MEDIA_BASE = PUBLIC_BASE_URL.rstrip("/") + "/media"

UNKNOWN_COLOR = "#eab308"
EXACT_SKU = "EXACT_SKU"
UNKNOWN = "UNKNOWN"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_width INTEGER NOT NULL,
            image_height INTEGER NOT NULL,
            verified_count INTEGER NOT NULL,
            unknown_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            class_name TEXT,
            decision TEXT NOT NULL,
            score REAL NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def class_color(class_name):
    """Deterministic, stable-across-restarts color for a SKU class name.
    Same class name always hashes to the same color, so it stays consistent
    between the list, preview, and dashboard views without a hardcoded
    palette (the real catalog has 32 classes, not the 5 the mock data
    shipped with)."""
    if not class_name:
        return UNKNOWN_COLOR
    digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.40, 0.55)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def new_image_id():
    return "img_" + uuid.uuid4().hex[:12]


def save_image_result(image_id, filename, image_path, image_width, image_height, detections):
    """Persists one processed shelf image and its detections.

    `image_path` is the path to the stored original photo, relative to
    backend/media (e.g. "uploads/img_xxx.jpg"), so it can be turned into a
    URL under the /media static mount.

    Status rule (matches the Uploads/Rejected copy already in the
    frontend): approved when verified (EXACT_SKU) crops are at least as
    numerous as true UNKNOWN crops. REJECT (unusable crop, failed the
    validity gate) and NON_TARGET (not a BP product at all) crops are
    excluded from this comparison entirely -- a shelf full of blurry/
    off-target crops shouldn't tip an otherwise-clean image into
    "rejected" just because they got folded into the same bucket as
    genuine classification ambiguity.
    """
    verified_count = sum(1 for d in detections if d["decision"] == EXACT_SKU)
    unknown_count = sum(1 for d in detections if d["decision"] == UNKNOWN)
    status = "approved" if verified_count >= unknown_count else "rejected"
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _connect()
    conn.execute(
        """INSERT INTO images
           (id, filename, status, image_path, image_width, image_height,
            verified_count, unknown_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (image_id, filename, status, image_path, image_width, image_height,
         verified_count, unknown_count, created_at),
    )
    conn.executemany(
        """INSERT INTO detections
           (image_id, class_name, decision, score, x1, y1, x2, y2)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (image_id, d["matched_class"], d["decision"], d["score"], *d["bbox"])
            for d in detections
        ],
    )
    conn.commit()
    conn.close()
    return get_image(image_id)


def _row_to_summary(row, media_base):
    return {
        "id": row["id"],
        "name": row["filename"],
        "status": row["status"],
        "captured": "Uploaded " + row["created_at"][:16].replace("T", " "),
        "image_url": f"{media_base}/{row['image_path']}",
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "verified_count": row["verified_count"],
        "unknown_count": row["unknown_count"],
        "created_at": row["created_at"],
    }


def _detections_for(conn, image_id):
    rows = conn.execute(
        "SELECT class_name, decision, score, x1, y1, x2, y2 FROM detections WHERE image_id = ? ORDER BY id",
        (image_id,),
    ).fetchall()
    return [
        {
            "class_name": r["class_name"],
            "decision": r["decision"],
            "score": r["score"],
            "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]],
            "color": class_color(r["class_name"]),
        }
        for r in rows
    ]


def list_images(status=None, media_base=DEFAULT_MEDIA_BASE):
    conn = _connect()
    if status:
        rows = conn.execute(
            "SELECT * FROM images WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM images ORDER BY created_at DESC").fetchall()

    out = []
    for row in rows:
        summary = _row_to_summary(row, media_base)
        summary["detections"] = _detections_for(conn, row["id"])
        out.append(summary)
    conn.close()
    return out


def get_image(image_id, media_base=DEFAULT_MEDIA_BASE):
    conn = _connect()
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None:
        conn.close()
        return None
    summary = _row_to_summary(row, media_base)
    summary["detections"] = _detections_for(conn, image_id)
    conn.close()
    return summary


def dashboard_data(media_base=DEFAULT_MEDIA_BASE):
    """Aggregation ported from the mock Dashboard.jsx page, now computed
    over real persisted images/detections instead of the 6-item mock
    catalog."""
    conn = _connect()
    approved_rows = conn.execute(
        "SELECT * FROM images WHERE status = 'approved' ORDER BY created_at DESC"
    ).fetchall()
    rejected_rows = conn.execute("SELECT * FROM images WHERE status = 'rejected'").fetchall()

    classified = sum(r["verified_count"] + r["unknown_count"] for r in approved_rows)
    verified = sum(r["verified_count"] for r in approved_rows)
    unknown_in_uploads = sum(r["unknown_count"] for r in approved_rows)
    unknown_in_rejected = sum(r["unknown_count"] for r in rejected_rows)

    per_class = {}
    for r in approved_rows:
        for d in conn.execute(
            "SELECT class_name FROM detections WHERE image_id = ? AND decision = 'EXACT_SKU'",
            (r["id"],),
        ):
            per_class[d["class_name"]] = per_class.get(d["class_name"], 0) + 1

    sku_rows = [
        {"name": name, "color": class_color(name), "classified": n, "verified": n, "rejected": 0}
        for name, n in sorted(per_class.items(), key=lambda kv: -kv[1])
    ]
    sku_rows.append({
        "name": "Unknown", "color": UNKNOWN_COLOR,
        "classified": unknown_in_uploads, "verified": 0, "rejected": unknown_in_uploads,
    })

    image_rows = [
        {
            "id": r["id"],
            "name": r["filename"],
            "classified": r["verified_count"] + r["unknown_count"],
            "verified": r["verified_count"],
            "rejected": r["unknown_count"],
        }
        for r in approved_rows
    ]

    conn.close()
    return {
        "approved_total": len(approved_rows),
        "stats": {
            "classified": classified,
            "verified": verified,
            "rejected_images": len(rejected_rows),
            "unknown_in_uploads": unknown_in_uploads,
            "unknown_in_rejected": unknown_in_rejected,
        },
        "sku_rows": sku_rows,
        "image_rows": image_rows,
    }
