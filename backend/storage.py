"""
Postgres persistence for processed shelf images (bytes + metadata), their
per-crop detections, and the classifier's reference embedding gallery
(pgvector), plus the deterministic per-class color used everywhere the
frontend draws a box or a legend swatch (so a class always renders the same
color across the Uploads/Rejected/Preview/Dashboard views without the
frontend needing its own copy of this logic).
"""

import colorsys
import hashlib
import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector

# image_url in every API response is built from this. It must be an
# ABSOLUTE url (not "/api/...") -- the frontend is served from a
# different origin (Vite on :5173) than this API (:8000), and a
# root-relative src in an <img> tag resolves against the PAGE's origin,
# not this server's, so it'd 404 against the frontend dev server instead
# of ever reaching this backend. Override with PUBLIC_BASE_URL if the API
# isn't reachable at localhost:8000 from the browser (e.g. deployed).
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000")

UNKNOWN_COLOR = "#eab308"
EXACT_SKU = "EXACT_SKU"
UNKNOWN = "UNKNOWN"

_POOL = psycopg2.pool.ThreadedConnectionPool(
    1,
    10,
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=os.environ.get("POSTGRES_PORT", "5433"),
    dbname=os.environ.get("POSTGRES_DB", "lipton-sku-classifier"),
    user=os.environ.get("POSTGRES_USER", "ans.shahzad@vaival.tech"),
    password=os.environ.get("POSTGRES_PASSWORD", ""),
    cursor_factory=psycopg2.extras.RealDictCursor,
)


def _connect():
    conn = _POOL.getconn()
    register_vector(conn)
    return conn


def _release(conn):
    _POOL.putconn(conn)


def init_db():
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            image_width INTEGER NOT NULL,
            image_height INTEGER NOT NULL,
            verified_count INTEGER NOT NULL,
            unknown_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS images_status_idx ON images(status);
        CREATE INDEX IF NOT EXISTS images_created_at_idx ON images(created_at DESC);

        CREATE TABLE IF NOT EXISTS image_files (
            image_id TEXT PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
            content_type TEXT NOT NULL,
            data BYTEA NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            class_name TEXT,
            decision TEXT NOT NULL,
            score REAL NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS detections_image_id_idx ON detections(image_id);

        CREATE TABLE IF NOT EXISTS reference_embeddings (
            id SERIAL PRIMARY KEY,
            class_name TEXT NOT NULL,
            embedding VECTOR(768) NOT NULL
        );
        """
    )
    conn.commit()
    _release(conn)


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


def save_image_result(image_id, filename, image_bytes, content_type, image_width, image_height, detections):
    """Persists one processed shelf image (bytes + metadata) and its detections.

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
    created_at = datetime.now(timezone.utc)

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO images
           (id, filename, status, image_width, image_height,
            verified_count, unknown_count, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (image_id, filename, status, image_width, image_height,
         verified_count, unknown_count, created_at),
    )
    cur.execute(
        """INSERT INTO image_files (image_id, content_type, data)
           VALUES (%s, %s, %s)""",
        (image_id, content_type, psycopg2.Binary(image_bytes)),
    )
    psycopg2.extras.execute_values(
        cur,
        """INSERT INTO detections
           (image_id, class_name, decision, score, x1, y1, x2, y2)
           VALUES %s""",
        [
            (image_id, d["matched_class"], d["decision"], d["score"], *d["bbox"])
            for d in detections
        ],
    )
    conn.commit()
    _release(conn)
    return get_image(image_id)


def _row_to_summary(row):
    return {
        "id": row["id"],
        "name": row["filename"],
        "status": row["status"],
        "captured": row["created_at"].strftime("Uploaded %Y-%m-%d %H:%M"),
        "image_url": f"{PUBLIC_BASE_URL}/api/images/{row['id']}/file",
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "verified_count": row["verified_count"],
        "unknown_count": row["unknown_count"],
        "created_at": row["created_at"].isoformat(),
    }


def _detections_for(conn, image_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT class_name, decision, score, x1, y1, x2, y2 FROM detections WHERE image_id = %s ORDER BY id",
        (image_id,),
    )
    rows = cur.fetchall()
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


def list_images(status=None):
    conn = _connect()
    cur = conn.cursor()
    if status:
        cur.execute(
            "SELECT * FROM images WHERE status = %s ORDER BY created_at DESC", (status,)
        )
    else:
        cur.execute("SELECT * FROM images ORDER BY created_at DESC")
    rows = cur.fetchall()

    out = []
    for row in rows:
        summary = _row_to_summary(row)
        summary["detections"] = _detections_for(conn, row["id"])
        out.append(summary)
    _release(conn)
    return out


def get_image(image_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE id = %s", (image_id,))
    row = cur.fetchone()
    if row is None:
        _release(conn)
        return None
    summary = _row_to_summary(row)
    summary["detections"] = _detections_for(conn, image_id)
    _release(conn)
    return summary


def get_image_file(image_id):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT content_type, data FROM image_files WHERE image_id = %s", (image_id,))
    row = cur.fetchone()
    _release(conn)
    if row is None:
        return None
    return {"content_type": row["content_type"], "data": bytes(row["data"])}


def dashboard_data():
    """Aggregation ported from the mock Dashboard.jsx page, now computed
    over real persisted images/detections instead of the 6-item mock
    catalog."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE status = 'approved' ORDER BY created_at DESC")
    approved_rows = cur.fetchall()
    cur.execute("SELECT * FROM images WHERE status = 'rejected'")
    rejected_rows = cur.fetchall()

    classified = sum(r["verified_count"] + r["unknown_count"] for r in approved_rows)
    verified = sum(r["verified_count"] for r in approved_rows)
    unknown_in_uploads = sum(r["unknown_count"] for r in approved_rows)
    unknown_in_rejected = sum(r["unknown_count"] for r in rejected_rows)

    per_class = {}
    for r in approved_rows:
        cur.execute(
            "SELECT class_name FROM detections WHERE image_id = %s AND decision = 'EXACT_SKU'",
            (r["id"],),
        )
        for d in cur.fetchall():
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

    _release(conn)
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


def match_reference_embeddings(embedding):
    """Per-class best cosine match against the reference gallery, scored
    0-100 like the pre-Postgres brute-force Python loop. Returns
    [(class_name, score), ...] sorted best first."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """SELECT class_name, MAX(1 - (embedding <=> %s)) * 100.0 AS score
           FROM reference_embeddings
           GROUP BY class_name
           ORDER BY score DESC""",
        (embedding,),
    )
    rows = cur.fetchall()
    _release(conn)
    return [(r["class_name"], float(r["score"])) for r in rows]


def bulk_insert_reference_embeddings(rows):
    """rows: iterable of (class_name, embedding_list). Used by the
    one-time backend/scripts/migrate_embeddings.py import; truncates first
    so it's safe to re-run."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("TRUNCATE reference_embeddings")
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO reference_embeddings (class_name, embedding) VALUES %s",
        list(rows),
    )
    conn.commit()
    _release(conn)
