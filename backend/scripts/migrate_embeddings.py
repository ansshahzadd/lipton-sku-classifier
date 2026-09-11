"""
One-time import of backend/data/embeddings.json (JSON-lines reference
gallery, {"class", "embedding"} per line) into the Postgres
reference_embeddings table. Only needed once per database -- the running
app reads reference embeddings from Postgres, not this file, from here on.

Safe to re-run: truncates reference_embeddings before inserting.

Usage (from backend/):
    python -m scripts.migrate_embeddings
"""

import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import storage  # noqa: E402

EMBEDDINGS_PATH = os.path.join(BACKEND_DIR, "data", "embeddings.json")


def main():
    if not os.path.isfile(EMBEDDINGS_PATH):
        raise SystemExit(f"{EMBEDDINGS_PATH} not found.")

    rows = []
    with open(EMBEDDINGS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                rows.append((item["class"], item["embedding"]))

    if not rows:
        raise SystemExit(f"{EMBEDDINGS_PATH} contained no rows -- nothing to import.")

    storage.init_db()
    storage.bulk_insert_reference_embeddings(rows)
    print(f"Imported {len(rows)} reference embeddings into Postgres.")


if __name__ == "__main__":
    main()
