"""
Permanent per-image instrumentation for the detect->embed->gate->classify->
OCR->decide cascade. Every number here goes to the terminal (stdout), never
into an API response -- the frontend has no idea this module exists.

Usage: one Timings() per uploaded image, threaded through pipeline.py's
call chain, then logged once (log_timings) after main.py's DB save
completes so "total" reflects the full request, not just the ML part.
"""

import logging
import os
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger("pipeline.timing")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _cuda_sync():
    # No-op on CPU-only boxes (torch.cuda.is_available() is cheap to call
    # repeatedly). On a GPU box, this is the whole point: CUDA calls return
    # before the kernel finishes, so a wall-clock timestamp taken without
    # syncing first measures "how long it took to queue the work", not "how
    # long the work took".
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Buckets match the stages in the engineering guide's cascade, split where
# the guide's "gallery load" + "cosine similarity" collapsed into a single
# pgvector query (storage.match_reference_embeddings) once embeddings moved
# into Postgres -- there's no separate in-process gallery/cosine step left
# to time apart, so both are reported under "match".
FIELDS = (
    "detect",
    "crop_prep",
    "embed",
    "gate",
    "match",
    "ocr_bicubic_en",
    "ocr_clahe_en",
    "ocr_bicubic_ar",
    "ocr_clahe_ar",
    "decision",
    "db_save",
)


class Timings:
    __slots__ = ("seconds", "n_detections", "m_ocr_attempts", "total", "modes")

    def __init__(self):
        self.seconds = {f: 0.0 for f in FIELDS}
        self.n_detections = 0       # N -- boxes YOLO found
        self.m_ocr_attempts = 0     # M -- crops that made it past GATE+CLASSIFY into OCR
        self.total = 0.0            # wall-clock for the whole request, set by main.py
        self.modes = {}             # stage -> plain-English description of what actually ran

    @contextmanager
    def measure(self, key):
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            self.seconds[key] += time.perf_counter() - t0

    def add(self, key, seconds):
        self.seconds[key] += seconds


# Spelled out on purpose: this report gets copy-pasted to people who have
# never opened pipeline.py, so a label has to carry its own meaning --
# "match" means nothing to them, "Stage 4 Classify (cosine match vs
# reference gallery)" does.
_LABELS = {
    "detect":         "Stage 1  Detect            (YOLO object detection)",
    "crop_prep":      "Stage 1  Crop prep         (EXIF fix, pad, JPEG encode)",
    "embed":          "Stage 2  Embed             (DINOv3 feature extraction)",
    "gate":           "Stage 3  Gate              (reject gate + BP target gate)",
    "match":          "Stage 4  Classify          (SKU head, or cosine match if absent)",
    "ocr_bicubic_en": "Stage 5  Verify - OCR      (bicubic image: detect once + read English)",
    "ocr_clahe_en":   "Stage 5  Verify - OCR      (CLAHE image: detect once + read English)",
    "ocr_bicubic_ar": "Stage 5  Verify - OCR      (bicubic image: read Arabic, reuses detection above)",
    "ocr_clahe_ar":   "Stage 5  Verify - OCR      (CLAHE image: read Arabic, reuses detection above)",
    "decision":       "Stage 6  Decide            (keyword match + pack-size check)",
    "db_save":        "         Database save     (Postgres insert)",
}
_LABEL_WIDTH = max(len(v) for v in _LABELS.values()) + 2
_RULE = "-" * (_LABEL_WIDTH + 10)

# Print order for the "what actually ran" summary -- one entry per guide
# stage, independent of the finer-grained timing buckets above.
STAGE_ORDER = ("detect", "embed", "gate", "classify", "verify", "decide")
_STAGE_TITLE = {
    "detect": "Stage 1  Detect  ",
    "embed": "Stage 2  Embed   ",
    "gate": "Stage 3  Gate    ",
    "classify": "Stage 4  Classify",
    "verify": "Stage 5  Verify  ",
    "decide": "Stage 6  Decide  ",
}


def log_timings(image_id, t: Timings):
    width = _LABEL_WIDTH + 10
    lines = [
        "=" * width,
        f"PIPELINE TIMING REPORT -- image {image_id}",
        "=" * width,
        f"{'Detections found by the YOLO model (N)':<{_LABEL_WIDTH}}{t.n_detections:>6d}",
        f"{'Crops that reached OCR verification (M)':<{_LABEL_WIDTH}}{t.m_ocr_attempts:>6d}",
        f"{'Total wall-clock time for this image':<{_LABEL_WIDTH}}{t.total:>6.3f}s",
    ]

    if t.modes:
        lines.append(_RULE)
        lines.append("WHAT EACH STAGE ACTUALLY USED -- gate/trained model, or fallback logic")
        for key in STAGE_ORDER:
            if key in t.modes:
                lines.append(f"{_STAGE_TITLE[key]}  {t.modes[key]}")

    lines.append(_RULE)
    for key in FIELDS:
        lines.append(f"{_LABELS[key]:<{_LABEL_WIDTH}}{t.seconds[key]:>6.3f}s")
    accounted = sum(t.seconds.values())
    lines.append(_RULE)
    lines.append(
        f"{'Unaccounted (cold-start model load, queue wait, etc.)':<{_LABEL_WIDTH}}"
        f"{max(0.0, t.total - accounted):>6.3f}s"
    )
    lines.append("=" * width)
    logger.info("\n".join(lines))


def log_startup_info(
    device,
    ocr_device,
    detector_weights_path,
    reject_gate_active=None,
    bp_gate_active=None,
    sku_head_active=None,
    scoring_mode=None,
):
    width = 42
    lines = [
        "=" * (width + 20),
        "PIPELINE STARTUP -- device & environment",
        "=" * (width + 20),
        f"{'Detection / embedding models run on':<{width}}: {device}",
        f"{'OCR models (PaddleOCR) run on':<{width}}: {ocr_device}",
        f"{'CUDA (GPU) available':<{width}}: {torch.cuda.is_available()}",
        f"{'CPU cores available (nproc)':<{width}}: {os.cpu_count()}",
        f"{'PyTorch CPU thread pool size':<{width}}: {torch.get_num_threads()}",
        f"{'Detector weights file in use':<{width}}: {detector_weights_path}",
        f"{'Detector weights file found on disk':<{width}}: {os.path.isfile(detector_weights_path)}",
    ]
    if scoring_mode is not None:
        lines += [
            "-" * (width + 20),
            f"{'Reject gate (drop unusable crops)':<{width}}: {'ACTIVE' if reject_gate_active else 'inactive (file missing)'}",
            f"{'BP target gate (drop non-target crops)':<{width}}: {'ACTIVE' if bp_gate_active else 'inactive (file missing)'}",
            f"{'32-class SKU head (trained classifier)':<{width}}: {'ACTIVE' if sku_head_active else 'inactive (file missing)'}",
            f"{'Classification mode this run':<{width}}: {scoring_mode}",
        ]
    lines.append("=" * (width + 20))
    logger.info("\n".join(lines))
