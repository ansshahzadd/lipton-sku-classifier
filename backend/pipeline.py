"""
Detect -> embed -> gate -> classify -> OCR -> decide pipeline for the BP
shelf SKU cascade, ported from the Streamlit app.py so the FastAPI layer in
main.py can call it directly without a Streamlit runtime.

Every threshold, model id, and crop-geometry constant below is copied
verbatim from the validated app.py / Brand Partners Engineering Guide. Do
not change one of these without the guide's frozen-baseline + strict-VAL
regression process (guide section 05) -- that discipline doesn't stop
applying just because this runs behind an API instead of Streamlit.

Model weights and reference data are NOT included in this repo. Drop them
into backend/models/ and backend/data/ (see backend/README.md for the exact
filenames expected) before calling process_shelf_image().
"""

import functools
import json
import os
import re
import unicodedata
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

import storage
import timing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")



# =============================================================================
# CONFIG -- every value below is a validated setting from the guide.
# =============================================================================

# ---- Stage 1: DETECT -------------------------------------------------------
# Accepts either filename the guide/your training runs have used.
_DETECTOR_CANDIDATES = ["product_detector_best.pt", "best_augmented.pt"]
WEIGHTS_PATH = next(
    (os.path.join(MODELS_DIR, f) for f in _DETECTOR_CANDIDATES
     if os.path.isfile(os.path.join(MODELS_DIR, f))),
    os.path.join(MODELS_DIR, _DETECTOR_CANDIDATES[0]),
)
DET_IMGSZ = 1280
DET_CONF = 0.20
DET_IOU = 0.70
DET_MAX_DET = 1000

# ---- Stage 2: EMBED --------------------------------------------------------
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
EMBED_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Crop geometry ("exact recognition crop parity") -----------------------
PAD_FRAC = 0.03
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0

# ---- Stage 3: GATE ---------------------------------------------------------
REJECT_GATE_PATH = os.path.join(MODELS_DIR, "REJECT_GATE_DINOV3_MLP_V3.pt")
REJECT_GATE_THRESHOLD = 0.279
BP_TARGET_GATE_PATH = os.path.join(MODELS_DIR, "BP_TARGET_GATE_DINOV3_LOGREG.joblib")
BP_TARGET_GATE_THRESHOLD = 0.85

# ---- Stage 4: CLASSIFY -----------------------------------------------------
SKU_HEAD_PATH = os.path.join(MODELS_DIR, "SKU_32CLASS_DINOV3_MLP.pt")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog_32class.txt")
EXACT_SKU_THRESHOLD = 0.875
MARGIN_FLOOR = 0.0

# ---- COSINE FALLBACK (NOT a guide setting) --------------------------------
COSINE_FALLBACK_THRESHOLD = 75.0
COSINE_FALLBACK_MARGIN = 0.0

# ---- Stage 5: VERIFY -------------------------------------------------------
OCR_DET_MODEL_NAME = "PP-OCRv5_server_det"
OCR_REC_MODEL_NAME_EN = "en_PP-OCRv5_mobile_rec"
OCR_REC_MODEL_NAME_AR = "arabic_PP-OCRv5_mobile_rec"

OCR_DEVICE = os.environ.get("OCR_DEVICE", "cpu")

OCR_ENABLE_MKLDNN = False
if not OCR_ENABLE_MKLDNN:
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

KEYWORDS_PATH = os.path.join(DATA_DIR, "class_keywords.json")

BBRL_CLASSES = {"bbrl-100", "bbrl-400g"}
GREEN_TEA_CLASSES = {
    "lipton-green-tea-lemon-25", "lipton-green-tea-lemon-50", "lipton-green-tea-lemon-100",
    "lipton-green-tea-mint-25", "lipton-green-tea-mint-50", "lipton-green-tea-mint-100",
    "lipton-green-tea-pure-25", "lipton-green-tea-pure-50", "lipton-green-tea-pure-100",
}
GREEN_TEA_MARKERS = ["green tea", "greentea"]

_CLASS_SIZE_SUFFIX = re.compile(r"-\d+g?$")


def _class_family(class_name):
    return _CLASS_SIZE_SUFFIX.sub("", class_name)


LIPTON_KEYWORD_AR = "ليبتون"
GREEN_TEA_MARKER_WORDS_AR = ["شاي", "اخضر"]

# EXPERIMENT: OCR_UPSCALE removed (was 2.0) to test its effect on OCR
# latency -- _to_bicubic_rgb/_to_clahe_gray no longer resize at all.
# Reinstate `upscale=2.0` on both functions' resize calls to restore it.

# =============================================================================
# Decision types (guide stage 6)
# =============================================================================
EXACT_SKU = "EXACT_SKU"
UNKNOWN = "UNKNOWN"
REJECT = "REJECT"
NON_TARGET = "NON_TARGET"


# =============================================================================
# CROP GEOMETRY
# =============================================================================

def load_image_exif_corrected(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def padded_box(x1, y1, x2, y2, image_width, image_height, pad_frac=PAD_FRAC):
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    px1 = max(0.0, x1 - pad_x)
    py1 = max(0.0, y1 - pad_y)
    px2 = min(float(image_width), x2 + pad_x)
    py2 = min(float(image_height), y2 + pad_y)
    return px1, py1, px2, py2


def make_recognition_crop(pil_image, x1, y1, x2, y2):
    px1, py1, px2, py2 = padded_box(x1, y1, x2, y2, pil_image.width, pil_image.height)
    crop = pil_image.crop((px1, py1, px2, py2))
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# =============================================================================
# CACHED LOADERS (functools.lru_cache stands in for Streamlit's
# st.cache_resource -- both just mean "build this once per process").
# =============================================================================

@functools.lru_cache(maxsize=1)
def log_startup_info():
    timing.log_startup_info(
        DEVICE,
        OCR_DEVICE,
        WEIGHTS_PATH,
        reject_gate_active=load_reject_gate() is not None,
        bp_gate_active=load_bp_target_gate() is not None,
        sku_head_active=load_sku_head() is not None,
        scoring_mode="sku_head" if load_sku_head() is not None else "cosine_fallback",
    )


def load_yolo_model():
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Detector weights not found at {WEIGHTS_PATH}. Drop "
            f"product_detector_best.pt (or best_augmented.pt) into {MODELS_DIR}."
        )
    return YOLO(WEIGHTS_PATH)


@functools.lru_cache(maxsize=1)
def load_dinov3_model():
    processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
    model = AutoModel.from_pretrained(DINOV3_MODEL_ID).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@functools.lru_cache(maxsize=1)
def load_catalog():
    if not os.path.isfile(CATALOG_PATH):
        return None
    with open(CATALOG_PATH) as f:
        return tuple(line.strip() for line in f if line.strip())


@functools.lru_cache(maxsize=1)
def load_reject_gate():
    if not os.path.isfile(REJECT_GATE_PATH):
        return None
    model = torch.load(REJECT_GATE_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def load_bp_target_gate():
    if not os.path.isfile(BP_TARGET_GATE_PATH):
        return None
    import joblib
    return joblib.load(BP_TARGET_GATE_PATH)


@functools.lru_cache(maxsize=1)
def load_sku_head():
    if not os.path.isfile(SKU_HEAD_PATH):
        return None
    model = torch.load(SKU_HEAD_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def load_keywords():
    keywords = {}
    if os.path.isfile(KEYWORDS_PATH):
        with open(KEYWORDS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                normalized = []
                for k in item["keywords"]:
                    nk = _normalize_ocr_text(k)
                    if nk and nk not in normalized:
                        normalized.append(nk)
                keywords[item["class"]] = normalized
    return keywords


@functools.lru_cache(maxsize=1)
def load_keywords_ar():
    keywords_ar = {}
    if os.path.isfile(KEYWORDS_PATH):
        with open(KEYWORDS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                normalized = []
                for k in item.get("keywords_ar", []):
                    nk = _normalize_ocr_text(k)
                    if nk and nk not in normalized:
                        normalized.append(nk)
                keywords_ar[item["class"]] = normalized
    return keywords_ar


@functools.lru_cache(maxsize=1)
def load_size_specs():
    specs = {}
    if os.path.isfile(KEYWORDS_PATH):
        with open(KEYWORDS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "size" in item and "size_type" in item:
                    specs[item["class"]] = (item["size"], item["size_type"])
    return specs


# Detection (where is there text) and recognition (what does it say) are two
# separate models. The English and Arabic passes only ever differ in
# recognition -- both use the exact same detector (OCR_DET_MODEL_NAME) -- so
# detection is run ONCE per preprocessing variant (bicubic/CLAHE) and its
# boxes are reused for both languages' recognition, instead of re-detecting
# from scratch for each language like the old fused-PaddleOCR-per-language
# setup did. Verified to produce byte-identical recognized text to the old
# approach on a test crop before this was wired in.
@functools.lru_cache(maxsize=1)
def load_ocr_detector():
    from paddleocr import TextDetection

    kwargs = dict(model_name=OCR_DET_MODEL_NAME, device=OCR_DEVICE)
    try:
        return TextDetection(enable_mkldnn=OCR_ENABLE_MKLDNN, **kwargs)
    except TypeError:
        return TextDetection(**kwargs)


def _build_ocr_recognizer(rec_model_name):
    from paddleocr import TextRecognition

    kwargs = dict(model_name=rec_model_name, device=OCR_DEVICE)
    try:
        return TextRecognition(enable_mkldnn=OCR_ENABLE_MKLDNN, **kwargs)
    except TypeError:
        return TextRecognition(**kwargs)


@functools.lru_cache(maxsize=1)
def load_ocr_model_en():
    return _build_ocr_recognizer(OCR_REC_MODEL_NAME_EN)


@functools.lru_cache(maxsize=1)
def load_ocr_model_ar():
    return _build_ocr_recognizer(OCR_REC_MODEL_NAME_AR)


# =============================================================================
# STAGE 1 -- DETECT
# =============================================================================

def _next_filename(folder, prefix, ext):
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(ext)]
    numbers = []
    for f in existing:
        name_part = f[len(prefix):-len(ext)]
        if name_part.isdigit():
            numbers.append(int(name_part))
    next_number = max(numbers, default=0) + 1
    return f"{prefix}{next_number:06d}{ext}"


def run_detection(image_bytes, timings):
    """Returns (crop_imgs, crop_boxes, image_width, image_height).
    crop_boxes are the RAW detector boxes (unpadded); the padded versions
    are used for the actual in-memory recognition crops (crop_imgs) --
    those are never written to disk, just fed straight into embed_crop/
    ocr_verify for this one request."""
    model = load_yolo_model()

    # Detect on the SAME image the crops are cut from. Boxes from a
    # differently-oriented source (e.g. the raw, un-rotated bytes) aren't
    # guaranteed to be valid coordinates once EXIF rotation swaps width and
    # height -- for a 90/270-degree-rotated photo this produced an inverted
    # crop box and crashed PIL's crop(). No effect on the common case (no
    # meaningful EXIF rotation): raw and corrected images are identical then.
    pil_image = load_image_exif_corrected(image_bytes)

    with timings.measure("detect"):
        results = model.predict(
            source=pil_image,
            imgsz=DET_IMGSZ,
            conf=DET_CONF,
            iou=DET_IOU,
            max_det=DET_MAX_DET,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()

    with timings.measure("crop_prep"):
        crop_imgs, crop_boxes = [], []

        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            px1, py1, px2, py2 = padded_box(x1, y1, x2, y2, pil_image.width, pil_image.height)
            if px2 <= px1 or py2 <= py1:
                # Defensive: skip a degenerate box rather than crash. Should
                # no longer be reachable now that detection and cropping
                # share the same image, but a corrupt/zero-area detector box
                # is still a theoretical possibility worth not crashing on.
                continue
            crop_imgs.append(make_recognition_crop(pil_image, x1, y1, x2, y2))
            crop_boxes.append(tuple(int(v) for v in (x1, y1, x2, y2)))

    timings.n_detections = len(crop_imgs)
    return crop_imgs, crop_boxes, pil_image.width, pil_image.height


# =============================================================================
# STAGE 2 -- EMBED
# =============================================================================

def embed_crop(crop_img, processor, model):
    img = crop_img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    cls = output.last_hidden_state[0][0].cpu().numpy().astype(np.float64)

    if cls.shape[0] != EMBED_DIM:
        raise ValueError(
            f"Backbone produced {cls.shape[0]}-D embedding, expected {EMBED_DIM}-D."
        )

    norm = np.linalg.norm(cls)
    if norm > 0:
        cls = cls / norm
    return cls


# =============================================================================
# STAGE 3 -- GATE
# =============================================================================

def reject_gate_score(embedding, gate):
    if gate is None:
        return None
    with torch.no_grad():
        x = torch.from_numpy(embedding).float().unsqueeze(0).to(DEVICE)
        logit = gate(x)
        return float(torch.sigmoid(logit).squeeze().item())


def bp_target_gate_score(embedding, gate):
    if gate is None:
        return None
    return float(gate.predict_proba(embedding.reshape(1, -1))[0, 1])


# =============================================================================
# STAGE 4 -- CLASSIFY
# =============================================================================

def sku_head_predict(embedding, head, catalog):
    with torch.no_grad():
        x = torch.from_numpy(embedding).float().unsqueeze(0).to(DEVICE)
        logits = head(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    order = np.argsort(probs)[::-1]
    top_idx, second_idx = order[0], order[1]
    ranked = [(catalog[i], float(probs[i])) for i in order]
    return (
        catalog[top_idx],
        float(probs[top_idx]),
        float(probs[top_idx] - probs[second_idx]),
        ranked,
    )


def cosine_fallback_predict(embedding):
    ranked = storage.match_reference_embeddings(embedding)
    if not ranked:
        return None, 0.0, 0.0, []

    top_class, top_score = ranked[0]
    margin = top_score - ranked[1][1] if len(ranked) > 1 else top_score
    return top_class, top_score, margin, ranked


# =============================================================================
# STAGE 5 -- VERIFY (OCR)
# =============================================================================

_DIGIT_LETTER_BOUNDARY = re.compile(r"(?<=[0-9])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[0-9])")
_ARABIC_TATWEEL = "ـ"


def _strip_diacritics(text):
    decomposed = unicodedata.normalize("NFKD", text)
    no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return no_marks.replace(_ARABIC_TATWEEL, "")


def _normalize_ocr_text(text):
    spaced = _DIGIT_LETTER_BOUNDARY.sub(" ", text)
    no_accents = _strip_diacritics(spaced)
    return re.sub(r"\s+", " ", no_accents.strip().lower())


def _to_bicubic_rgb(pil_crop):
    arr = np.array(pil_crop.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _to_clahe_gray(pil_crop):
    gray = np.array(pil_crop.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# The next three helpers replace the fused PaddleOCR().predict() call. They
# reproduce its internal detect -> sort boxes -> perspective-crop ->
# recognize steps by hand so detection can be shared across languages;
# ported directly from paddlex's own SortQuadBoxes / CropByPolys operators
# (paddlex/inference/pipelines/components/common/{sort_boxes,crop_image_regions}.py)
# rather than importing those internal classes, and verified to produce
# byte-identical text to the old fused-pipeline call before being wired in.

def _sort_quad_boxes(dt_polys):
    boxes = list(sorted(dt_polys, key=lambda box: (box[0][1], box[0][0])))
    n = len(boxes)
    for i in range(n - 1):
        for j in range(i, -1, -1):
            if (abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10
                    and boxes[j + 1][0][0] < boxes[j][0][0]):
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def _crop_quad_box(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    pts = sorted(list(cv2.boxPoints(bounding_box)), key=lambda p: p[0])
    if pts[1][1] > pts[0][1]:
        index_a, index_d = 0, 1
    else:
        index_a, index_d = 1, 0
    if pts[3][1] > pts[2][1]:
        index_b, index_c = 2, 3
    else:
        index_b, index_c = 3, 2
    box = np.array([pts[index_a], pts[index_b], pts[index_c], pts[index_d]], dtype=np.float32)

    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    transform = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(
        img, transform, (width, height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    if crop.shape[0] / max(crop.shape[1], 1) >= 1.5:
        crop = np.rot90(crop)
    return crop


def _detect_text_boxes(array):
    detector = load_ocr_detector()
    result = list(detector.predict(array))[0]
    polys = _sort_quad_boxes(list(result["dt_polys"]))
    return [_crop_quad_box(array, poly) for poly in polys]


def _recognize_text(sub_images, recognizer):
    if not sub_images:
        return ""
    text_parts = [r["rec_text"] for r in recognizer.predict(sub_images) if r.get("rec_text")]
    return _normalize_ocr_text(" ".join(text_parts))


# EXPERIMENT (revert to `and` to restore two-variant consensus): a keyword
# counts if EITHER OCR preprocessing variant (bicubic-RGB or CLAHE-gray)
# read it, not just when both agree.
def _kw_in_consensus(keyword, text_a, text_b):
    kw = keyword.lower()
    return kw in text_a or kw in text_b


def _any_kw_in_consensus(keywords, text_a, text_b):
    return any(_kw_in_consensus(kw, text_a, text_b) for kw in keywords)


def _all_kw_in_consensus(keywords, text_a, text_b):
    return all(_kw_in_consensus(kw, text_a, text_b) for kw in keywords)


def _first_matching_kw(keywords, text_a, text_b):
    for kw in keywords:
        if _kw_in_consensus(kw, text_a, text_b):
            return kw
    return None


def _is_unique_to_family(keyword, owner_class, keywords, keywords_ar):
    owner_family = _class_family(owner_class)
    for cls, kws in keywords.items():
        if _class_family(cls) != owner_family and keyword in kws:
            return False
    for cls, kws in keywords_ar.items():
        if _class_family(cls) != owner_family and keyword in kws:
            return False
    return True


def _find_all_indices(text, sub):
    idx = text.find(sub)
    while idx != -1:
        yield idx
        idx = text.find(sub, idx + 1)


def _size_token_present(number, size_type, text):
    for idx in _find_all_indices(text, number):
        after = text[idx + len(number):idx + len(number) + 2].lstrip()
        followed_by_g = after.startswith("g")
        if size_type == "weight" and followed_by_g:
            return True
        if size_type == "count" and not followed_by_g:
            return True
    return False


# EXPERIMENT (revert to `and` to restore two-variant consensus): see
# _kw_in_consensus above.
def _size_confirmed_consensus(number, size_type, text_a, text_b):
    return (_size_token_present(number, size_type, text_a)
            or _size_token_present(number, size_type, text_b))


def _size_confirmed(class_name, text_a, text_b, ar_a, ar_b, size_specs):
    spec = size_specs.get(class_name)
    if spec is None:
        return True
    number, size_type = spec
    return (_size_confirmed_consensus(number, size_type, text_a, text_b)
            or _size_confirmed_consensus(number, size_type, ar_a, ar_b))


def _resolve_size_variant(flavor_class, text_a, text_b, ar_a, ar_b, size_specs, all_classes):
    family = _class_family(flavor_class)
    siblings = [c for c in all_classes if _class_family(c) == family]
    confirmed = [
        c for c in siblings
        if _size_confirmed(c, text_a, text_b, ar_a, ar_b, size_specs)
    ]
    return confirmed[0] if len(confirmed) == 1 else None


def _best_keyword_match_consensus(text_a, text_b, keywords, exclude_classes=()):
    best_class, best_hits = None, 0
    for class_name, class_keywords in keywords.items():
        if class_name in exclude_classes:
            continue
        hits = sum(1 for kw in class_keywords if _kw_in_consensus(kw, text_a, text_b))
        if hits > best_hits:
            best_hits, best_class = hits, class_name
    return best_class


def ocr_verify(crop_img, top_class, ocr_model_en, ocr_model_ar, keywords, keywords_ar, size_specs, timings):
    pil_crop = crop_img.convert("RGB")

    # Detection runs once per preprocessing variant and is shared by both
    # languages' recognition below -- timed under the EN buckets since that's
    # the first consumer; the AR buckets that follow only pay for
    # recognition, reusing these same boxes.
    with timings.measure("ocr_bicubic_en"):
        bicubic_boxes = _detect_text_boxes(_to_bicubic_rgb(pil_crop))
        text_a = _recognize_text(bicubic_boxes, ocr_model_en)

    with timings.measure("ocr_clahe_en"):
        clahe_boxes = _detect_text_boxes(_to_clahe_gray(pil_crop))
        text_b = _recognize_text(clahe_boxes, ocr_model_en)

    with timings.measure("ocr_bicubic_ar"):
        ar_a = _recognize_text(bicubic_boxes, ocr_model_ar)

    with timings.measure("ocr_clahe_ar"):
        ar_b = _recognize_text(clahe_boxes, ocr_model_ar)

    with timings.measure("decision"):
        return _ocr_decide(top_class, text_a, text_b, ar_a, ar_b, keywords, keywords_ar, size_specs)


def _ocr_decide(top_class, text_a, text_b, ar_a, ar_b, keywords, keywords_ar, size_specs):
    all_classes = list(keywords.keys())

    def _finalize(flavor_class):
        resolved = _resolve_size_variant(
            flavor_class, text_a, text_b, ar_a, ar_b, size_specs, all_classes
        )
        return resolved, text_a, text_b, ar_a, ar_b

    for bbrl_class in BBRL_CLASSES:
        bbrl_kw = keywords.get(bbrl_class, [])
        bbrl_kw_ar = keywords_ar.get(bbrl_class, [])
        if _any_kw_in_consensus(bbrl_kw, text_a, text_b) or _any_kw_in_consensus(bbrl_kw_ar, ar_a, ar_b):
            return _finalize(bbrl_class)

    brand_en = _kw_in_consensus("lipton", text_a, text_b)
    brand_ar = _kw_in_consensus(LIPTON_KEYWORD_AR, ar_a, ar_b)
    if not (brand_en or brand_ar):
        return None, text_a, text_b, ar_a, ar_b

    if top_class in GREEN_TEA_CLASSES:
        family_en = _any_kw_in_consensus(GREEN_TEA_MARKERS, text_a, text_b)
        family_ar = _all_kw_in_consensus(GREEN_TEA_MARKER_WORDS_AR, ar_a, ar_b)
        flavor_keywords = [
            kw for kw in keywords.get(top_class, [])
            if kw not in GREEN_TEA_MARKERS
        ]
        flavor_keywords_ar = keywords_ar.get(top_class, [])
        matched_flavor_en = _first_matching_kw(flavor_keywords, text_a, text_b)
        matched_flavor_ar = _first_matching_kw(flavor_keywords_ar, ar_a, ar_b)

        if family_en or family_ar:
            if matched_flavor_en or matched_flavor_ar:
                return _finalize(top_class)
            return None, text_a, text_b, ar_a, ar_b

        for matched in (matched_flavor_ar, matched_flavor_en):
            if matched and _is_unique_to_family(matched, top_class, keywords, keywords_ar):
                return _finalize(top_class)

        exclude = GREEN_TEA_CLASSES | BBRL_CLASSES
        fallback = _best_keyword_match_consensus(text_a, text_b, keywords, exclude_classes=exclude)
        if fallback is None:
            return None, text_a, text_b, ar_a, ar_b
        return _finalize(fallback)

    class_keywords = keywords.get(top_class, [])
    class_keywords_ar = keywords_ar.get(top_class, [])
    matched_en = _first_matching_kw(class_keywords, text_a, text_b)
    matched_ar = _first_matching_kw(class_keywords_ar, ar_a, ar_b)
    if matched_en or matched_ar:
        # A non-green-tea class can share a flavor word with a green-tea
        # sibling (e.g. "mint" is a keyword on both lipton-pepermint and
        # lipton-green-tea-mint-*). If the box also shows a green-tea
        # marker, that shared word doesn't actually confirm top_class --
        # it's more likely the crop is really the green-tea sibling that
        # the visual model mis-ranked. Don't confirm on the shared word
        # alone in that case; fall through to the class-search below
        # (which excludes green-tea classes, so it resolves to whichever
        # plain class fits, or UNKNOWN).
        looks_like_green_tea = (
            _any_kw_in_consensus(GREEN_TEA_MARKERS, text_a, text_b)
            or _all_kw_in_consensus(GREEN_TEA_MARKER_WORDS_AR, ar_a, ar_b)
        )
        matched = matched_en or matched_ar
        shared_with_other_family = not _is_unique_to_family(matched, top_class, keywords, keywords_ar)
        if not (looks_like_green_tea and shared_with_other_family):
            return _finalize(top_class)

    exclude = GREEN_TEA_CLASSES | BBRL_CLASSES
    fallback = _best_keyword_match_consensus(text_a, text_b, keywords, exclude_classes=exclude)
    if fallback is None:
        return None, text_a, text_b, ar_a, ar_b
    return _finalize(fallback)


# =============================================================================
# STAGE 6 -- DECIDE
# =============================================================================

def classify_crop(crop_img, processor, dino_model, catalog,
                   reject_gate, bp_gate, sku_head, ocr_model_en, ocr_model_ar,
                   keywords, keywords_ar, size_specs, timings):
    row = {
        "decision": None,
        "matched_class": None,
        "top_class": None,
        "top_score": 0.0,
        "margin": 0.0,
        "scoring_mode": "sku_head" if sku_head is not None else "cosine_fallback",
        # Per-stage diagnostic trail for the frontend's "why was this crop
        # classified this way" sidebar -- one entry per stage the crop
        # actually reached, in cascade order, so the UI can render every
        # stage and mark whichever ones never ran as "not reached" once the
        # cascade stopped earlier.
        "steps": [],
    }

    with timings.measure("embed"):
        embedding = embed_crop(crop_img, processor, dino_model)

    with timings.measure("gate"):
        valid_prob = reject_gate_score(embedding, reject_gate)
    reject_active = reject_gate is not None
    reject_passed = valid_prob >= REJECT_GATE_THRESHOLD if reject_active else None
    row["steps"].append({
        "stage": "reject_gate",
        "active": reject_active,
        "passed": reject_passed,
        "score": valid_prob,
        "threshold": REJECT_GATE_THRESHOLD if reject_active else None,
    })
    if reject_active and not reject_passed:
        row["decision"] = REJECT
        return row

    with timings.measure("gate"):
        bp_prob = bp_target_gate_score(embedding, bp_gate)
    bp_active = bp_gate is not None
    bp_passed = bp_prob >= BP_TARGET_GATE_THRESHOLD if bp_active else None
    row["steps"].append({
        "stage": "bp_gate",
        "active": bp_active,
        "passed": bp_passed,
        "score": bp_prob,
        "threshold": BP_TARGET_GATE_THRESHOLD if bp_active else None,
    })
    if bp_active and not bp_passed:
        row["decision"] = NON_TARGET
        return row

    if sku_head is not None:
        if not catalog:
            raise ValueError(
                f"{SKU_HEAD_PATH} is installed but {CATALOG_PATH} is missing."
            )
        with timings.measure("match"):
            top_class, top_score, margin, _ = sku_head_predict(embedding, sku_head, catalog)
        passes = top_score >= EXACT_SKU_THRESHOLD and margin >= MARGIN_FLOOR
        match_threshold = EXACT_SKU_THRESHOLD
    else:
        with timings.measure("match"):
            top_class, top_score, margin, _ = cosine_fallback_predict(embedding)
        if top_class is None:
            raise ValueError(
                f"Neither {SKU_HEAD_PATH} nor any reference embeddings in "
                "the Postgres reference_embeddings table are available -- "
                "there is no way to classify."
            )
        passes = (
            top_score >= COSINE_FALLBACK_THRESHOLD
            and margin >= COSINE_FALLBACK_MARGIN
        )
        match_threshold = COSINE_FALLBACK_THRESHOLD

    row["top_class"] = top_class
    row["top_score"] = top_score
    row["margin"] = margin
    row["steps"].append({
        "stage": "sku_head",
        "active": True,
        "mode": row["scoring_mode"],
        "passed": passes,
        "top_class": top_class,
        "score": top_score,
        "margin": margin,
        "threshold": match_threshold,
    })

    if not passes:
        score_too_low = (
            top_score < EXACT_SKU_THRESHOLD if sku_head is not None
            else top_score < COSINE_FALLBACK_THRESHOLD
        )
        if score_too_low and bp_gate is None:
            row["decision"] = NON_TARGET
        elif score_too_low:
            row["decision"] = UNKNOWN
        else:
            row["decision"] = UNKNOWN
        return row

    timings.m_ocr_attempts += 1
    verified_class, *_ = ocr_verify(
        crop_img, top_class, ocr_model_en, ocr_model_ar, keywords, keywords_ar, size_specs, timings
    )
    row["steps"].append({
        "stage": "ocr",
        "active": True,
        "passed": verified_class is not None,
        "matched_class": verified_class,
    })

    if verified_class is None:
        row["decision"] = UNKNOWN
        return row

    row["decision"] = EXACT_SKU
    row["matched_class"] = verified_class
    return row


# =============================================================================
# ORCHESTRATOR -- runs the full cascade on one uploaded shelf image.
# =============================================================================

def process_shelf_image(image_bytes, on_progress=None):
    """Runs stages 1-6 on every product detected in image_bytes.

    on_progress, if given, is called as on_progress(percent, stage) at each
    real milestone (model load, detection done, each crop classified) --
    percent reflects work actually completed, not a simulated/timed guess.

    Returns {
        "detections": [{"decision", "matched_class", "bbox": (x1,y1,x2,y2), "score",
                        "steps": [...per-stage diagnostic trail...]}, ...],
        "image_width": int, "image_height": int, "timings": timing.Timings,
    }
    """
    def _report(percent, stage):
        if on_progress:
            on_progress(percent, stage)

    timings = timing.Timings()

    _report(5, "loading models")
    catalog = load_catalog()
    reject_gate = load_reject_gate()
    bp_gate = load_bp_target_gate()
    sku_head = load_sku_head()
    processor, dino_model = load_dinov3_model()
    keywords = load_keywords()
    keywords_ar = load_keywords_ar()
    size_specs = load_size_specs()
    ocr_model_en = load_ocr_model_en()
    ocr_model_ar = load_ocr_model_ar()

    timings.modes = {
        "detect": f"YOLO detector -- {os.path.basename(WEIGHTS_PATH)}",
        "embed": "DINOv3 backbone (frozen) -- same either way",
        "gate": (
            f"reject gate {'ACTIVE (trained MLP, threshold ' + str(REJECT_GATE_THRESHOLD) + ')' if reject_gate is not None else 'INACTIVE -- file missing, no crop is ever rejected here'}; "
            f"BP target gate {'ACTIVE (trained LogReg, threshold ' + str(BP_TARGET_GATE_THRESHOLD) + ')' if bp_gate is not None else 'INACTIVE -- file missing, no crop is ever dropped here'}"
        ),
        "classify": (
            f"32-class SKU head ACTIVE (trained MLP, threshold {EXACT_SKU_THRESHOLD}) -- the guide's real classifier"
            if sku_head is not None else
            f"SKU head INACTIVE -- falling back to cosine match vs the Postgres reference gallery (threshold {COSINE_FALLBACK_THRESHOLD})"
        ),
        "verify": "OCR: EN + AR branches, consensus = OR across bicubic/CLAHE variants (unchanged by the gate)",
        "decide": "standard EXACT_SKU / UNKNOWN / REJECT / NON_TARGET rules (unchanged by the gate)",
    }

    _report(15, "detecting products")
    crop_imgs, crop_boxes, width, height = run_detection(image_bytes, timings)

    total = len(crop_imgs)
    detections = []
    for i, (crop_img, box) in enumerate(zip(crop_imgs, crop_boxes)):
        row = classify_crop(
            crop_img, processor, dino_model, catalog,
            reject_gate, bp_gate, sku_head, ocr_model_en, ocr_model_ar,
            keywords, keywords_ar, size_specs, timings,
        )
        detections.append({
            "decision": row["decision"],
            "matched_class": row["matched_class"],
            "bbox": box,
            "score": row["top_score"],
            "steps": row["steps"],
        })
        if total:
            _report(20 + int(75 * (i + 1) / total), f"classifying crop {i + 1} of {total}")

    _report(100, "done")
    return {
        "detections": detections,
        "image_width": width,
        "image_height": height,
        "timings": timings,
    }
