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
import math
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MEDIA_DIR = os.path.join(BASE_DIR, "media")
CROPS_DIR = os.path.join(MEDIA_DIR, "crops")
ANNOTATED_DIR = os.path.join(MEDIA_DIR, "annotated")
UPLOAD_DIR = os.path.join(MEDIA_DIR, "uploads")

for folder in (CROPS_DIR, ANNOTATED_DIR, UPLOAD_DIR):
    os.makedirs(folder, exist_ok=True)


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
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.json")
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

OCR_UPSCALE = 2.0

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

def load_image_exif_corrected(path):
    img = Image.open(path)
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
def load_embeddings():
    if not os.path.isfile(EMBEDDINGS_PATH):
        return None
    embeddings = []
    with open(EMBEDDINGS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                embeddings.append(json.loads(line))
    if not embeddings:
        return None

    dim = len(embeddings[0]["embedding"])
    if dim != EMBED_DIM:
        raise ValueError(
            f"{EMBEDDINGS_PATH} contains {dim}-D embeddings, but DINOv3 "
            f"ViT-B/16 produces {EMBED_DIM}-D. Regenerate them with "
            f"{DINOV3_MODEL_ID} before using this app."
        )
    return tuple(embeddings)


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


def _build_ocr_model(rec_model_name, lang):
    from paddleocr import PaddleOCR

    kwargs = dict(
        text_detection_model_name=OCR_DET_MODEL_NAME,
        text_recognition_model_name=rec_model_name,
        lang=lang,
        device=OCR_DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    try:
        return PaddleOCR(enable_mkldnn=OCR_ENABLE_MKLDNN, **kwargs)
    except TypeError:
        return PaddleOCR(**kwargs)


@functools.lru_cache(maxsize=1)
def load_ocr_model_en():
    return _build_ocr_model(OCR_REC_MODEL_NAME_EN, lang="en")


@functools.lru_cache(maxsize=1)
def load_ocr_model_ar():
    return _build_ocr_model(OCR_REC_MODEL_NAME_AR, lang="ar")


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


def run_detection(image_path, image_id):
    """Returns (crop_paths, crop_boxes, image_width, image_height).
    crop_boxes are the RAW detector boxes (unpadded); the padded versions
    are used for the actual recognition crops written to disk."""
    model = load_yolo_model()
    results = model.predict(
        source=image_path,
        imgsz=DET_IMGSZ,
        conf=DET_CONF,
        iou=DET_IOU,
        max_det=DET_MAX_DET,
        verbose=False,
    )
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()

    pil_image = load_image_exif_corrected(image_path)
    crop_paths, crop_boxes = [], []

    crop_subdir = os.path.join(CROPS_DIR, image_id)
    os.makedirs(crop_subdir, exist_ok=True)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        crop_img = make_recognition_crop(pil_image, x1, y1, x2, y2)
        crop_path = os.path.join(crop_subdir, f"crop-{i:04d}.jpg")
        crop_img.save(crop_path, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
        crop_paths.append(crop_path)
        crop_boxes.append(tuple(int(v) for v in (x1, y1, x2, y2)))

    return crop_paths, crop_boxes, pil_image.width, pil_image.height


# =============================================================================
# STAGE 2 -- EMBED
# =============================================================================

def embed_crop(crop_path, processor, model):
    img = Image.open(crop_path).convert("RGB")
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


def cosine_fallback_predict(embedding, embeddings):
    class_scores = {}
    emb_list = embedding.tolist()
    size1 = math.sqrt(sum(a * a for a in emb_list))

    for item in embeddings:
        old = item["embedding"]
        name = item["class"]
        dot = sum(a * b for a, b in zip(emb_list, old))
        size2 = math.sqrt(sum(b * b for b in old))
        if size1 == 0 or size2 == 0:
            continue
        score = (dot / (size1 * size2)) * 100.0
        if name not in class_scores or score > class_scores[name]:
            class_scores[name] = score

    if not class_scores:
        return None, 0.0, 0.0, []

    ranked = sorted(class_scores.items(), key=lambda kv: kv[1], reverse=True)
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


def _to_bicubic_rgb(pil_crop, upscale=OCR_UPSCALE):
    arr = np.array(pil_crop.convert("RGB"))
    h, w = arr.shape[:2]
    resized = cv2.resize(arr, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def _to_clahe_gray(pil_crop, upscale=OCR_UPSCALE):
    gray = np.array(pil_crop.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    h, w = enhanced.shape[:2]
    resized = cv2.resize(enhanced, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def _run_ocr_variant(array, ocr_model):
    results = ocr_model.predict(array)
    text_parts = []
    for result in results:
        res = result.json.get("res", {})
        rec_texts = res.get("rec_texts", [])
        if rec_texts:
            text_parts.extend(rec_texts)
    return _normalize_ocr_text(" ".join(text_parts))


def read_ocr_both_variants(crop_path, ocr_model, label):
    pil_crop = Image.open(crop_path).convert("RGB")
    bicubic_text = _run_ocr_variant(_to_bicubic_rgb(pil_crop), ocr_model)
    clahe_text = _run_ocr_variant(_to_clahe_gray(pil_crop), ocr_model)
    return bicubic_text, clahe_text


def _kw_in_consensus(keyword, text_a, text_b):
    kw = keyword.lower()
    return kw in text_a and kw in text_b


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


def _size_confirmed_consensus(number, size_type, text_a, text_b):
    return (_size_token_present(number, size_type, text_a)
            and _size_token_present(number, size_type, text_b))


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


def ocr_verify(crop_path, top_class, ocr_model_en, ocr_model_ar, keywords, keywords_ar, size_specs):
    text_a, text_b = read_ocr_both_variants(crop_path, ocr_model_en, label="EN")
    ar_a, ar_b = read_ocr_both_variants(crop_path, ocr_model_ar, label="AR")

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
    if (_any_kw_in_consensus(class_keywords, text_a, text_b)
            or _any_kw_in_consensus(class_keywords_ar, ar_a, ar_b)):
        return _finalize(top_class)

    exclude = GREEN_TEA_CLASSES | BBRL_CLASSES
    fallback = _best_keyword_match_consensus(text_a, text_b, keywords, exclude_classes=exclude)
    if fallback is None:
        return None, text_a, text_b, ar_a, ar_b
    return _finalize(fallback)


# =============================================================================
# STAGE 6 -- DECIDE
# =============================================================================

def classify_crop(crop_path, processor, dino_model, catalog, embeddings,
                   reject_gate, bp_gate, sku_head, ocr_model_en, ocr_model_ar,
                   keywords, keywords_ar, size_specs):
    row = {
        "crop_path": crop_path,
        "decision": None,
        "matched_class": None,
        "top_class": None,
        "top_score": 0.0,
        "margin": 0.0,
        "scoring_mode": "sku_head" if sku_head is not None else "cosine_fallback",
    }

    embedding = embed_crop(crop_path, processor, dino_model)

    valid_prob = reject_gate_score(embedding, reject_gate)
    if valid_prob is not None and valid_prob < REJECT_GATE_THRESHOLD:
        row["decision"] = REJECT
        return row

    bp_prob = bp_target_gate_score(embedding, bp_gate)
    if bp_prob is not None and bp_prob < BP_TARGET_GATE_THRESHOLD:
        row["decision"] = NON_TARGET
        return row

    if sku_head is not None:
        if not catalog:
            raise ValueError(
                f"{SKU_HEAD_PATH} is installed but {CATALOG_PATH} is missing."
            )
        top_class, top_score, margin, _ = sku_head_predict(embedding, sku_head, catalog)
        passes = top_score >= EXACT_SKU_THRESHOLD and margin >= MARGIN_FLOOR
    else:
        if not embeddings:
            raise ValueError(
                f"Neither {SKU_HEAD_PATH} nor a usable {EMBEDDINGS_PATH} is "
                "available -- there is no way to classify."
            )
        top_class, top_score, margin, _ = cosine_fallback_predict(embedding, embeddings)
        passes = (
            top_score >= COSINE_FALLBACK_THRESHOLD
            and margin >= COSINE_FALLBACK_MARGIN
        )

    row["top_class"] = top_class
    row["top_score"] = top_score
    row["margin"] = margin

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

    verified_class, *_ = ocr_verify(
        crop_path, top_class, ocr_model_en, ocr_model_ar, keywords, keywords_ar, size_specs
    )

    if verified_class is None:
        row["decision"] = UNKNOWN
        return row

    row["decision"] = EXACT_SKU
    row["matched_class"] = verified_class
    return row


# =============================================================================
# ORCHESTRATOR -- runs the full cascade on one uploaded shelf image.
# =============================================================================

def process_shelf_image(image_path, image_id):
    """Runs stages 1-6 on every product detected in image_path.

    Returns {
        "detections": [{"decision", "matched_class", "bbox": (x1,y1,x2,y2), "score"}, ...],
        "image_width": int, "image_height": int,
    }
    """
    catalog = load_catalog()
    reject_gate = load_reject_gate()
    bp_gate = load_bp_target_gate()
    sku_head = load_sku_head()
    embeddings = load_embeddings() if sku_head is None else None
    processor, dino_model = load_dinov3_model()
    keywords = load_keywords()
    keywords_ar = load_keywords_ar()
    size_specs = load_size_specs()
    ocr_model_en = load_ocr_model_en()
    ocr_model_ar = load_ocr_model_ar()

    crop_paths, crop_boxes, width, height = run_detection(image_path, image_id)

    detections = []
    for crop_path, box in zip(crop_paths, crop_boxes):
        row = classify_crop(
            crop_path, processor, dino_model, catalog, embeddings,
            reject_gate, bp_gate, sku_head, ocr_model_en, ocr_model_ar,
            keywords, keywords_ar, size_specs,
        )
        detections.append({
            "decision": row["decision"],
            "matched_class": row["matched_class"],
            "bbox": box,
            "score": row["top_score"],
        })

    return {"detections": detections, "image_width": width, "image_height": height}
