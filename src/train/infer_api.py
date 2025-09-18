# FastAPI ONNX inference API: single-image (/infer, /infer_form) + sequence (/infer_sequence with Kalman smoothing)

import os
import io
import json
from typing import List, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
import onnxruntime as ort
import torch

# ---- Settings (can be overridden by env) ----
ONNX_PATH = os.getenv("ONNX_PATH", "models/fer2013_cnn.onnx")
CLASSES_PATH = os.getenv("CLASSES_PATH", "")  # optional: per-line class names
IMG_SIZE = (48, 48)

# ---- App ----
app = FastAPI(title="Edge Perception ONNX Inference API", version="0.1.0")

# ---- Load ONNX session ----
if not os.path.isfile(ONNX_PATH):
    raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

# Prefer GPU if available; fall back to CPU
_available = ort.get_available_providers()
_providers_pref = [p for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in _available]
sess = ort.InferenceSession(ONNX_PATH, providers=_providers_pref)

# ---- Classes (optional) ----
classes: Optional[List[str]] = None
if CLASSES_PATH and os.path.isfile(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]

# ---- Utils ----
def preprocess_np(img: Image.Image) -> np.ndarray:
    """Luma → resize → [0,1] → (1,1,H,W) float32"""
    x = img.convert("L").resize(IMG_SIZE)
    x = np.asarray(x, dtype=np.float32) / 255.0
    return x[None, None, :, :]

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def label_of(index: int) -> str | int:
    return classes[index] if classes and 0 <= index < len(classes) else index

# ---- Routes ----
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "onnx": os.path.basename(ONNX_PATH),
        "providers": _providers_pref,
        "classes_loaded": bool(classes),
    }

@app.post("/infer")  # raw binary
async def infer(file: bytes = File(...)):
    try:
        img = Image.open(io.BytesIO(file))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    x = preprocess_np(img)                                   # (1,1,H,W)
    try:
        logits = sess.run(["logits"], {"input": x})[0]       # (1,C)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"onnxruntime error: {e}")

    probs = softmax(logits, axis=1)[0]
    top1 = int(np.argmax(probs))
    conf = float(probs[top1])

    return {"label": label_of(top1), "confidence": conf, "probs": probs.tolist()}

@app.post("/infer_form")  # multipart single file
async def infer_form(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    x = preprocess_np(img)
    logits = sess.run(["logits"], {"input": x})[0]
    probs = softmax(logits, axis=1)[0]
    top1 = int(np.argmax(probs))
    conf = float(probs[top1])

    return {"filename": file.filename, "label": label_of(top1), "confidence": conf, "probs": probs.tolist()}

# ---- Sequence inference with Kalman smoothing ----
# requires: src/robustness/kalman.py -> kf_smooth_probs(prob_seq: (T,C) torch.Tensor) -> (T,C) torch.Tensor
from src.robustness.kalman import kf_smooth_probs

@app.post("/infer_sequence")
async def infer_sequence(files: List[UploadFile] = File(...)):
    """
    Upload multiple images (multipart). Files are sorted by filename and treated as a time sequence.
    Returns raw top1/confidence and Kalman-smoothed top1/confidence across the sequence.
    """
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images for a sequence.")

    # load & sort by filename to ensure temporal order
    frames: list[tuple[str, Image.Image]] = []
    for f in files:
        try:
            content = await f.read()
            img = Image.open(io.BytesIO(content))
            frames.append((f.filename, img))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image {f.filename}: {e}")
    frames.sort(key=lambda x: x[0])

    # per-frame inference -> probs list
    probs_list, raw_top1, raw_conf = [], [], []
    for name, img in frames:
        x = preprocess_np(img)
        logits = sess.run(["logits"], {"input": x})[0]
        p = softmax(logits, axis=1)[0]
        probs_list.append(p)
        t1 = int(np.argmax(p))
        raw_top1.append(label_of(t1))
        raw_conf.append(float(p[t1]))

    # Kalman smoothing on (T,C)
    probs_np = np.stack(probs_list, axis=0).astype(np.float32)   # (T,C)
    probs_t  = torch.from_numpy(probs_np)
    probs_s  = kf_smooth_probs(probs_t, q=1e-3, r=1e-2).numpy()

    sm_top1, sm_conf = [], []
    for i in range(probs_s.shape[0]):
        t1 = int(np.argmax(probs_s[i]))
        sm_top1.append(label_of(t1))
        sm_conf.append(float(probs_s[i, t1]))

    return {
        "count": len(frames),
        "order": [n for n, _ in frames],
        "raw": {"top1": raw_top1, "confidence": raw_conf},
        "smoothed": {"top1": sm_top1, "confidence": sm_conf},
        # if the complete "smoothed probs" is needed, cancel the comment of the following line (warning: this may cause return of large amount)
        # "smoothed_probs": probs_s.tolist(),
    }

