import os
import io
import json
import numpy as np
from typing import List, Optional
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException

# ---- setting ----
ONNX_PATH = os.getenv("ONNX_PATH", "models/fer2013_cnn.onnx")
CLASSES_PATH = os.getenv("CLASSES_PATH", "")  # 可選，txt 檔每行一類別名稱
IMG_SIZE = (48, 48)

# ---- loading ONNX ----
if not os.path.isfile(ONNX_PATH):
    raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# classes name
classes: Optional[List[str]] = None
if CLASSES_PATH and os.path.isfile(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = [ln.strip() for ln in f if ln.strip()]

# ---- preprocess / postprocess ----
def preprocess(img: Image.Image) -> np.ndarray:
    # 灰階 → resize → [0,1] → (1,1,H,W)
    x = img.convert("L").resize(IMG_SIZE)
    x = np.asarray(x, dtype=np.float32) / 255.0  # (H,W)
    x = x[None, None, :, :]                      # (1,1,H,W)
    return x

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def postprocess(logits: np.ndarray):
    # logits: (1,C)
    probs = softmax(logits, axis=1)[0]          # (C,)
    top1 = int(probs.argmax())
    conf = float(probs[top1])
    label = classes[top1] if classes and top1 < len(classes) else str(top1)
    return label, conf, probs.tolist()

# ---- FastAPI ----
app = FastAPI(title="Edge Perception ONNX Inference API")

@app.get("/healthz")
def healthz():
    return {"ok": True, "onnx": os.path.basename(ONNX_PATH), "classes": bool(classes)}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    x = preprocess(img)                         # (1,1,48,48)
    try:
        logits = sess.run(["logits"], {"input": x})[0]  # (1,C)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"onnxruntime error: {e}")

    label, conf, probs = postprocess(logits)
    return {
        "filename": file.filename,
        "label": label,
        "confidence": conf,
        "probs": probs
    }

@app.post("/infer_form")
async def infer_form(file: UploadFile = File(...)):
    return await infer(file)

