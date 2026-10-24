import io, time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app import config

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = torch.jit.load(config.MODEL_PATH, map_location="cpu")
    model.eval()
    state["model"] = model          # loaded ONCE at startup
    yield
    state.clear()

app = FastAPI(title="Brain Tumor MRI Classifier", lifespan=lifespan)

def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0           # must match training
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # normalization baked into model

@app.get("/health")
def health():
    return {"status": "ok", "framework": config.FRAMEWORK, "classes": config.CLASS_NAMES}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="File must be a JPEG or PNG image")
    try:
        img = Image.open(io.BytesIO(file.file.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image")
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(state["model"](preprocess(img)), dim=1)[0].tolist()
    latency_ms = round((time.perf_counter() - start) * 1000)
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "framework": config.FRAMEWORK,
        "label": config.CLASS_NAMES[best],
        "confidence": round(probs[best], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(config.CLASS_NAMES, probs)},
        "latency_ms": latency_ms,
    }
