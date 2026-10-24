"""Contract tests for the FastAPI service. Run from T20/: .venv/bin/python -m pytest tests/ -q
(or plain: .venv/bin/python tests/test_contract.py) with the server NOT required — TestClient
loads the app in-process (needs model/model_T20.pt present)."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

CLS = ["glioma", "meningioma", "notumor", "pituitary"]


def _img_bytes(fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (120, 80, 60)).save(buf, format=fmt)
    return buf.getvalue()


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "framework": "pytorch", "classes": CLS}


def test_predict_jpeg():
    with TestClient(app) as c:
        r = c.post("/predict", files={"file": ("t.jpg", _img_bytes("JPEG"), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["framework"] == "pytorch"
        assert body["label"] in CLS
        assert set(body["probabilities"]) == set(CLS)
        assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
        assert 0 < body["confidence"] <= 1
        assert isinstance(body["latency_ms"], int)


def test_predict_png():
    with TestClient(app) as c:
        r = c.post("/predict", files={"file": ("t.png", _img_bytes("PNG"), "image/png")})
        assert r.status_code == 200


def test_non_image_is_4xx():
    with TestClient(app) as c:
        r = c.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 415


def test_corrupted_image_is_4xx():
    with TestClient(app) as c:
        r = c.post("/predict", files={"file": ("bad.jpg", b"\xff\xd8\xff broken", "image/jpeg")})
        assert r.status_code == 400


def test_missing_file_is_4xx():
    with TestClient(app) as c:
        r = c.post("/predict")
        assert r.status_code == 422


def test_docs():
    with TestClient(app) as c:
        assert c.get("/docs").status_code == 200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} contract tests passed")
