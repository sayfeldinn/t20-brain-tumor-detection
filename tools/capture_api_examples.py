"""Capture the 2 example API request/response pairs for report §11.

Run from T20/ with the FINAL model in config.py and no server running:
    .venv/bin/python tools/capture_api_examples.py
Starts uvicorn itself, sends:
  1. GET  /health
  2. POST /predict with a real test image
  3. POST /predict with a non-image file (4xx case)
Writes report/api_examples.md and prints the block to paste into report_source.md §11.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8017  # avoid clashes with a dev server on 8000
IMAGE = ROOT / "data/Testing/glioma/Te-gl_100.jpg"
NONIMAGE = Path("/tmp/opencode/api_notes.txt")


def wait_health(timeout=30) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become healthy")


def curl(args: list[str]) -> tuple[int, str]:
    out = subprocess.run(["curl", "-s", "-w", "\n%{http_code}"] + args,
                         capture_output=True, text=True)
    body, _, code = out.stdout.rpartition("\n")
    return int(code), body


def main() -> None:
    if not IMAGE.is_file():
        sys.exit(f"missing test image {IMAGE}")
    NONIMAGE.write_text("this is not an image\n")

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        health = wait_health()
        print("health:", health)

        code1, body1 = curl(["http://127.0.0.1:%d/health" % PORT])
        code2, body2 = curl(["-X", "POST", f"http://127.0.0.1:{PORT}/predict",
                             "-F", f"file=@{IMAGE}"])
        code3, body3 = curl(["-X", "POST", f"http://127.0.0.1:{PORT}/predict",
                             "-F", f"file=@{NONIMAGE}"])
        assert code1 == 200 and code2 == 200 and 400 <= code3 < 500, (code1, code2, code3)

        # sanity: probabilities sum to ~1, label in class list
        pred = json.loads(body2)
        assert abs(sum(pred["probabilities"].values()) - 1) < 0.01
        assert pred["label"] in health["classes"]

        block = f"""## 11. Example API requests

```console
$ curl -s http://localhost:8000/health
{body1}

$ curl -s -X POST http://localhost:8000/predict -F "file=@{IMAGE.relative_to(ROOT)}"
{json.dumps(pred, indent=2)}

$ curl -s -X POST http://localhost:8000/predict -F "file=@notes.txt"   # status {code3}
{body3}
```
"""
        (ROOT / "report/api_examples.md").write_text(block, encoding="utf-8")
        print("\n" + block)
        print("wrote report/api_examples.md — paste into report_source.md §11")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
