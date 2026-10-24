"""Measure CPU latency / model size / startup time for BOTH final models (plan §6).

Run from T20/ after both model files are in model/:
    .venv/bin/python tools/measure_models.py

Temporarily swaps app/config.py FRAMEWORK/MODEL_PATH (restored afterwards),
starts the API, warms up, times 30 predictions on a real test image,
and appends the numbers to report/deployment_table.md.
"""
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
PORT = 8018
IMAGE = ROOT / "data/Testing/glioma/Te-gl_100.jpg"
CFG = ROOT / "app/config.py"

CASES = [
    ("pytorch", "model/model_T20.pt"),
    ("tensorflow", "model/model_T20.keras"),
]


def set_config(framework: str, model_path: str) -> None:
    src = CFG.read_text(encoding="utf-8")
    src = re.sub(r'FRAMEWORK = "[^"]+"', f'FRAMEWORK = "{framework}"', src)
    src = re.sub(r'MODEL_PATH = "[^"]+"', f'MODEL_PATH = "{model_path}"', src)
    CFG.write_text(src, encoding="utf-8")


def measure() -> dict:
    start = time.time()
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while True:
            if time.time() - start > 60:
                raise RuntimeError("startup timeout — missing model or import error")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    r.read()
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited early (code {server.returncode})")
                time.sleep(0.3)
        startup_s = time.time() - start

        # warm-up (10), then 30 timed predictions through the API
        img_arg = f"file=@{IMAGE}"
        subprocess.run(["curl", "-s", "-o", "/dev/null", "-X", "POST",
                        f"http://127.0.0.1:{PORT}/predict", "-F", img_arg], check=True)
        for _ in range(9):
            subprocess.run(["curl", "-s", "-o", "/dev/null", "-X", "POST",
                            f"http://127.0.0.1:{PORT}/predict", "-F", img_arg], check=True)
        lat = []
        for _ in range(30):
            t = time.perf_counter()
            subprocess.run(["curl", "-s", "-o", "/dev/null", "-X", "POST",
                            f"http://127.0.0.1:{PORT}/predict", "-F", img_arg], check=True)
            lat.append((time.perf_counter() - t) * 1000)
        return {"startup_s": round(startup_s, 2),
                "latency_median_ms": round(median(lat)),
                "latency_min_ms": round(min(lat)),
                "latency_max_ms": round(max(lat))}
    finally:
        server.terminate()
        server.wait(timeout=10)


def main() -> None:
    shutil.copy(CFG, CFG.with_suffix(".py.bak"))
    rows = []
    try:
        for fw, path in CASES:
            if not (ROOT / path).is_file():
                print(f"SKIP {fw}: {path} not present yet")
                continue
            set_config(fw, path)
            m = measure()
            m.update({"framework": fw, "model_mb": round((ROOT / path).stat().st_size / 1e6, 1)})
            print(m)
            rows.append(m)
    finally:
        shutil.move(str(CFG.with_suffix(".py.bak")), CFG)  # original restored

    if rows:
        out = ["| Criterion | " + " | ".join(r["framework"] for r in rows) + " |",
               "|" + "---|" * (len(rows) + 1)]
        fields = [("Model file size (MB)", "model_mb"),
                  ("CPU latency median (ms, 30 req, post warm-up)", "latency_median_ms"),
                  ("CPU latency min / max (ms)", None),
                  ("API startup time (s)", "startup_s")]
        for label, key in fields:
            if key:
                cells = [str(r[key]) for r in rows]
            else:
                cells = [f"{r['latency_min_ms']} / {r['latency_max_ms']}" for r in rows]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        table = "\n".join(out)
        (ROOT / "report/deployment_table.md").write_text(
            "<!-- measured by tools/measure_models.py; extend manually with accuracy, "
            "dependency size and ease-of-serving notes -->\n\n" + table + "\n", encoding="utf-8")
        print("\n" + table)
        print("\nwrote report/deployment_table.md")


if __name__ == "__main__":
    main()
