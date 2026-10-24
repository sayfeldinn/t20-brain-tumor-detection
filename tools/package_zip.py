"""Build and verify the submission ZIP (plan Section 9).

Run from T20/:  .venv/bin/python tools/package_zip.py
Produces ../T20.zip next to the T20 folder (NOT inside it).

What it does:
  1. Verifies the required tree exists (Section 1), with exact Team ID names.
  2. Checks config.py contents (FRAMEWORK, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE).
  3. Builds the ZIP from a staged copy: required files + data/ + model file,
     excluding .git, venv, caches, checkpoints, helpers (tools/, tests/, report/).
  4. Enforces: name <T20>.zip, exactly one top-level folder T20/, size < 1 GB.
  5. Prints the final tree and size.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

TEAM_ID = "T20"
ROOT = Path(__file__).resolve().parents[1]      # .../T20
OUT_ZIP = ROOT.parent / f"{TEAM_ID}.zip"
MAX_BYTES = 1024 * 1024 * 1024                 # 1 GB

REQUIRED_FILES = [
    "requirements.txt",
    "app/main.py",
    "app/config.py",
    "notebooks/pytorch_T20.ipynb",
    "notebooks/tensorflow_T20.ipynb",
]
REQUIRED_DIRS = ["data", "model", "notebooks"]
MODEL_GLOBS = ["model/model_T20.pt", "model/model_T20.keras"]
EXCLUDE_NAMES = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
                 "tools", "tests", "report", "results", "wandb", "checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".ckpt", ".pth"}
EXCLUDE_FILES = {".gitignore", ".gitattributes", "README.md", "TEAMMATES.md",
                 "split_manifest.csv",            # not in Section 1 tree; manifest is dev-only
                 "notebooks/shared_cells.py",     # dev reference; cells are pasted into notebooks
                 "notebooks/README_split.md"}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_config() -> str:
    cfg = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(cfg, "config.py", "exec"), ns)  # our own file, trusted
    assert ns["CLASS_NAMES"] == ["glioma", "meningioma", "notumor", "pituitary"], "CLASS_NAMES order wrong"
    assert ns["IMAGE_SIZE"] == 224, "IMAGE_SIZE wrong"
    assert ns["FRAMEWORK"] in ("pytorch", "tensorflow"), "FRAMEWORK wrong"
    expected_ext = ".pt" if ns["FRAMEWORK"] == "pytorch" else ".keras"
    assert ns["MODEL_PATH"] == f"model/model_{TEAM_ID}{expected_ext}", \
        f"MODEL_PATH must be model/model_{TEAM_ID}{expected_ext}, got {ns['MODEL_PATH']}"
    return ns["FRAMEWORK"]


def main() -> None:
    # 1. required tree
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            fail(f"missing required file: {rel}")
    for rel in REQUIRED_DIRS:
        if not (ROOT / rel).is_dir():
            fail(f"missing required dir: {rel}")
    report = ROOT / f"report_{TEAM_ID}.pdf"
    if not report.is_file():
        fail(f"missing report_{TEAM_ID}.pdf (build it first: .venv/bin/python tools/build_report.py)")

    # model file present (either extension, matching config)
    framework = check_config()
    want = f"model/model_{TEAM_ID}.pt" if framework == "pytorch" else f"model/model_{TEAM_ID}.keras"
    if not (ROOT / want).is_file():
        fail(f"missing model file: {want} (FRAMEWORK={framework})")
    # reject the 6KB dummy model from tools/make_dummy_model.py (contract-testing only!)
    size_mb = (ROOT / want).stat().st_size / 1e6
    if size_mb < 5:
        fail(f"{want} is only {size_mb:.1f} MB — that's the dummy model, not a trained one. "
             f"Replace it with the final export from the winning run before packaging.")
    # stale model of the other framework is excluded automatically by glob logic below

    # notebook outputs saved? (nbformat: outputs list non-empty)
    import json
    for nb_rel in ("notebooks/pytorch_T20.ipynb", "notebooks/tensorflow_T20.ipynb"):
        nb = json.loads((ROOT / nb_rel).read_text(encoding="utf-8"))
        n_out = sum(len(c.get("outputs", [])) for c in nb["cells"] if c["cell_type"] == "code")
        if n_out == 0:
            fail(f"{nb_rel} has NO saved outputs — re-run top to bottom and save with outputs")

    # 2. stage
    tmp = Path(tempfile.mkdtemp(prefix="t20_pkg_"))
    stage = tmp / TEAM_ID
    shutil.copytree(ROOT, stage,
                    ignore=shutil.ignore_patterns(*EXCLUDE_NAMES, "*.pyc", ".git*"))

    # prune excluded files and stray model files
    for p in sorted(stage.rglob("*"), reverse=True):
        if p.is_file():
            rel = str(p.relative_to(stage))
            if p.name in EXCLUDE_FILES or rel in EXCLUDE_FILES or p.suffix in EXCLUDE_SUFFIXES:
                p.unlink()
            elif p.parent.name == "model" and rel != want:
                p.unlink()  # drop stale dummy/other-framework model
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()

    # 3. zip
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(tmp))

    # 4. verify
    size = OUT_ZIP.stat().st_size
    if size >= MAX_BYTES:
        fail(f"ZIP is {size / 1e9:.2f} GB — over 1 GB")
    with zipfile.ZipFile(OUT_ZIP) as zf:
        names = zf.namelist()
        tops = {n.split("/")[0] for n in names}
        if tops != {TEAM_ID}:
            fail(f"top-level entries must be exactly {{{TEAM_ID}}}, got {tops}")
        bad = [n for n in names if any(seg in EXCLUDE_NAMES or seg == ".git"
                                       for seg in Path(n).parts[1:-1])
               or n.endswith((".pyc", ".pth", ".ckpt"))]
        if bad:
            fail(f"forbidden entries in ZIP: {bad[:10]}")
        for req in REQUIRED_FILES + [f"report_{TEAM_ID}.pdf", want]:
            if f"{TEAM_ID}/{req}" not in names:
                fail(f"ZIP missing required entry: {TEAM_ID}/{req}")
        n_data = sum(1 for n in names if n.startswith(f"{TEAM_ID}/data/") and n.count("/") >= 3)
        if n_data < 7200:
            fail(f"ZIP data/ has {n_data} images, expected 7200")

    shutil.rmtree(tmp)
    print(f"OK: {OUT_ZIP}  ({size / 1e6:.1f} MB, {len(names)} entries, framework={framework})")
    print("\nFinal tree:")
    for n in sorted({"/".join(n.split("/")[:2]) for n in names}):
        print("  ", n)


if __name__ == "__main__":
    main()
