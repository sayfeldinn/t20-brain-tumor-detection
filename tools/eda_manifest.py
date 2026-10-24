"""EDA + duplicate/leakage check + split manifest for the T20 assignment.

Run from T20/:  .venv/bin/python tools/eda_manifest.py
Outputs:
  - printed report (numbers are copied into the report PDF)
  - results/eda_report.json
  - split_manifest.csv  (seed 42, stratified 15% val from Training/, test = Testing/)
  - results/duplicate_pairs.csv (if any duplicates found)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob
import hashlib
import json

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
ROOT = Path("data")
SEED = 42

# ---------- 1. inventory ----------
rows = []
for split_dir, tag in [("Training", "trainval"), ("Testing", "test")]:
    for c in CLASSES:
        for p in sorted(glob.glob(str(ROOT / split_dir / c / "*"))):
            rows.append({"path": p, "label": c, "split": tag})
df = pd.DataFrame(rows)
print(f"total images: {len(df)}")
print(df.groupby(["split", "label"]).size().unstack())

# ---------- 2. image properties, corrupt files ----------
modes, sizes, corrupt, extensions = {}, {}, [], {}
for p in df["path"]:
    ext = Path(p).suffix.lower()
    extensions[ext] = extensions.get(ext, 0) + 1
    try:
        with Image.open(p) as im:
            modes[im.mode] = modes.get(im.mode, 0) + 1
            sizes[im.size] = sizes.get(im.size, 0) + 1
            im.verify()
    except Exception as e:
        corrupt.append({"path": p, "error": str(e)})
print("\nextensions:", extensions)
print("modes:", modes)
print("distinct sizes:", len(sizes))
for s, n in sorted(sizes.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {s}: {n}")
print("corrupt files:", len(corrupt))
for c in corrupt[:20]:
    print("  ", c)

# ---------- 3. duplicates (sha256) ----------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

df["sha256"] = [sha256(p) for p in df["path"]]

# exact duplicates: same hash
dup_hashes = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
n_dup_files = dup_hashes["sha256"].nunique()
print(f"\nexact duplicate groups (same sha256): {n_dup_files}")
print(f"files involved in duplication: {len(dup_hashes)}")

# leakage: hash appears in more than one split
by_hash = df.groupby("sha256").agg(
    n=("path", "size"),
    splits=("split", lambda s: sorted(set(s))),
    labels=("label", lambda s: sorted(set(s))),
)
leak = by_hash[(by_hash.n > 1) & (by_hash.splits.apply(len) > 1)]
print(f"leakage hashes (present in >1 split): {len(leak)}")

# same-hash groups with different labels (label noise indicator)
cross_label = by_hash[by_hash.labels.apply(len) > 1]
print(f"same-image different-label groups: {len(cross_label)}")

dup_rows = []
for h, grp in df[df.duplicated('sha256', keep=False)].groupby('sha256'):
    dup_rows.append({
        "sha256": h,
        "n": len(grp),
        "splits": "|".join(sorted(set(grp["split"]))),
        "labels": "|".join(sorted(set(grp["label"]))),
        "paths": "|".join(grp["path"]),
    })
pd.DataFrame(dup_rows).to_csv("results/duplicate_pairs.csv", index=False)

# ---------- 4. split manifest (seed 42, stratified 15% val) ----------
tv = df[df.split == "trainval"]
train_idx, val_idx = train_test_split(
    tv.index, test_size=0.15, stratify=tv.label, random_state=SEED)
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
manifest = df[["path", "label", "split"]]
manifest.to_csv("split_manifest.csv", index=False)
print("\nsplit manifest:")
print(manifest.groupby(["split", "label"]).size().unstack())
print("split totals:", manifest.split.value_counts().to_dict())

# ---------- 5. save eda report ----------
report = {
    "total_images": int(len(df)),
    "counts": {f"{s}/{l}": int(n) for (s, l), n in
               df.groupby(["split", "label"]).size().items()},
    "extensions": extensions,
    "modes": modes,
    "distinct_sizes": len(sizes),
    "top_sizes": {str(k): v for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:10]},
    "corrupt_files": corrupt,
    "duplicate_groups": int(n_dup_files),
    "duplicate_files_involved": int(len(dup_hashes)),
    "leakage_hashes_across_splits": int(len(leak)),
    "cross_label_duplicate_groups": int(len(cross_label)),
    "split_totals": {k: int(v) for k, v in manifest.split.value_counts().items()},
}
Path("results").mkdir(exist_ok=True)
Path("results/eda_report.json").write_text(json.dumps(report, indent=2))
print("\nwrote results/eda_report.json and split_manifest.csv")
