# Split manifest & shared cells (Role 1)

Run by Role 1 once; the resulting `split_manifest.csv` is committed to this repo and
pulled by Roles 2 and 3. Do **not** resplit.

```python
import os, glob, pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
ROOT = "data"   # contains Training/ and Testing/
rows = []
for split_dir, tag in [("Training", "trainval"), ("Testing", "test")]:
    for c in CLASSES:
        for p in glob.glob(os.path.join(ROOT, split_dir, c, "*")):
            rows.append({"path": p, "label": c, "split": tag})
df = pd.DataFrame(rows)

tv = df[df.split == "trainval"]
train_idx, val_idx = train_test_split(
    tv.index, test_size=0.15, stratify=tv.label, random_state=42)
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df.to_csv("split_manifest.csv", index=False)
print(df.groupby(["split", "label"]).size().unstack())
```

Paths in the manifest are relative to the `T20/` folder root (`data/Training/...`).
Colab: upload `data/` to Drive and adjust `ROOT` accordingly (keep the same manifest
rows; only the prefix differs — document the prefix rule in the notebook header).
