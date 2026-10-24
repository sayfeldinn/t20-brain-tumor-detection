# Brain Tumor MRI Classification — TensorFlow vs PyTorch, served with FastAPI

**Course:** AI Platforms, Fall 2026 — Faculty of Computers and Data Science, Alexandria University<br>
**Assignment:** 1 &nbsp;|&nbsp; **Team ID:** T20<br>
**Team members:** Seif (Role 1 — data, FastAPI, report), Ahmed Refaat (Role 2 — TensorFlow), Moatasem (Role 3 — PyTorch)

---

## 1. Introduction

<!-- Role 1. 3-5 sentences: task (4-class brain MRI classification), why transfer learning,
why the framework comparison matters for platform selection, how the report is organized. -->

---

## 2. Dataset and split

**Source:** Brain Tumor MRI Dataset (Kaggle, Masoud Nickparvar) — `Training/` and `Testing/` folders, 4 class folders each. All files are JPEG.

| Split | glioma | meningioma | notumor | pituitary | Total |
|---|---|---|---|---|---|
| Train | 1,190 | 1,190 | 1,190 | 1,190 | **4,760** |
| Validation | 210 | 210 | 210 | 210 | **840** |
| Test (official `Testing/`) | 400 | 400 | 400 | 400 | **1,600** |
| **Total** | 1,600 | 1,600 | 1,600 | 1,600 | **7,200** |

- The dataset is **perfectly class-balanced** (1,400 trainval + 400 test per class); no class weights were needed.
- **Split rule (contract):** the official `Testing/` folder is the test set; a stratified 15% of `Training/` (seed 42) is carved as validation. One shared `split_manifest.csv` is used verbatim by both frameworks; no re-splitting was performed.
- **Image properties (EDA):** 0 corrupt files; **mixed color modes** — 3,067 grayscale (mode L), 4,129 RGB, 3 RGBA, 1 palette → all pipelines call `convert("RGB")`. **447 distinct resolutions** (5,014 of 7,200 are 512×512); all images are resized to 224×224.
- **File-name quirk:** meningioma test images are named `Te-aug-*`, indicating the source dataset stores augmented copies for that class; we treat them as given test data.

### Duplicate and leakage analysis (SHA-256 over all 7,200 files)

| Check | Result |
|---|---|
| Exact duplicate groups (same hash) | 153 groups, 340 files (all within-class; 0 cross-label conflicts) |
| Test set leaking into train/val | **0 groups — test set is clean** ✅ |
| Train ↔ val duplicates | 42 groups; **44 of 840 validation images (5.2%)** are exact copies of training images (30 notumor, 10 pituitary, 4 meningioma) |

**Decision:** per the shared contract the split was not altered. The 5.2% train/val duplication mildly optimizes *validation* metrics (model selection), but the headline **test numbers are unaffected**, and both frameworks use the identical manifest, so the framework comparison remains fair. See Limitations (§9).

---

## 3. Preprocessing and augmentation

<!-- Role 1, input from both roles. Pre-fill from the contract:
- Resize to 224×224, RGB conversion.
- Native ImageNet preprocessing per architecture, baked into the saved model so the API only resizes and passes pixels:
  - ResNet50 (both frameworks): per-channel mean/std subtraction (classic ResNet preprocessing).
  - EfficientNet-B0: TF expects raw 0–255 (Rescaling inside the model); Torchvision expects 0–1 floats.
- Augmentation (train only): rotation ±10°, translation ±10%, scale 0.9–1.1, brightness/contrast 0.9–1.1.
- Deliberately NOT used: vertical flip, hue/saturation jitter, heavy distortion, (horizontal flip: not used / used with justification: ______).
- Why this suits MRI: small positional jitter is safe; photometric transforms beyond brightness/contrast do not model real scanner variation and can destroy subtle contrast that separates tumor tissue; flips are not clinically justified.
- Validation/test: resize + normalize only, no augmentation. -->

---

## 4. PyTorch implementation <!-- Role 3 = Moatasem -->

<!-- Architecture(s): torchvision ResNet50 & EfficientNet-B0, ImageNet weights, 4-class head.
Transfer learning setup: R1 frozen backbone (head only) vs R2/R4 fine-tune (unfreeze last block, LR 1e-5..1e-4).
Hyperparameters: Adam/AdamW, batch 32, max 15 epochs, early stopping on val loss patience 3, seed 42.
Loss curves (embed from notebook), test Accuracy / Precision / Recall / F1 (macro + per-class), confusion matrix.
Experiment table R1–R4 (+R5 if run): per-run metrics from results/metrics_pytorch.json. -->

**Deliverable:** fill from `notebooks/pytorch_T20.ipynb` + `results/metrics_pytorch.json`.

---

## 5. TensorFlow implementation <!-- Role 2 = Ahmed Refaat -->

<!-- Same structure as Section 4, from notebooks/tensorflow_T20.ipynb + results/metrics_tensorflow.json.
Note Keras specifics: BatchNorm kept in inference mode during fine-tune (training=False),
preprocessing inside the .keras model (ResNet50 preprocess_input; EfficientNetB0 raw 0–255). -->

**Deliverable:** fill from `notebooks/tensorflow_T20.ipynb` + `results/metrics_tensorflow.json`.

---

## 6. Interpretability (GradCAM)

<!-- Both models, same fixed test images (shared_cells.py CELL G: 6 images + 1 misclassified added later).
PyTorch target layers: ResNet50 layer4[-1], EfficientNet-B0 features[-1].
TF target layers: conv5_block3_out / top_activation.
Show correct + misclassified cases; discuss what regions are highlighted
(ventricle/tumor mass vs background) and differences between the two models. -->

**Deliverable:** GradCAM figures from both notebooks (identical image list).

---

## 7. Framework comparison

<!-- Role 1 writes; evidence from everyone's notes (plan Section 7):
1. Ease of implementation and debugging: pipeline LOC, errors hit + fix time, error message quality,
   tensor inspection, GradCAM difficulty.
2. Training speed & resources: seconds/epoch, total train time, peak GPU MB — same T4, same batch 32.
3. Flexibility: swapping backbones, freezing, hooks, custom loops/layers.
4. Performance & stability: test metrics, run-to-run variance, LR sensitivity, diverged runs.
Tables must use real numbers from results/metrics_*.json and notebook outputs. -->

---

## 8. Architecture / hyperparameter / strategy comparison

<!-- Consolidated R1–R5 table (both frameworks side by side):
Run | Framework | Architecture | Strategy | best_val_acc | best_val_loss | epochs | s/epoch | peak GPU MB | notes
-->

---

## 9. Limitations

<!-- Role 1. Pre-filled bullets to expand:
**Dataset:**
- 5.2% of validation images are exact duplicates of training images (SHA-256); model-selection metrics mildly optimistic; test set verified clean (0 cross-split duplicates).
- No patient-level split: slices from the same patient may appear across train/val, inflating validation scores.
- Single source, mostly 2D axial slices; limited scanner/protocol diversity; possible label noise (radiologist labels, no second reader).
- Meningioma test folder contains pre-augmented copies (`Te-aug-*`).
- Class balance is artificial (exactly balanced); real clinical prevalence differs (notumor would dominate).

**Models:**
- Not clinically validated; no calibration; risk under domain shift to new scanners.
- GradCAM shows where the model looks, not that the reasoning is clinically correct.
- Limited tuning (single seed 42, ≤15 epochs, small hyperparameter search, Colab T4 constraints).
- Frame of reference: accuracy on this benchmark ≠ clinical utility.

**Work:**
- Single seed; limited hyperparameter search (R5 optional); Colab GPU shared/limited time. -->

---

## 10. Deployment (which model was served and why)

<!-- Role 1. Section 6 table with REAL measurements:

| Criterion | PyTorch final | TensorFlow final |
|---|---|---|
| Test Accuracy / macro F1 | | |
| Recall on tumor classes | | |
| CPU latency through the API (median of 20+, after warm-up) | | |
| Model file size (MB) | | |
| Dependency size / startup time | | |
| Ease of serving | | |

Rule: clearly higher test F1 wins; if close, prefer smaller/faster/simpler.
Describe the service: FastAPI, model loaded once at startup, offline CPU-only,
preprocessing parity with training (normalization baked into the model),
input validation (JPEG/PNG only → 4xx otherwise). -->

---

## 11. Example API requests

<!-- Role 1: 2 real request/response pairs:
(a) curl POST /predict with a real test image → full JSON response (correct prediction case)
(b) curl POST /predict with a non-image file → 4xx response (or a lower-confidence case)
Include the /health response as well. -->

```
$ curl -s http://localhost:8000/health
<!-- PASTE REAL RESPONSE -->

$ curl -s -X POST http://localhost:8000/predict -F "file=@data/Testing/..."
<!-- PASTE REAL RESPONSE -->

$ curl -s -X POST http://localhost:8000/predict -F "file=@notes.txt"
<!-- PASTE REAL RESPONSE (4xx) -->
```

---

## 12. Appendix: environment and package versions

<!-- Pinned requirements.txt content + Colab versions (Python, TF, torch, GPU = T4, driver),
recorded by each role from their notebooks. -->
