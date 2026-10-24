# Getting started — for Refaat (Role 2, TensorFlow) & Moatasem (Role 3, PyTorch)

**Repo:** https://github.com/sayfeldinn/t20-brain-tumor-detection (private — accept Seif's invite first)
**Deadline: Monday Oct 26. Today is Sunday Oct 25 — training must run TODAY.**

## 0. What you own (and only that)

| You | Deliverables |
|---|---|
| Refaat | `notebooks/tensorflow_T20.ipynb` + final `model_T20.keras` |
| Moatasem | `notebooks/pytorch_T20.ipynb` + final `model_T20.pt` |

## 1. Clone (5 min)

```bash
git clone https://github.com/sayfeldinn/t20-brain-tumor-detection.git
cd t20-brain-tumor-detection
```

Already there: **`data/` (the full 7,200-image dataset — comes with the clone, no Drive needed)**,
`split_manifest.csv` (the shared split — **never resplit**), `notebooks/shared_cells.py`
(paste its cells **verbatim** into your notebook), `app/` (API — not yours), `results/metrics_<yourfw>.json`
(fill one entry per run).

## 2. Colab setup (10 min)

1. Runtime → **Change runtime type → T4 GPU** (same GPU as the other framework — fairness rule).
2. Get the data in Colab — pick one:
   - **Option A (repo clone, recommended):** in Colab,
     ```python
     # private repo: use a GitHub personal access token (repo scope) — do NOT hardcode it
     # in a pushed notebook; paste at runtime or use userdata/Colab secrets
     !git clone https://<TOKEN>@github.com/sayfeldinn/t20-brain-tumor-detection.git
     ```
     then work in `t20-brain-tumor-detection/` — `data/` and `split_manifest.csv` come with it.
   - **Option B (local path):** clone on your machine, upload `data/` to Colab's session storage, set `ROOT = "/content/data"`.
3. Copy your notebook into Colab; **prepend the manifest path fix** (shared_cells.py CELL C):
   adjust `df["path"]` to wherever you put `data/` (e.g. `"/content/t20-brain-tumor-detection/" + df["path"]`).
4. Paste shared cells A–G verbatim. Cell D = augmentation spec (implement it in your framework's API).

## 3. The experiment grid — 4 runs TODAY (same in both frameworks)

| Run | Architecture | Strategy |
|---|---|---|
| R1 | ResNet50 | frozen backbone (train head only) |
| R2 | ResNet50 | fine-tune last block, LR 1e-5..1e-4 |
| R3 | EfficientNet-B0 | frozen backbone |
| R4 | EfficientNet-B0 | fine-tune, LR 1e-5..1e-4 |

- Seed 42, Adam/AdamW, batch 32, cross-entropy, **max 15 epochs, early stopping on val loss patience 3**.
- Log for EVERY run (append to `results/metrics_<fw>.json`): `epochs_run, seconds_per_epoch, total_train_seconds, peak_gpu_memory_mb, trainable_params, total_params, saved_model_size_mb, best_val_loss, best_val_acc`.
- Loss curves for every run. Best run by **validation** = your final model → test set evaluated **ONCE**.
- If time is short: drop R5, cut epochs to 10 — **never** drop GradCAM, final test eval, or outputs.

## 4. Preprocessing rules (prediction parity with the API — critical)

Normalization is **baked into the saved model**; the API only resizes + passes 0–1 (PT) or 0–255 (TF) pixels.

- **TF:** ResNet50 → `preprocess_input` layer inside the model; EfficientNetB0 → raw 0–255 (Rescaling inside model). Save as `model_T20.keras`.
- **PT:** ImageNet normalize inside the `nn.Sequential` you save (mean/std), input = float tensor [0,1], `1x3x224x224`. Export **TorchScript**: `torch.jit.trace(model.eval(), example)` → `torch.jit.save(...)` to `model_T20.pt`. Verify with `torch.jit.load(..., map_location="cpu")` in a fresh runtime.
- All images: `convert("RGB")` → resize 224×224 (3,067 of 7,200 are grayscale!).

## 5. GradCAM (mandatory, both models)

Same fixed images from shared_cells.py CELL G (6 paths, +1 misclassified — Seif appends it later):
- **PT:** ResNet50 `model.layer4[-1]`, EfficientNet-B0 `model.features[-1]`
- **TF:** ResNet50 `conv5_block3_out`, EfficientNetB0 `top_activation`
Save the figures into the notebook outputs (they must be visible when the notebook is opened).

## 6. Hand off to Seif (tonight / early Monday)

1. Push your notebook **with outputs saved** (run top-to-bottom in ONE session first):
   ```bash
   git add notebooks/<yours>.ipynb results/metrics_<yours>.json
   git commit -m "<fw>: R1-R4, final eval, GradCAM"; git push
   ```
2. Export the final model, then commit and push it too (everything lives on GitHub):
   ```bash
   git add model/model_T20.<pt|keras>
   git commit -m "<fw>: final model"; git push
   ```
   ⚠️ GitHub rejects files >100 MB — check `ls -lh model/` first. EfficientNet-B0 (~20 MB) is
   always safe; ResNet50 (~90–100 MB) may need half-precision export. Tell Seif if a push fails.
3. Write your report section (plan §8, sections 4 & 5: architecture, transfer-learning setup,
   hyperparameters, loss curves, test metrics, confusion matrix) — fill
   `report/section_<yours>.md` and push it, or send it to Seif.

## 7. Rules that cost marks (plan Section 0)

- Never call an external AI API; never serve a model you didn't train.
- Same split, seed, batch, epochs budget, GPU type across frameworks (fairness).
- Test set: evaluate ONCE, only your final model. Never tune on test.
- No vertical flip / hue / saturation augmentation; val & test get NO augmentation.
- Notebooks submitted **with outputs visible** — graders do not re-run them.
