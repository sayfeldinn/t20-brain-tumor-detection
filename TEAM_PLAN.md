# T20 — Execution plan for Roles 2 & 3

**Refaat (Role 2, TensorFlow) · Moatasem (Role 3, PyTorch)**
**Deadline: Monday Oct 26. Everything must be in the repo by Monday 10:00 AM.
Today (Sun Oct 25) is training day. Work your own way — but these outputs must exist.**

Repo: https://github.com/sayfeldinn/t20-brain-tumor-detection (dataset included — just clone)

---

## 0. Non-negotiable rules (violations cost marks)

1. Never call an external AI API; only serve models you trained yourself (ImageNet weights allowed **and required**).
2. Use the repo's `split_manifest.csv` — **never resplit**, seed 42.
3. Test set: evaluated **once**, only on your final model. Never tune on test.
4. Same budget in both frameworks: batch 32, ≤15 epochs, early stopping patience 3 on val loss (fairness is graded).
5. Augmentation limits: ±10° rotation, ±10% shift/scale, 0.9–1.1 zoom, brightness/contrast 0.9–1.1. **No** vertical flip, hue/saturation changes. Validation/test get **no** augmentation.
6. Notebooks must be pushed **with outputs visible** (graders never re-run them).

---

## 1. Deliverables matrix — the definition of "done"

| # | Deliverable | File | Acceptance check (Seif verifies) |
|---|---|---|---|
| 1 | Notebook, full run | `notebooks/tensorflow_T20.ipynb` (Refaat) / `notebooks/pytorch_T20.ipynb` (Moatasem) | Opens with all outputs: loss curves, metrics tables, confusion matrix, GradCAM images |
| 2 | Final trained model | `model/model_T20.keras` (Refaat) / `model/model_T20.pt` (Moatasem) | Loads in a fresh runtime; >5 MB; API returns same predictions as the notebook |
| 3 | Run metrics | `results/metrics_tensorflow.json` / `results/metrics_pytorch.json` | All 4 runs (R1–R4) + final-model test metrics filled, no nulls left |
| 4 | Report section | `report/section_tensorflow.md` / `report/section_pytorch.md` | No `<!-- FILL -->` markers left; numbers copied from notebook outputs |
| 5 | Comparison evidence | `report/framework_evidence.md` (your column only) | Filled with numbers, not vibes |
| 6 | Git pushes on `main` | — | All of the above committed (command card in §7) |

**Nothing else.** Do not touch `app/`, `split_manifest.csv`, `notebooks/shared_cells.py`, the other
framework's notebook, or report §7/§10 (Seif owns those).

---

## 2. Experiment grid (both frameworks, identical)

| Run | Architecture | Strategy | Output needed |
|---|---|---|---|
| R1 | ResNet50 | frozen backbone, head only | loss curve + val metrics |
| R2 | ResNet50 | fine-tune last block, LR 1e-5…1e-4 | loss curve + val metrics |
| R3 | EfficientNet-B0 | frozen backbone | loss curve + val metrics |
| R4 | EfficientNet-B0 | fine-tune, LR 1e-5…1e-4 | loss curve + val metrics |
| R5 | — | *only if time*: one hyperparameter change (LR or batch) | optional |

→ **Choose your final model by validation metrics** (best fine-tuned run) → evaluate the test set
**once** → GradCAM → export. Log per run into `results/metrics_<fw>.json`:
`epochs_run, seconds_per_epoch, total_train_seconds, peak_gpu_memory_mb, trainable_params,
total_params, saved_model_size_mb, best_val_loss, best_val_acc`.

Paste shared cells A–G from `notebooks/shared_cells.py` **verbatim** into your notebook
(classes/seed/manifest/augmentation spec/metrics function/memory helper/GradCAM paths).

---

## 3. Framework-specific instructions

### Refaat — TensorFlow → `tensorflow_T20.ipynb` + `model_T20.keras`

- `tf.keras.applications.ResNet50` / `EfficientNetB0`, `weights="imagenet"`, `include_top=False`, new 4-class head.
- Preprocessing **inside the model**: ResNet50 → `preprocess_input` layer; EfficientNetB0 → raw 0–255 input (Rescaling lives inside the model). The saved `.keras` must take `224×224×3` RGB.
- Fine-tuning: call the base with `training=False` (keep BatchNorm in inference mode), low LR (1e-5…1e-4).
- GPU memory log: `tf.config.experimental.get_memory_info("GPU:0")`.
- GradCAM via `tf.GradientTape`. Target layers: ResNet50 `conv5_block3_out`, EfficientNet-B0 `top_activation`.
- Export `model_T20.keras`; verify `tf.keras.models.load_model` works in a **fresh runtime**; record exact TF/Keras versions (report §12).
- ⚠️ If the file exceeds **100 MB**, GitHub rejects the push — tell Seif immediately (fix: half-precision save).

### Moatasem — PyTorch → `pytorch_T20.ipynb` + `model_T20.pt`

- `torchvision.models.resnet50` / `efficientnet_b0` with ImageNet weights, new 4-class head.
- ImageNet normalization **inside** the module you save. Input contract: `float32` in `[0,1]`, shape `1×3×224×224` (the API resizes and divides by 255 — nothing else).
- Export **TorchScript**: `torch.jit.trace(model.eval(), example_input)` → save to `model_T20.pt`;
  verify with `torch.jit.load(path, map_location="cpu")` in a fresh runtime. (If trace chokes, use `torch.jit.script`.)
- GPU memory log: `torch.cuda.max_memory_allocated()`.
- GradCAM (hand-rolled or `pytorch-grad-cam`). Target layers: ResNet50 `model.layer4[-1]`, EfficientNet-B0 `model.features[-1]`.
- ⚠️ ResNet50 TorchScript ≈ 97 MB — run `ls -lh model/` before pushing; if rejected, tell Seif.

**Both:** always `convert("RGB")` first — 3,067 of 7,200 images are grayscale (mode L).

---

## 4. GradCAM requirements (identical for both models)

- Use the **exact 6 paths** in `shared_cells.py` CELL G. Seif appends a 7th (a misclassified case)
  once your final model exists — message him when you're done so he can add it and you paste it in.
- Figures must be **in the notebook outputs**: all 6 images (original + heatmap overlay),
  including at least one **misclassified** case.
- 2–3 sentences of commentary per model: which regions are highlighted (tumor mass / ventricle /
  background) and whether the two architectures differ.

---

## 5. Today's checkpoints (Sun Oct 25)

| Time | What Seif expects |
|---|---|
| **~14:00** | Message: "cloned, Colab running, R1 started" |
| **~18:00** | R1 + R2 pushed (partial notebook OK — commits tell the story) |
| **~22:00** | R3 + R4 done; final model chosen by val; test eval done once; GradCAM started |
| **Mon 10:00** | **All 6 deliverables in the repo — hard cutoff.** |

**Slip rule:** drop R5 first → cut epochs to 10 → shorten GradCAM to 4 images.
**Never drop:** final test eval, confusion matrix, saved notebook outputs, model export.

---

## 6. Colab setup (appendix)

1. Runtime → Change runtime type → **T4 GPU** (same GPU for both frameworks — fairness rule).
2. Get the data — pick one:
   - **Option A (recommended):** clone the repo inside Colab:
     ```python
     # private repo — put your GitHub PAT in Colab Secrets (🔑), never hardcode in the notebook
     !git clone https://<TOKEN>@github.com/sayfeldinn/t20-brain-tumor-detection.git
     ```
     `data/` and `split_manifest.csv` come with the clone; run everything from the repo root.
   - **Option B:** upload `data/` to session storage / your own Drive and set the path prefix.
3. Manifest prefix on Colab (CELL C): `df["path"] = "/content/t20-brain-tumor-detection/" + df["path"]`.
4. Paste shared cells A–G verbatim; implement CELL D's augmentation spec in your framework's API.
5. Save your final notebook **into the repo** (`notebooks/`) before the hand-off — Colab's Drive copy
   is only your backup (Colab disconnects mid-training is the plan's known risk).

---

## 7. Hand-off command card (end of your work)

```bash
git add notebooks/<yours>.ipynb results/metrics_<yours>.json report/section_<yours>.md report/framework_evidence.md model/
git commit -m "<fw>: R1-R4, final eval, GradCAM, final model"
git push
```

Then message Seif: **"done — final model = `<file>`, chosen run = `R_`, test acc = `X`"**

⚠️ Reminders before you push:
- Notebook **with outputs saved** (run top-to-bottom in one session, then save).
- Model file >5 MB (packaging rejects the dummy) and ≤100 MB (GitHub limit).
- No `<!-- FILL -->` markers left in your report section.
