# Framework comparison — evidence log (plan Section 7 → report §7)

Each teammate fills their column **while working** (opinions need numbers).
Push updates as you go: `git add report/framework_evidence.md && git commit -m "evidence" && git push`

## 1. Ease of implementation & debugging

| Item | PyTorch (Moatasem) | TensorFlow (Refaat) |
|---|---|---|
| Training-pipeline LOC (data loader → eval) | | |
| Notable errors hit (list, short) | | |
| Time spent debugging (total, min) | | |
| Error-message quality (1–5 + example) | | |
| Inspecting intermediate tensors (ease 1–5) | | |
| GradCAM implementation difficulty (1–5, notes) | | |

## 2. Training speed & resources (same T4, batch 32 — copy from metrics_<fw>.json)

| Item | PyTorch | TensorFlow |
|---|---|---|
| ResNet50 frozen: s/epoch (avg) | | |
| ResNet50 fine-tune: s/epoch (avg) | | |
| EfficientNet-B0 frozen: s/epoch | | |
| EfficientNet-B0 fine-tune: s/epoch | | |
| Total train time all runs (min) | | |
| Peak GPU memory (MB, max over runs) | | |
| Data-pipeline notes (tf.data vs DataLoader) | | |

## 3. Flexibility (ease 1–5 + one concrete example each)

| Item | PyTorch | TensorFlow |
|---|---|---|
| Swapping backbone (ResNet ↔ EfficientNet) | | |
| Freeze/unfreeze layers | | |
| Hooks / GradCAM layer access | | |
| Custom training loop control | | |
| Custom layers / loss (if any) | | |

## 4. Performance & stability

| Item | PyTorch | TensorFlow |
|---|---|---|
| Best test Accuracy / macro F1 (final model) | | |
| Val-acc spread across epochs (stable?) | | |
| Sensitivity to LR (did any run diverge?) | | |
| Runs that needed restart | | |
| Framework version(s) used | | |

## Notes to quote in the report (fill as you work)

- **PyTorch (Moatasem):** _1–3 sentences you'd stand behind, with a number_ ...
- **TensorFlow (Refaat):** _1–3 sentences you'd stand behind, with a number_ ...
