# PyTorch implementation (report §4) — for Moatasem

> Fill every `<!-- FILL -->` in Markdown, then either push this file (`git add report/section_pytorch.md`)
> or paste the content into `report/report_source.md` under "## 4. PyTorch implementation".
> Tables below mirror plan §8 item 4. Copy numbers from your notebook outputs / `results/metrics_pytorch.json` — never estimate.

## 4.1 Architecture & transfer learning setup

- Backbone(s): `torchvision.models.resnet50(weights=IMAGENET1K_V1)`, `torchvision.models.efficientnet_b0(weights=IMAGENET1K_V1)`; new `Linear(..., 4)` head.
- Frozen runs (R1/R3): which layers were frozen, head structure, what trained.
- Fine-tune runs (R2/R4): which block(s) unfrozen, optimizer + LR used (exact values), epochs actually run, early-stopping trigger.
- Normalization baked into the saved model: mean/std values used, input contract `float32 [0,1], 1x3x224x224`.

## 4.2 Hyperparameters (state once, then reference)

| Param | Value |
|---|---|
| Seed | 42 |
| Optimizer | <!-- FILL: Adam/AdamW, betas, weight decay --> |
| Batch size | 32 |
| Loss | CrossEntropyLoss |
| Max epochs / early stop | 15 / patience 3 on val loss |
| Augmentation | per shared spec (±10° rot, ±10% shift, 0.9–1.1 zoom, brightness/contrast 0.9–1.1) |

## 4.3 Run grid (R1–R4, +R5 if run)

| Run | Arch | Strategy | epochs_run | s/epoch | total s | peak GPU MB | best_val_loss | best_val_acc | trainable/total params |
|---|---|---|---|---|---|---|---|---|---|
| R1 | ResNet50 | frozen | | | | | | | |
| R2 | ResNet50 | fine-tune | | | | | | | |
| R3 | EffNet-B0 | frozen | | | | | | | |
| R4 | EffNet-B0 | fine-tune | | | | | | | |
| R5 | | | | | | | | | |

<!-- FILL: embed loss-curve figures for every run (from notebook outputs) here -->

## 4.4 Final model — single test-set evaluation

<!-- FILL: chosen run (by val), then test results from notebook output -->

- Chosen run: <!-- FILL -->
- Test Accuracy: <!-- FILL -->
- Macro Precision / Recall / F1: <!-- FILL -->
- Per-class P/R/F1: glioma <!-- -->, meningioma <!-- -->, notumor <!-- -->, pituitary <!-- -->
- Confusion matrix (rows = true): <!-- FILL: paste 4x4 -->

<!-- FILL: confusion-matrix figure -->
