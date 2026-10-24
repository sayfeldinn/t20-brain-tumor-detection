# TensorFlow implementation (report §5) — for Refaat

> Fill every `<!-- FILL -->` in Markdown, then either push this file (`git add report/section_tensorflow.md`)
> or paste the content into `report/report_source.md` under "## 5. TensorFlow implementation".
> Copy numbers from your notebook outputs / `results/metrics_tensorflow.json` — never estimate.

## 5.1 Architecture & transfer learning setup

- Backbone(s): `tf.keras.applications.ResNet50(weights="imagenet", include_top=False)`, `tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False)`; new Dense(4, softmax) head.
- Frozen runs (R1/R3): base frozen (`trainable=False`), head trained.
- Fine-tune runs (R2/R4): which layer(s) unfrozen, **BatchNorm kept in inference mode** (`base(x, training=False)`), exact LR (1e-5..1e-4), epochs actually run, early-stopping trigger.
- Preprocessing **inside the model**: ResNet50 → `preprocess_input` layer; EfficientNetB0 → raw 0–255 input (Rescaling inside). Saved `.keras` takes `224x224x3` RGB.

## 5.2 Hyperparameters (state once, then reference)

| Param | Value |
|---|---|
| Seed | 42 (`tf.random.set_seed`, numpy, random) |
| Optimizer | <!-- FILL: Adam/AdamW, LR per run --> |
| Batch size | 32 |
| Loss | CategoricalCrossentropy |
| Max epochs / early stop | 15 / patience 3 on val loss |
| Augmentation | per shared spec (±10° rot, ±10% shift, 0.9–1.1 zoom, brightness/contrast 0.9–1.1) |

## 5.3 Run grid (R1–R4, +R5 if run)

| Run | Arch | Strategy | epochs_run | s/epoch | total s | peak GPU MB | best_val_loss | best_val_acc | trainable/total params |
|---|---|---|---|---|---|---|---|---|---|
| R1 | ResNet50 | frozen | | | | | | | |
| R2 | ResNet50 | fine-tune | | | | | | | |
| R3 | EffNet-B0 | frozen | | | | | | | |
| R4 | EffNet-B0 | fine-tune | | | | | | | |
| R5 | | | | | | | | | |

<!-- FILL: embed loss-curve figures for every run (from notebook outputs) here -->

## 5.4 Final model — single test-set evaluation

<!-- FILL: chosen run (by val), then test results from notebook output -->

- Chosen run: <!-- FILL -->
- Test Accuracy: <!-- FILL -->
- Macro Precision / Recall / F1: <!-- FILL -->
- Per-class P/R/F1: glioma <!-- -->, meningioma <!-- -->, notumor <!-- -->, pituitary <!-- -->
- Confusion matrix (rows = true): <!-- FILL: paste 4x4 -->

<!-- FILL: confusion-matrix figure -->
