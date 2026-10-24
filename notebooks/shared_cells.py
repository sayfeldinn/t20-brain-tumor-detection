"""Shared code cells for BOTH notebooks (paste verbatim — notebooks must be self-contained).

Source of truth for Role 1 (Seif). Any change here must be re-pasted into both
notebooks by their owners (Refaat: TensorFlow, Moatasem: PyTorch).
"""
# ============================================================
# CELL A — constants (identical in both notebooks)
# ============================================================
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]   # index 0..3, do not change
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
# Split manifest: columns path,label,split with split in {train,val,test}.
# Manifest itself is committed at split_manifest.csv — never resplit.

# ============================================================
# CELL B — seeding
# ------------------------------------------------------------
# TensorFlow:
import os, random
import numpy as np
import tensorflow as tf
def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
seed_everything()
# PyTorch:
# import os, random
# import numpy as np
# import torch
# def seed_everything(seed=SEED):
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# seed_everything()

# ============================================================
# CELL C — manifest loading (paths are relative to T20/ repo root)
# ------------------------------------------------------------
# The dataset (data/) and the manifest are both IN THE REPO — after
# `git clone`, paths just work if the notebook runs from the repo root.
# On Colab: clone the repo (TEAMMATES.md §2) and prefix once, e.g.
#   df["path"] = "/content/t20-brain-tumor-detection/" + df["path"]
# Never re-generate the split.
# import pandas as pd
# df = pd.read_csv("split_manifest.csv")
# train_df = df[df.split == "train"].reset_index(drop=True)
# val_df   = df[df.split == "val"].reset_index(drop=True)
# test_df  = df[df.split == "test"].reset_index(drop=True)

# ============================================================
# CELL D — augmentation spec (train only; val/test: resize + normalize only)
# ------------------------------------------------------------
# PARAMETERS (same semantics in both frameworks):
#   - rotation: +/-10 degrees
#   - translation: +/-10% of width/height
#   - zoom/scale: 0.9 .. 1.1
#   - brightness: 0.9 .. 1.1 ; contrast: 0.9 .. 1.1
#   - horizontal flip: NOT applied (left/right asymmetry in MRI is not
#     guaranteed to be invariant) — if a run enables it, justify in report
#   - FORBIDDEN: vertical flip, hue/saturation changes, heavy distortions
#   - val/test: no augmentation, only resize to 224x224 + native preprocessing
# TensorFlow equivalent: RandomRotation(10/360), RandomTranslation(0.1,0.1),
#   RandomZoom(0.1), RandomBrightness(0.1), RandomContrast(0.1), no RandomFlip
# PyTorch equivalent: RandomAffine(degrees=10, translate=(0.1,0.1),
#   scale=(0.9,1.1)), ColorJitter(brightness=0.1, contrast=0.1), no RandomHorizontalFlip

# ============================================================
# CELL E — metrics function (copy verbatim into both notebooks)
# ------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# def evaluate_predictions(y_true, y_pred, classes=CLASSES, title=""):
#     """y_true/y_pred: integer labels 0..3. Prints report; returns dict + figs."""
#     acc = accuracy_score(y_true, y_pred)
#     rep = classification_report(y_true, y_pred, target_names=classes, digits=4,
#                                 output_dict=True, zero_division=0)
#     print(f"=== {title} ===")
#     print(f"Accuracy: {acc:.4f}")
#     print(classification_report(y_true, y_pred, target_names=classes,
#                                 digits=4, zero_division=0))
#     cm = confusion_matrix(y_true, y_pred)
#     fig, ax = plt.subplots(figsize=(5, 4))
#     im = ax.imshow(cm, cmap="Blues")
#     ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
#     ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
#     ax.set_xlabel("Predicted"); ax.set_ylabel("True")
#     for i in range(len(classes)):
#         for j in range(len(classes)):
#             ax.text(j, i, cm[i, j], ha="center", va="center",
#                     color="white" if cm[i, j] > cm.max() / 2 else "black")
#     ax.set_title(f"{title} confusion matrix")
#     fig.tight_layout()
#     return {"accuracy": acc,
#             "macro_precision": rep["macro avg"]["precision"],
#             "macro_recall": rep["macro avg"]["recall"],
#             "macro_f1": rep["macro avg"]["f1-score"],
#             "per_class": {c: rep[c] for c in classes},
#             "confusion_matrix": cm.tolist()}

# ============================================================
# CELL F — timing / memory logging helper
# ------------------------------------------------------------
# TensorFlow:
# import time, tensorflow as tf
# def gpu_mem_mb():
#     try: return tf.config.experimental.get_memory_info("GPU:0")["peak_bytes"] // 1024**2
#     except Exception: return 0
# def reset_gpu_mem():
#     try: tf.config.experimental.reset_memory_stats("GPU:0")
#     except Exception: pass
# PyTorch:
# import time, torch
# def gpu_mem_mb():
#     return torch.cuda.max_memory_allocated() // 1024**2 if torch.cuda.is_available() else 0
# def reset_gpu_mem():
#     if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
# Per-run record (append one dict per run, dump to results/metrics_<fw>.json):
# {framework, architecture, strategy, epochs_run, seconds_per_epoch,
#  total_train_seconds, peak_gpu_memory_mb, trainable_params, total_params,
#  saved_model_size_mb, best_val_loss, best_val_acc}

# ============================================================
# CELL G — fixed GradCAM test images (same for both models)
# ------------------------------------------------------------
# 6 fixed images + 1 misclassified image (added after final models exist).
GRADCAM_TEST_PATHS = [
    "data/Testing/glioma/Te-gl_100.jpg",
    "data/Testing/meningioma/Te-aug-me_100.jpg",
    "data/Testing/notumor/Te-no_100.jpg",
    "data/Testing/pituitary/Te-pi_100.jpg",
    "data/Testing/glioma/Te-gl_109.jpg",
    "data/Testing/pituitary/Te-pi_109.jpg",
    # ROLE 1 WILL APPEND ONE MISCLASSIFIED TEST IMAGE HERE after the final
    # models exist (same path for both frameworks' notebooks).
]
# GradCAM target layers:
#   PyTorch:    ResNet50 model.layer4[-1] ; EfficientNet-B0 model.features[-1]
#   TensorFlow: ResNet50 "conv5_block3_out" ; EfficientNetB0 "top_activation"
