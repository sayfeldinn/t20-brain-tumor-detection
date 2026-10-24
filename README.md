# T20 — Brain Tumor MRI Classification (AI Platforms, Assignment 1)

Private repo: `t20-brain-tumor-detection` — https://github.com/sayfeldinn/t20-brain-tumor-detection

```bash
git clone https://github.com/sayfeldinn/t20-brain-tumor-detection.git
```

Team: Seif (Role 1 — data, FastAPI, report), Ahmed Refaat (Role 2 — TensorFlow), Moatasem (Role 3 — PyTorch).

## Structure
```
T20/
├── report_T20.pdf
├── requirements.txt
├── app/            # FastAPI service (main.py, config.py)
├── data/           # dataset (not in git — synced via Drive)
├── model/          # model_T20.pt / model_T20.keras (not in git — synced via Drive)
├── notebooks/      # pytorch_T20.ipynb, tensorflow_T20.ipynb (committed with outputs)
├── results/        # metrics_pytorch.json, metrics_tensorflow.json
└── split_manifest.csv
```

## Run the API (from inside T20/)
```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
# /health, /predict, /docs
```

## Git rules
- Private repo. Never commit `data/`, model files, checkpoints, or secrets (see `.gitignore`).
- Notebooks are committed **with outputs saved**.
- Model artifacts travel via Google Drive; added locally only when building the final ZIP.
