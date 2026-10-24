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
├── data/           # dataset — 7,200 images, committed to git
├── model/          # model_T20.pt / model_T20.keras — final models, committed to git
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
- Private repo. **Everything lives in git** — data, models, notebooks (team decision: no Drive).
- Notebooks are committed **with outputs saved**.
- Never commit checkpoints (`runs/`, `*.ckpt`, `*.pth`) or secrets (`kaggle.json`) — see `.gitignore`.
- ⚠️ GitHub rejects single files >100 MB — verify model size before `git add model/`.
