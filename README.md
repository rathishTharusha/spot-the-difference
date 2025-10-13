# Spot the Difference — Enhanced ML Pipeline

This repository contains a full pipeline to detect added, removed, and changed objects between two images using:

- Open-vocabulary detection (OWL-ViT; optional GroundingDINO)
- Siamese ViT change localization (heatmaps)
- CLIP-assisted matching + IoU
- TTA + Weighted Boxes Fusion (WBF)
- Threshold tuning on a validation split

## Project structure

- `notebooks/` — Jupyter notebooks (enhanced and baseline)
- `src/` — Python modules (future extraction of notebook logic)
- `models/` — Saved weights/checkpoints
- `data/` — Data staging area (raw/processed). Do not commit large data.
- `outputs/` — Generated artifacts (predictions, figures)

Key files in root:
- `spot_the_difference_workflow.ipynb` — Baseline notebook
- `spot_the_difference_workflow_enhanced.ipynb` — Enhanced notebook
- `spot_the_difference_procedure.md` — Pipeline description
- `requirements.txt` — Python dependencies
- `.gitignore` — Ignore large/cache artifacts

## Quickstart

1. Create a virtual environment (recommended) and install requirements:

```bash
python -m pip install -r requirements.txt
```

2. Open the enhanced notebook and run cells top-to-bottom:
- `spot_the_difference_workflow_enhanced.ipynb`

3. Outputs:
- `submission.csv`, `eval_metrics.txt` at repo root
- Model weights in `models/` (if configured)

## Data layout

Expected layout for images:

```
data/
  train.csv
  test.csv
  data/
    <img_id>_1.png
    <img_id>_2.png
```

Adjust paths in the notebooks if your layout differs.

## Notes
- GPU is recommended; first run will download models.
- GroundingDINO is optional; code falls back to OWL-ViT.
- Consider widening the threshold tuning ranges for better performance.
