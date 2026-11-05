# Notebooks

Run in order on Google Colab (GPU runtime). Each notebook starts with the standard setup cell
that mounts Google Drive and imports `ksl_utils.py`.

| # | Notebook | Purpose | Term paper part |
|---|---|---|---|
| 00 | `00_project_setup.ipynb` | Drive folders, helper module, GPU check, environment log | — |
| 01 | `01_data_collection.ipynb` | Download KSL dataset into `data/raw/` | (b) |
| 02 | `02_eda.ipynb` | Explore the dataset: orientation, quality, near-duplicates, average images | (b) |
| 03 | `03_preprocessing.ipynb` | Resize images, group-aware train/val/test split | (c) |
| 04 | `04_classic_ml.ipynb` | Classic ML baselines | (c) |
| 05 | `05_deep_learning.ipynb` | CNN and transfer-learning models | (c) |
| 06 | `06_evaluation.ipynb` | Compare all models, error analysis | (c) |

`_TEMPLATE.ipynb` is the starting point for every new notebook.

**Before committing a notebook:** Runtime → Restart and run all, then File → Download → .ipynb into this folder.
