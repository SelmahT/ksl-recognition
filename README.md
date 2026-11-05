# Recognising Kenyan Sign Language with Deep Learning

Can a computer recognise **Kenyan Sign Language (KSL)** from an ordinary phone photo, and does **deep learning** beat **classic machine learning** at it? This project trains and compares 12 models on 6,238 photos of nine everyday KSL signs.

**Term paper · DSA 4050-XA Deep Learning · United States International University–Africa (Fall 2026)**  
Selmah Tzindori (669602) · Lecturer: Dr. Alex Wambua

🔗 **Live dashboard:** [ksl-recognition.streamlit.app](https://ksl-recognition.streamlit.app)  
📄 **Report:** [`report/`](report/) · 🎞️ **Slides:** [`slides/`](slides/)

## Key results

| Model | Type | Test accuracy | Macro-F1 |
|---|---|---|---|
| **EfficientNetB0 (fine-tuned in 3 stages)** | Deep learning | **86.0%** (95% CI 82.7–87.2%) | **0.859** |
| EfficientNetB0 | Deep learning | 84.9% | 0.849 |
| CNN from scratch | Deep learning | 84.3% | 0.843 |
| MobileNetV2 | Deep learning | 81.6% | 0.817 |
| HOG + SVM (best classic) | Classic ML | 60.0% (95% CI 57.1–63.2%) | 0.603 |

- Deep learning beats every classic model, by **+25.9 points** for the best one; random guessing scores 11.1%.
- **Grad-CAM** shows the model looks at the hands and fingers, not backgrounds or clothing.
- **Top-2 accuracy 94.7%**; at a confidence threshold of 0.80 the model answers 79% of photos with 95.9% accuracy.
- Robust to darkness and rotation (30°: 74% vs 17% for classic ML); weakest under heavy blur.

## Dataset

Images from the *Kenyan Sign Language Classification Challenge* (Zindi Africa), accessed through the Kaggle re-upload [KSLC Dataset](https://www.kaggle.com/datasets/awsaf49/kslc-dataset) by awsaf49. Nine signs: Church, Enough/Satisfied, Friend, Love, Me, Mosque, Seat, Temple, You. 6,249 labelled photos; 11 photos showing no sign were removed. No licence is stated on Kaggle, so both sources are credited. The dataset itself is **not** stored in this repo.

## Repository layout

```
ksl-recognition/
├── notebooks/    Colab notebooks 00–06, run in order (with outputs and findings)
├── src/          ksl_utils.py, shared helpers used by every notebook
├── dashboard/    Streamlit app (app.py) + assets (trained models, results, sample photos)
├── outputs/      figures produced by the notebooks
├── report/       term paper (Word + PDF)
├── slides/       presentation (PowerPoint)
└── data/, models/   placeholders; data and full models live on Google Drive
```

| Notebook | Purpose |
|---|---|
| `00_project_setup` | Drive folders, helper module, GPU check, environment log |
| `01_data_collection` | Download from Kaggle, integrity checks, image catalogue |
| `02_eda` | Balance, orientation, quality, near-duplicates, average images |
| `03_preprocessing` | Remove invalid photos, centre crop to 224×224, stratified group-aware split |
| `04_classic_ml` | Raw-pixel and HOG features × Logistic Regression, SVM, Random Forest, k-NN |
| `05_deep_learning` | CNN from scratch, MobileNetV2, EfficientNetB0 (transfer learning) |
| `06_evaluation` | Confidence intervals, confusion matrices, Grad-CAM, confidence, robustness |

## How to run

**Notebooks (Google Colab, T4 GPU):** open `notebooks/00_project_setup.ipynb` in Colab, run it once, add a Kaggle API token to Colab Secrets as `KAGGLE_API_TOKEN`, then run notebooks 01–06 in order. Data, models and outputs are stored on Google Drive under `MyDrive/DSA4050/ksl-recognition/`.

**Dashboard (local, Python 3.10–3.13):**
```bash
cd dashboard
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## License

Code: MIT (see `LICENSE`). Dataset: see the original sources above.