# KSL Recognition Dashboard

Interactive Streamlit dashboard for the DSA 4050-XA term paper *Recognising Kenyan Sign Language with Deep Learning*.

| Page | What it shows |
|---|---|
| **Overview** | The problem, key results and how the project works |
| **📊 Dataset** | Sample photos per sign, class balance and split, image quality, cleaning steps |
| **🏆 Model showdown** | All 12 models: leaderboard with confidence intervals, overfitting, size vs speed, per-sign F1, confusions, training curves |
| **📷 Try it live** | Predict the sign in a test photo, an uploaded photo or a webcam shot, with confidence, a "not sure" threshold and a Grad-CAM heatmap |
| **🌪️ Stress test** | Blur, darken, rotate or add noise to a photo and compare the deep and classic models live |

## Run it

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. It needs the files from `dashboard_assets.zip` unzipped into `assets/` (see `assets/README.md`).

**Python:** 3.10–3.13 (TensorFlow requirement). `scikit-learn` is pinned to 1.6.1, the version the classic model was saved with in Colab.
