"""
utils.py: shared helpers for the KSL Recognition dashboard.

Every page imports this module for:
  - paths to the assets (models, figures, tables, sample images)
  - cached loading of the models and result files
  - preprocessing identical to the training pipeline (centre crop -> 224x224)
  - prediction, Grad-CAM heatmaps and image alterations for the stress test
  - a consistent page style

Author: Selmah Tzindori (669602), DSA 4050-XA Deep Learning, USIU-Africa
"""

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

# -----------------------------------------------------------------------------
# Paths: everything the app needs lives in assets/ (unzipped dashboard_assets)
# -----------------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets"
FIG = ASSETS / "outputs" / "figures"
TAB = ASSETS / "outputs" / "tables"
LOGS = ASSETS / "outputs" / "logs"
SAMPLES = ASSETS / "samples"

IMG_SIZE = 224

# Keras is not thread-safe, and Streamlit runs every click in a new thread.
# One lock makes sure only one thread loads or uses the model at a time.
_MODEL_LOCK = threading.RLock()
GREEN, GREY, RED, AMBER = "#2E7D6B", "#B0B0B0", "#B5482B", "#E0A458"

# Short descriptions of each sign, used on the prediction page
SIGN_INFO = {
    "Church": "Flat hands stacked horizontally in front of the body.",
    "Enough/Satisfied": "Flat hand moves across the chest, palm down; hands horizontal.",
    "Friend": "Arms crossed over the chest with open hands.",
    "Love": "Arms crossed over the chest with closed fists.",
    "Me": "A hand or finger points to the signer's own chest.",
    "Mosque": "Fingers of one hand rest or tap on the other hand or wrist.",
    "Seat": "Two bent fingers of one hand rest on two fingers of the other.",
    "Temple": "Hands form a peak or are held vertically, fingertips up.",
    "You": "Index finger or thumb points forward, towards the viewer.",
}


# -----------------------------------------------------------------------------
# Page style
# -----------------------------------------------------------------------------
def page_setup(title: str, icon: str = "🤟"):
    """Common page style (CSS) and sidebar caption. Page config is set once in app.py."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        div[data-testid="stMetricValue"] {color: #2E7D6B;}
        .big-pred {font-size: 2.6rem; font-weight: 700; color: #2E7D6B; line-height: 1.1;}
        .unsure {font-size: 2.0rem; font-weight: 700; color: #B5482B; line-height: 1.1;}
        .muted {color: #6B7A76; font-size: 0.95rem;}
        .card {background: #F2F7F5; border-radius: 12px; padding: 1rem 1.2rem; height: 100%;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### 🤟 KSL Recognition")
        st.caption("DSA 4050-XA Deep Learning · USIU-Africa · Fall 2026")
        st.caption("Selmah Tzindori (669602) · Lecturer: Dr. Alex Wambua")


def check_assets():
    """Stop with a clear message if the assets folder is missing."""
    if not (ASSETS / "models" / "dl_best.keras").exists():
        st.error("Assets not found. Unzip **dashboard_assets.zip** into the `assets/` folder "
                 "next to `app.py` (it should contain `models/`, `outputs/`, `samples/`, "
                 "`class_names.json` and `catalogue_split.csv`).")
        st.stop()


# -----------------------------------------------------------------------------
# Cached data loading
# -----------------------------------------------------------------------------
@st.cache_data
def class_names() -> list:
    return json.loads((ASSETS / "class_names.json").read_text())


@st.cache_data
def results() -> dict:
    """Key numbers from results.json as {key: value}."""
    raw = json.loads((ASSETS / "outputs" / "results.json").read_text())
    return {k: v["value"] for k, v in raw.items()}


@st.cache_data
def experiments() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "outputs" / "experiments.csv")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv")


@st.cache_data
def catalogue() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "catalogue_split.csv")


@st.cache_data
def samples() -> pd.DataFrame:
    """Demo test images: file names are '<Sign>__<ImageID>.jpg' ('/' saved as '-')."""
    names = {n.replace("/", "-"): n for n in class_names()}
    rows = [{"path": str(p), "label": names.get(p.stem.split("__")[0], p.stem.split("__")[0]),
             "image_id": p.stem.split("__")[-1]} for p in sorted(SAMPLES.glob("*.jpg"))]
    return pd.DataFrame(rows)


def figure(name: str):
    """Path of a saved PNG figure, or None if missing."""
    p = FIG / f"{name}.png"
    return str(p) if p.exists() else None


def conf_threshold() -> float:
    return float(results().get("dashboard_conf_threshold", 0.8))


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _strip_keys(obj, keys: set):
    """Recursively remove the given keys from a nested dict/list (a Keras model config)."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(v, keys) for v in obj]
    return obj


def _unknown_keys(message: str) -> set:
    """Setting names an older Keras rejected, read from its error message."""
    import re
    found = set(re.findall(r"unexpected keyword argument '([A-Za-z_]+)'", message))
    for block in re.findall(r"Unrecognized keyword arguments passed to \w+: \{([^}]*)\}", message):
        found |= set(re.findall(r"'([A-Za-z_]+)'\s*:", block))
    return found


def _load_compatible(path: Path):
    """
    Load a .keras model saved by a NEWER Keras (Colab) in an OLDER Keras (laptop).
    Newer Keras versions write extra settings (e.g. 'quantization_config') that older
    versions do not recognise. If loading fails for that reason, those settings are
    removed from the model's config.json and loading is retried. Weights are untouched.
    """
    import tempfile
    import zipfile
    import keras

    removed, current = set(), path
    for _ in range(15):
        try:
            model = keras.models.load_model(current, compile=False)
            if removed:
                print(f"[dashboard] Loaded model after removing newer-Keras settings: {sorted(removed)}")
            return model
        except (TypeError, ValueError) as e:
            new = _unknown_keys(str(e)) - removed
            if not new:
                raise
            removed |= new
            with zipfile.ZipFile(path) as src:
                config = _strip_keys(json.loads(src.read("config.json")), removed)
                tmp = Path(tempfile.mkdtemp()) / "model_compat.keras"
                with zipfile.ZipFile(tmp, "w") as dst:
                    for item in src.infolist():
                        data = json.dumps(config).encode() if item.filename == "config.json" else src.read(item.filename)
                        dst.writestr(item, data)
            current = tmp
    raise RuntimeError("Could not make the model compatible with this Keras version.")


@st.cache_resource(show_spinner="Loading the deep-learning model (first time only, about a minute)…")
def dl_model():
    """Load the trained model once (thread-safe, compatible with older Keras versions)."""
    import keras
    path = ASSETS / "models" / "dl_best.keras"
    with _MODEL_LOCK:
        try:
            return _load_compatible(path)
        except AttributeError:              # Keras name-scope state reset by another thread
            keras.backend.clear_session()
            return _load_compatible(path)


@st.cache_resource(show_spinner="Loading the classic model…")
def classic_model():
    import joblib
    try:
        return joblib.load(ASSETS / "models" / "classic_best.pkl")
    except Exception as e:                       # e.g. scikit-learn version mismatch
        st.warning(f"Classic model could not be loaded ({e.__class__.__name__}); "
                   "showing deep-learning results only.")
        return None


# -----------------------------------------------------------------------------
# Preprocessing (identical to notebook 03): centre square crop -> 224x224 RGB
# -----------------------------------------------------------------------------
def prepare(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def predict_dl(image: np.ndarray) -> np.ndarray:
    """Probabilities for the 9 signs from the deep model (input: uint8 224x224x3)."""
    model = dl_model()
    with _MODEL_LOCK:
        return model(image[None].astype("float32"), training=False).numpy()[0]


def predict_classic(image: np.ndarray):
    """Predicted class index from the classic model (HOG + SVM), or None."""
    clf = classic_model()
    if clf is None:
        return None
    from skimage.feature import hog
    size = 64 if clf["features"] == "hog" else 32
    grey = np.asarray(Image.fromarray(image).convert("L").resize((size, size), Image.Resampling.BILINEAR),
                      dtype=np.float32) / 255.0
    feat = (hog(grey, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True) if clf["features"] == "hog" else grey.ravel())
    return int(clf["pipeline"].predict(feat[None].astype(np.float32))[0])


# -----------------------------------------------------------------------------
# Grad-CAM (same method as notebook 06)
# -----------------------------------------------------------------------------
def grad_cam(image: np.ndarray, class_idx: int = None) -> np.ndarray:
    """Heatmap (224x224, 0-1) of the regions that drove the prediction."""
    import keras
    import tensorflow as tf
    model = dl_model()
    with _MODEL_LOCK:
        return _grad_cam(model, image, class_idx)


def _grad_cam(model, image, class_idx):
    import keras
    import tensorflow as tf
    x = tf.convert_to_tensor(image[None].astype("float32"))
    nested = [l for l in model.layers if isinstance(l, keras.Model) and len(l.layers) > 20]
    with tf.GradientTape() as tape:
        if nested:
            base = nested[0]
            idx = model.layers.index(base)
            for l in model.layers[1:idx]:
                if l.name != "augmentation":
                    x = l(x)
            conv = base(x, training=False)
            tape.watch(conv)
            out = conv
            for l in model.layers[idx + 1:]:
                out = l(out, training=False)
        else:
            last_conv = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)][-1]
            conv, out = keras.Model(model.inputs, [last_conv.output, model.output])(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(out[0]))
        score = out[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights[0], axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0].numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (np.clip((1 - alpha) * image / 255.0 + alpha * heat, 0, 1) * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Image alterations for the stress test (same as notebook 06)
# -----------------------------------------------------------------------------
def alter(image: np.ndarray, blur=0.0, brightness=1.0, rotation=0, noise=0) -> np.ndarray:
    pil = Image.fromarray(image)
    if blur > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(blur))
    if brightness != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
    if rotation:
        pil = pil.rotate(rotation, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    arr = np.asarray(pil).astype(np.float32)
    if noise > 0:
        arr = arr + np.random.default_rng(42).normal(0, noise, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def pretty_model(name: str) -> str:
    """Readable model names for charts."""
    mapping = {"efficientnetb0_ft2": "EfficientNetB0 (extended FT)", "efficientnetb0": "EfficientNetB0",
               "mobilenetv2": "MobileNetV2", "cnn_scratch": "CNN from scratch"}
    if name in mapping:
        return mapping[name]
    feat, _, model = name.partition("+")
    feat = {"hog": "HOG", "raw_pixels": "Raw pixels"}.get(feat, feat)
    model = {"svm_rbf": "SVM", "logreg": "Logistic Reg.", "random_forest": "Random Forest",
             "knn": "k-NN"}.get(model, model)
    return f"{feat} + {model}"
