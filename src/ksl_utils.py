"""
ksl_utils.py
============
Shared helper functions for the KSL Recognition term paper (DSA 4050-XA).

Every notebook imports this file so that figures, tables, models and results
are saved in the SAME place with the SAME naming style. That makes writing
the final report easy: everything the report needs is already in outputs/.

Author : Selmah Tzindori (669602)
Course : DSA 4050-XA Deep Learning, USIU-Africa, Fall 2026
"""

import os
import json
import random
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------------------------------------------
# On Colab the project lives on Google Drive. If we are NOT on Colab
# (e.g. testing on a laptop) we fall back to a local folder so the code
# still runs. You can also override the location with the KSL_ROOT env var.
DRIVE_ROOT = "/content/drive/MyDrive/DSA4050/ksl-recognition"
LOCAL_ROOT = os.path.join(os.path.expanduser("~"), "DSA4050", "ksl-recognition")
PROJECT_ROOT = os.environ.get(
    "KSL_ROOT", DRIVE_ROOT if os.path.isdir("/content/drive") else LOCAL_ROOT
)

# Every folder the project uses, in one dictionary.
PATHS = {
    "root":       PROJECT_ROOT,
    "raw":        os.path.join(PROJECT_ROOT, "data", "raw"),        # original downloads, never edited
    "processed":  os.path.join(PROJECT_ROOT, "data", "processed"),  # cleaned / resized images, splits
    "models":     os.path.join(PROJECT_ROOT, "models"),             # trained models (.keras, .pkl)
    "checkpoints":os.path.join(PROJECT_ROOT, "models", "checkpoints"),  # best weights saved during training
    "figures":    os.path.join(PROJECT_ROOT, "outputs", "figures"), # .png charts for report and slides
    "tables":     os.path.join(PROJECT_ROOT, "outputs", "tables"),  # .csv tables for report
    "logs":       os.path.join(PROJECT_ROOT, "outputs", "logs"),    # training histories, run logs
    "src":        os.path.join(PROJECT_ROOT, "src"),                # this helper file lives here
}

# Raw dataset backup on Drive (one zip) and its fast local copy on Colab's disk.
# Reading thousands of small images straight from Drive is very slow, so each
# session we unzip the Drive backup onto Colab's local disk and work from there.
RAW_ZIP = os.path.join(PATHS["raw"], "kslc_dataset.zip")
LOCAL_DATA = "/content/ksl_data" if os.path.isdir("/content") else os.path.join(PROJECT_ROOT, "_local_data")

# Pre-processed (cropped + resized) images: zip on Drive, unzipped locally each session
PROCESSED_ZIP = os.path.join(PATHS["processed"], "ksl_224.zip")
LOCAL_PROCESSED = "/content/ksl_224" if os.path.isdir("/content") else os.path.join(PROJECT_ROOT, "_local_224")

RESULTS_FILE = os.path.join(PROJECT_ROOT, "outputs", "results.json")         # key numbers for the report
EXPERIMENTS_FILE = os.path.join(PROJECT_ROOT, "outputs", "experiments.csv")  # one row per trained model

SEED = 42  # one seed for the whole project so results can be reproduced


def make_folders(verbose: bool = True) -> None:
    """Create every project folder (safe to run many times; existing folders are kept)."""
    for name, path in PATHS.items():
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"  ✔ {name:<12} {path}")


def restore_raw_data(local_dir: str = LOCAL_DATA) -> str:
    """
    Make the raw dataset available on Colab's fast local disk.
    - If it is already unzipped this session, do nothing.
    - Otherwise copy the zip from Drive and unzip it (about 1-2 minutes).
    Returns the local folder that holds the dataset.
    """
    import shutil, time, zipfile
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"Data already available at {local_dir}")
        return local_dir
    if not os.path.exists(RAW_ZIP):
        raise FileNotFoundError(f"{RAW_ZIP} not found. Run 01_data_collection first.")
    start = time.time()
    tmp_zip = os.path.join(os.path.dirname(local_dir), "kslc_dataset_tmp.zip")
    shutil.copy(RAW_ZIP, tmp_zip)                  # Drive -> local disk (one big file is fast)
    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(local_dir)                    # unzip on local disk
    os.remove(tmp_zip)                             # free space
    print(f"Data restored to {local_dir} in {time.time() - start:.0f}s")
    return local_dir


def restore_processed_data(local_dir: str = LOCAL_PROCESSED) -> str:
    """
    Make the pre-processed 224x224 images (made in 03_preprocessing) available locally.
    Same idea as restore_raw_data(), but for the much smaller processed zip.
    """
    import shutil, time, zipfile
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"Processed data already available at {local_dir}")
        return local_dir
    if not os.path.exists(PROCESSED_ZIP):
        raise FileNotFoundError(f"{PROCESSED_ZIP} not found. Run 03_preprocessing first.")
    start = time.time()
    tmp_zip = os.path.join(os.path.dirname(local_dir), "ksl_224_tmp.zip")
    shutil.copy(PROCESSED_ZIP, tmp_zip)
    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(local_dir)
    os.remove(tmp_zip)
    print(f"Processed data restored to {local_dir} in {time.time() - start:.0f}s")
    return local_dir


def load_split_catalogue():
    """Read the final catalogue (one row per kept image, with its train/val/test split)."""
    return pd.read_csv(os.path.join(PATHS["processed"], "catalogue_split.csv"))


def load_class_names() -> list:
    """Sign names in label-index order (index 0 = first name), saved by 03_preprocessing."""
    with open(os.path.join(PATHS["processed"], "class_names.json")) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# 2. REPRODUCIBILITY
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds so every run gives the same splits and (near) same results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf  # only if TensorFlow is installed
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"Random seed set to {seed}")


def _timestamp() -> str:
    """Current date-time as text, used to stamp every saved result."""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------------------------------
# 3. SAVING OUTPUTS FOR THE REPORT
# -----------------------------------------------------------------------------
def save_figure(fig, name: str, notebook: str, caption: str = "", dpi: int = 300) -> str:
    """
    Save a matplotlib figure as a high-resolution PNG in outputs/figures/.

    name     : short file name, e.g. "02_class_distribution"
    notebook : which notebook made it, e.g. "02_eda" (recorded in the log)
    caption  : one sentence describing the figure (re-used in the report)
    """
    path = os.path.join(PATHS["figures"], f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _log_output("figure", name, path, notebook, caption)
    print(f"Figure saved -> {path}")
    return path


def save_table(df: pd.DataFrame, name: str, notebook: str, caption: str = "", index: bool = False) -> str:
    """Save a pandas DataFrame as CSV in outputs/tables/ (opens in Excel/Word for the report)."""
    path = os.path.join(PATHS["tables"], f"{name}.csv")
    df.to_csv(path, index=index)
    _log_output("table", name, path, notebook, caption)
    print(f"Table saved  -> {path}")
    return path


def log_result(key: str, value, notebook: str) -> None:
    """
    Store one key number/fact for the report in outputs/results.json.
    Example: log_result("n_images_total", 6249, "01_data")
    Running it again with the same key overwrites the old value.
    """
    results = load_results()
    results[key] = {"value": value, "notebook": notebook, "saved_at": _timestamp()}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Result logged: {key} = {value}")


def load_results() -> dict:
    """Read every logged result (empty dict if nothing logged yet)."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def log_experiment(model_name: str, model_type: str, metrics: dict, notebook: str,
                   params: dict | None = None, notes: str = "") -> pd.DataFrame:
    """
    Add one row to outputs/experiments.csv for every trained model.
    This table becomes the model-comparison table in the report.

    model_type : "classic_ml" or "deep_learning"
    metrics    : e.g. {"accuracy": 0.91, "macro_f1": 0.90, "train_time_s": 120}
    params     : hyper-parameters used, e.g. {"epochs": 20, "lr": 1e-3}
    If the same model_name already exists, its row is replaced.
    """
    row = {"model": model_name, "type": model_type, **metrics,
           "params": json.dumps(params or {}), "notes": notes,
           "notebook": notebook, "saved_at": _timestamp()}
    if os.path.exists(EXPERIMENTS_FILE):
        df = pd.read_csv(EXPERIMENTS_FILE)
        df = df[df["model"] != model_name]             # drop old run of the same model
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(EXPERIMENTS_FILE, index=False)
    print(f"Experiment logged: {model_name} -> {metrics}")
    return df


def _log_output(kind: str, name: str, path: str, notebook: str, caption: str) -> None:
    """Keep a list of every saved figure/table (outputs/logs/outputs_index.csv) with captions."""
    index_file = os.path.join(PATHS["logs"], "outputs_index.csv")
    row = pd.DataFrame([{"kind": kind, "name": name, "path": path, "notebook": notebook,
                         "caption": caption, "saved_at": _timestamp()}])
    if os.path.exists(index_file):
        old = pd.read_csv(index_file)
        old = old[~((old["kind"] == kind) & (old["name"] == name))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(index_file, index=False)


# -----------------------------------------------------------------------------
# 4. SAVING MODELS AND TRAINING HISTORY
# -----------------------------------------------------------------------------
def model_path(name: str, ext: str = "keras") -> str:
    """Full Drive path for a model file, e.g. model_path("cnn_scratch") -> .../models/cnn_scratch.keras"""
    return os.path.join(PATHS["models"], f"{name}.{ext}")


def save_history(history, name: str, notebook: str) -> str:
    """Save a Keras training history (loss/accuracy per epoch) to outputs/logs/ as CSV."""
    hist = history.history if hasattr(history, "history") else history
    df = pd.DataFrame(hist)
    df.index.name = "epoch"
    path = os.path.join(PATHS["logs"], f"history_{name}.csv")
    df.to_csv(path)
    print(f"History saved -> {path}")
    return path


def plot_history(history, name: str, notebook: str):
    """Plot and save training vs validation accuracy and loss curves side by side."""
    hist = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        if metric in hist:
            ax.plot(hist[metric], label="train")
        if f"val_{metric}" in hist:
            ax.plot(hist[f"val_{metric}"], label="validation")
        ax.set_title(f"{name}: {metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    save_figure(fig, f"history_{name}", notebook, caption=f"Training curves for {name}.")
    return fig


# -----------------------------------------------------------------------------
# 5. QUICK SUMMARY
# -----------------------------------------------------------------------------
def project_summary() -> None:
    """Print what has been saved so far: number of figures, tables, models and results."""
    def count(folder, ext):
        return len([f for f in os.listdir(folder) if f.endswith(ext)]) if os.path.isdir(folder) else 0
    print("Project root :", PROJECT_ROOT)
    print("Figures      :", count(PATHS["figures"], ".png"))
    print("Tables       :", count(PATHS["tables"], ".csv"))
    print("Models       :", count(PATHS["models"], ".keras") + count(PATHS["models"], ".pkl"))
    print("Results      :", len(load_results()))
    n_exp = len(pd.read_csv(EXPERIMENTS_FILE)) if os.path.exists(EXPERIMENTS_FILE) else 0
    print("Experiments  :", n_exp)
