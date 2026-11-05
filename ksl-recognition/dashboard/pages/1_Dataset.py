"""Dataset page: what the data looks like and how it was prepared."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import utils as u

u.page_setup("Dataset")
u.check_assets()
R = u.results()
cat = u.catalogue()
NAMES = u.class_names()

st.title("📊 The dataset")
st.markdown("Photos of nine KSL signs taken by many different people on their phones, at home. "
            "Source: *Kenyan Sign Language Classification Challenge* (Zindi Africa), accessed through the "
            "Kaggle re-upload *KSLC Dataset*.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Original labelled photos", f"{R.get('n_labelled_total', 6249):,}")
c2.metric("Removed (no sign visible)", R.get("n_excluded_no_sign", 11))
c3.metric("Used for modelling", f"{len(cat):,}")
c4.metric("Images per sign", f"{cat['label'].value_counts().min()}–{cat['label'].value_counts().max()}")

tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Browse signs", "📈 Balance & split", "🔬 Image quality", "🧹 Cleaning"])

# ---------------------------------------------------------------- browse
with tab1:
    sign = st.selectbox("Choose a sign", NAMES, index=NAMES.index("Love") if "Love" in NAMES else 0)
    st.markdown(f"**{sign}:** {u.SIGN_INFO.get(sign, '')}")
    smp = u.samples()
    imgs = smp[smp["label"] == sign]
    if len(imgs):
        cols = st.columns(len(imgs))
        for col, (_, r) in zip(cols, imgs.iterrows()):
            col.image(Image.open(r["path"]), width="stretch")
        st.caption("Test-set photos after preprocessing (centre crop, 224×224). Note how signers, clothing and "
                   "backgrounds vary while the hand shape stays the same.")
    st.markdown("#### The 'average' photo of each sign")
    fig = u.figure("02_average_image_per_sign")
    if fig:
        st.image(fig, width="stretch")
        st.caption("Every photo of a sign averaged together. Backgrounds blur to grey; only the typical hand "
                   "and arm position remains (e.g. the X of crossed arms for *Love* and *Friend*).")

# ---------------------------------------------------------------- balance & split
with tab2:
    a, b = st.columns(2)
    split_counts = cat.groupby(["label", "split"]).size().reset_index(name="images")
    split_counts["split"] = split_counts["split"].map({"train": "Train (70%)", "val": "Validation (15%)",
                                                      "test": "Test (15%)"})
    fig = px.bar(split_counts, x="label", y="images", color="split",
                 color_discrete_map={"Train (70%)": u.GREEN, "Validation (15%)": u.AMBER, "Test (15%)": u.RED},
                 title="Images per sign, by data split", labels={"label": "", "images": "Images"})
    fig.update_layout(legend_title="", legend_orientation="h", legend_y=1.12, height=430)
    a.plotly_chart(fig, width="stretch")

    orient = cat.groupby("label")["orientation"].apply(lambda s: (s == "landscape").mean() * 100).reset_index()
    fig = px.bar(orient, x="label", y="orientation", title="Share of landscape photos per sign (before cropping)",
                 labels={"label": "", "orientation": "% landscape"}, color_discrete_sequence=[u.AMBER])
    fig.update_layout(height=430)
    b.plotly_chart(fig, width="stretch")
    st.markdown("- **Perfectly balanced:** ~694 photos per sign, so random guessing scores 11.1%.\n"
                "- **Orientation varied by sign** (15–28% landscape), a shortcut a model could exploit, so every "
                "photo was **centre-cropped to a square**.\n"
                "- The split is **stratified** (equal shares of each sign) and **group-aware** (near-identical "
                "photos never cross between train and test).")

# ---------------------------------------------------------------- quality
with tab3:
    metric = st.radio("Measure", ["brightness", "contrast", "sharpness"], horizontal=True)
    fig = px.box(cat, x="label", y=metric, color_discrete_sequence=[u.GREEN], points=False,
                 log_y=(metric == "sharpness"), labels={"label": "", metric: metric.capitalize()},
                 title=f"{metric.capitalize()} of photos per sign")
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")
    st.markdown("Lighting, contrast and sharpness are **almost identical across signs**, so the model cannot "
                "guess a sign from photo quality; it has to read the hands. The blurriest photos are most "
                "common for *You* and *Me* (hand movement).")
    fig = u.figure("02_blurriest_images")
    if fig:
        with st.expander("See the blurriest photos"):
            st.image(fig, width="stretch")

# ---------------------------------------------------------------- cleaning
with tab4:
    total = R.get("n_labelled_total", 6249)
    removed = R.get("n_excluded_no_sign", 11)
    fig = go.Figure(go.Funnel(
        y=["Downloaded labelled photos", "Readable & complete", "Show a sign (after review)",
           "Train set", "Validation set", "Test set"],
        x=[total, R.get("n_labelled_usable", total), total - removed,
           R.get("n_train", 4362), R.get("n_val", 935), R.get("n_test", 941)],
        marker_color=[u.GREY, u.GREY, u.GREEN, u.GREEN, u.AMBER, u.RED]))
    fig.update_layout(title="From raw download to model-ready data", height=420)
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "1. **Integrity check:** all 6,249 photos opened correctly (0 missing, 0 corrupt).\n"
        "2. **Near-duplicates:** only 25 photos (0.4%) had near-copies; they revealed the same signer appearing "
        "under several signs.\n"
        "3. **No-sign photos:** a skin-colour screen ranked suspicious photos, then a person reviewed them; "
        f"**{removed} photos** (blank surfaces, floors, fabric, a street scene) were removed.\n"
        "4. **A bias lesson:** the skin-colour rule missed the hands of **darker-skinned signers**, which is why "
        "a human made the final call and why learned features beat hand-written rules.")
    fig = u.figure("03_review_low_skin")
    if fig:
        with st.expander("See the photos reviewed by hand"):
            st.image(fig, width="stretch")
