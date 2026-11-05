"""
KSL Recognition dashboard: Overview page.
Overview page (opened from app.py)
"""
import streamlit as st

import utils as u

u.page_setup("Overview")
u.check_assets()
R = u.results()

# ---------------------------------------------------------------- hero
st.title("🤟 Recognising Kenyan Sign Language with Deep Learning")
st.markdown(
    "<p class='muted'>Term paper, DSA 4050-XA Deep Learning · United States International "
    "University–Africa · Selmah Tzindori (669602) · Lecturer: Dr. Alex Wambua</p>",
    unsafe_allow_html=True)

st.markdown(
    "**Kenyan Sign Language (KSL)** is the main language of Kenya's Deaf community, yet few hearing "
    "Kenyans understand it. This project trains computer-vision models to recognise **nine everyday KSL signs** "
    "from ordinary smartphone photos, a first step towards tools that help hearing people understand "
    "signers. Classic machine learning is compared with deep learning on the same data.")

# ---------------------------------------------------------------- key numbers
best_dl = R.get("best_dl_test_accuracy", 0.86)
best_cl = R.get("best_classic_test_accuracy", 0.60)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Labelled images", f"{R.get('n_images_after_cleaning', 6238):,}")
c2.metric("KSL signs", R.get("n_classes", 9))
c3.metric("Best deep model", f"{best_dl:.1%}", help="EfficientNetB0 with extended fine-tuning, test accuracy")
c4.metric("Best classic model", f"{best_cl:.1%}", help="HOG features + SVM, test accuracy")
c5.metric("Deep-learning gain", f"+{R.get('dl_vs_classic_gain_pts', 25.9):.1f} pts",
          delta=f"top-2 accuracy {R.get('dl_top2_accuracy', 0.947):.1%}", delta_color="off")

st.divider()

# ---------------------------------------------------------------- the signs
left, right = st.columns([1.15, 1])
with left:
    st.subheader("The nine signs")
    fig = u.figure("01_sample_grid")
    if fig:
        st.image(fig, width="stretch")
with right:
    st.subheader("How the project works")
    steps = [
        ("📥", "Data", "6,249 phone photos from the Zindi KSL challenge (via Kaggle); 11 invalid photos removed."),
        ("🔍", "Exploration", "Balanced classes, look-alike signs, blur and near-duplicate checks."),
        ("✂️", "Preparation", "Centre crop to a square, resize to 224×224, 70/15/15 group-aware split."),
        ("📐", "Classic ML", "HOG and raw-pixel features with SVM, Random Forest, k-NN and Logistic Regression."),
        ("🧠", "Deep learning", "CNN from scratch, MobileNetV2 and EfficientNetB0 with transfer learning."),
        ("🧪", "Evaluation", "Confidence intervals, Grad-CAM heatmaps and robustness tests."),
    ]
    for icon, name, text in steps:
        st.markdown(f"**{icon} {name}:** {text}")
    st.markdown("")
    st.info(f"**Why deep learning?** The signs differ only in fine hand shapes (open vs closed hands, "
            f"finger positions) while signers, clothing and backgrounds change completely. Hand-crafted "
            f"features reach {best_cl:.0%}; features learned by a CNN reach {best_dl:.0%}.")

st.divider()

# ---------------------------------------------------------------- navigation
st.subheader("Explore the dashboard")
n1, n2, n3, n4 = st.columns(4)
cards = [
    (n1, "📊 Dataset", "What the photos look like, class balance, quality and the cleaning steps."),
    (n2, "🏆 Model showdown", "All 12 models compared: accuracy, size, speed, confusions and training curves."),
    (n3, "📷 Try it live", "Upload a photo or use your webcam: see the predicted sign and where the model looked."),
    (n4, "🌪️ Stress test", "Blur, darken, rotate or add noise to a photo and watch the prediction change."),
]
for col, title, text in cards:
    col.markdown(f"<div class='card'><b>{title}</b><br><span class='muted'>{text}</span></div>",
                 unsafe_allow_html=True)
st.caption("Use the sidebar on the left to open each page.")
