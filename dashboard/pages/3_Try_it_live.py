"""Try it live: predict the sign in a photo and show where the model looked."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import utils as u

u.page_setup("Try it live")
u.check_assets()
NAMES = u.class_names()
THRESH = u.conf_threshold()

st.title("📷 Try it live")
st.markdown("Give the model a photo of one of the nine signs. It shows its prediction, how sure it is, and a "
            "**Grad-CAM heatmap** of the parts of the photo that drove its decision.")

# ---------------------------------------------------------------- choose an input
source = st.radio("Photo source", ["🖼️ Example test photo", "⬆️ Upload a photo", "📸 Webcam"], horizontal=True)
img, true_label = None, None

if source.startswith("🖼️"):
    smp = u.samples()
    c1, c2 = st.columns([1, 2])
    sign = c1.selectbox("Sign", NAMES)
    options = smp[smp["label"] == sign].reset_index(drop=True)
    k = c2.select_slider("Example", options=list(range(1, len(options) + 1)), value=1)
    row = options.iloc[k - 1]
    img, true_label = Image.open(row["path"]), row["label"]
    st.caption("Example photos come from the **test set**: the model never saw them during training.")
elif source.startswith("⬆️"):
    up = st.file_uploader("Photo of a KSL sign (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
else:
    st.caption("Frame your chest and hands like the dataset photos (face not needed), then take the photo.")
    shot = st.camera_input("Make one of the nine signs")
    if shot:
        img = Image.open(shot)

if img is None:
    st.info("Choose or take a photo to see a prediction.")
    st.stop()

# ---------------------------------------------------------------- predict
image = u.prepare(img)                       # centre crop + 224x224, same as training
with st.spinner("Thinking…"):
    probs = u.predict_dl(image)
order = np.argsort(probs)[::-1]
top, conf = int(order[0]), float(probs[order[0]])
classic_idx = u.predict_classic(image)

col_img, col_res = st.columns([1, 1.3])
with col_img:
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    if show_cam:
        cam = u.grad_cam(image, top)
        st.image(u.overlay(image, cam), caption="Red = regions that most influenced the prediction",
                 width="stretch")
    else:
        st.image(image, caption="What the model sees (centre crop, 224×224)", width="stretch")

with col_res:
    if conf >= THRESH:
        st.markdown(f"<div class='muted'>Predicted sign</div><div class='big-pred'>{NAMES[top]}</div>",
                    unsafe_allow_html=True)
        st.progress(conf, text=f"Confidence {conf:.0%}")
    else:
        st.markdown(f"<div class='muted'>Predicted sign</div><div class='unsure'>Not sure 🤔</div>",
                    unsafe_allow_html=True)
        st.progress(conf, text=f"Best guess **{NAMES[top]}** at only {conf:.0%} confidence "
                               f"(below the {THRESH:.0%} threshold)")
        st.caption(f"Below {THRESH:.0%} confidence the model is right much less often, so it asks for another photo "
                   "instead of guessing. Try better lighting, a steadier phone, or framing the hands in the centre.")

    st.markdown(f"**{NAMES[top]}:** {u.SIGN_INFO.get(NAMES[top], '')}")
    if true_label is not None:
        if NAMES[top] == true_label:
            st.success(f"✅ Correct: this photo is labelled **{true_label}**.")
        else:
            st.error(f"❌ Labelled **{true_label}** in the dataset. (Some dataset labels are themselves "
                     "questionable; see the evaluation notes.)")
    if classic_idx is not None:
        mark = "✅" if (true_label is None or NAMES[classic_idx] == true_label) else "❌"
        st.markdown(f"<span class='muted'>Classic model (HOG + SVM) says: <b>{NAMES[classic_idx]}</b> "
                    f"{mark if true_label else ''}</span>", unsafe_allow_html=True)

    df = pd.DataFrame({"sign": NAMES, "probability": probs}).sort_values("probability")
    fig = px.bar(df, x="probability", y="sign", orientation="h", text=[f"{p:.0%}" for p in df["probability"]],
                 color=df["sign"] == NAMES[top], color_discrete_map={True: u.GREEN, False: u.GREY},
                 labels={"sign": "", "probability": "Probability"})
    fig.add_vline(x=THRESH, line_dash="dot", line_color=u.RED)
    fig.update_layout(height=360, showlegend=False, xaxis_range=[0, 1.05], margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

with st.expander("How does this work?"):
    st.markdown(
        "1. The photo is **centre-cropped to a square and resized to 224×224 pixels**, exactly as during training.\n"
        "2. **EfficientNetB0**, a CNN pretrained on 1.3 million ImageNet photos and fine-tuned on 4,362 KSL "
        "photos, outputs a probability for each of the nine signs.\n"
        f"3. If the highest probability is below **{THRESH:.0%}**, the app says *Not sure*. On the test set, "
        f"answers above this threshold were correct **{u.results().get('acc_above_threshold', 0.959):.1%}** of the time.\n"
        "4. **Grad-CAM** traces the prediction back through the last convolutional layer to show which regions "
        "mattered most. Good predictions focus on the hands and fingers.")
