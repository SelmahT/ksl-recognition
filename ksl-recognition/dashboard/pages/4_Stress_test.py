"""Stress test: alter a photo and watch both models react."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import utils as u

u.page_setup("Stress test")
u.check_assets()
NAMES = u.class_names()
THRESH = u.conf_threshold()

st.title("🌪️ Stress test")
st.markdown("Real users will not take perfect photos. Pick a test photo, then **blur, darken, rotate or add "
            "noise** with the sliders and watch how the deep model and the classic model react.")

smp = u.samples()
c1, c2 = st.columns([1, 2])
sign = c1.selectbox("Sign", NAMES, index=NAMES.index("Seat") if "Seat" in NAMES else 0)
options = smp[smp["label"] == sign].reset_index(drop=True)
k = c2.select_slider("Example photo", options=list(range(1, len(options) + 1)), value=1)
original = u.prepare(Image.open(options.iloc[k - 1]["path"]))

DEFAULTS = {"blur": 0.0, "bright": 1.0, "rot": 0, "noise": 0}
def reset():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

for key, value in DEFAULTS.items():          # first visit: start from the defaults
    st.session_state.setdefault(key, value)

s1, s2, s3, s4 = st.columns(4)
blur = s1.slider("Blur (radius)", 0.0, 6.0, step=0.5, key="blur")
bright = s2.slider("Brightness", 0.2, 1.6, step=0.05, key="bright")
rot = s3.slider("Rotation (°)", -45, 45, step=5, key="rot")
noise = s4.slider("Noise (σ)", 0, 60, step=5, key="noise")
st.button("↺ Reset sliders", on_click=reset)

altered = u.alter(original, blur=blur, brightness=bright, rotation=rot, noise=noise)

def verdict(idx, conf=None):
    ok = NAMES[idx] == sign
    icon = "✅" if ok else "❌"
    text = f"{icon} **{NAMES[idx]}**"
    if conf is not None:
        text += f" ({conf:.0%} sure)"
    return text

p_orig, p_alt = u.predict_dl(original), u.predict_dl(altered)
c_orig, c_alt = u.predict_classic(original), u.predict_classic(altered)

a, b = st.columns(2)
with a:
    st.image(original, caption="Original", width="stretch")
    st.markdown(f"Deep model: {verdict(int(p_orig.argmax()), float(p_orig.max()))}")
    if c_orig is not None:
        st.markdown(f"Classic model: {verdict(c_orig)}")
with b:
    st.image(altered, caption="Altered", width="stretch")
    conf = float(p_alt.max())
    st.markdown(f"Deep model: {verdict(int(p_alt.argmax()), conf)}"
                + ("" if conf >= THRESH else "  ·  *the app would say **Not sure***"))
    if c_alt is not None:
        st.markdown(f"Classic model: {verdict(c_alt)}")

df = pd.DataFrame({"sign": NAMES, "Original": p_orig, "Altered": p_alt}).melt(
    id_vars="sign", var_name="photo", value_name="probability")
fig = px.bar(df, x="sign", y="probability", color="photo", barmode="group",
             color_discrete_map={"Original": "#9CC9BD", "Altered": u.GREEN},
             labels={"sign": "", "probability": "Deep model probability"})
fig.update_layout(height=360, legend_title="", yaxis_range=[0, 1.05])
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Across all 941 test photos")
rob = u.table("06_robustness")
long = rob.melt(id_vars="condition", value_vars=["deep_learning", "classic"], var_name="model", value_name="accuracy")
long["model"] = long["model"].map({"deep_learning": "Deep (EfficientNetB0)", "classic": "Classic (HOG + SVM)"})
fig = px.bar(long, x="condition", y="accuracy", color="model", barmode="group",
             color_discrete_map={"Deep (EfficientNetB0)": u.GREEN, "Classic (HOG + SVM)": u.GREY},
             text=[f"{v:.0%}" for v in long["accuracy"]], labels={"condition": "", "accuracy": "Test accuracy"})
fig.add_hline(y=1 / 9, line_dash="dash", line_color=u.RED, annotation_text="random guess")
fig.update_layout(height=430, legend_title="", yaxis_tickformat=".0%", yaxis_range=[0, 1])
st.plotly_chart(fig, width="stretch")
st.markdown("- **Darkness** barely matters to either model.\n"
            "- **Rotation** is where deep learning shines: at 30° the classic model falls to near-random guessing, "
            "while the deep model keeps about three-quarters of its accuracy.\n"
            "- **Heavy blur** is the deep model's weak spot, because it relies on fine finger detail. Hold the "
            "phone steady!")
