"""Model showdown: every classic and deep-learning model compared."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils as u

u.page_setup("Model showdown")
u.check_assets()
R = u.results()

st.title("🏆 Model showdown")
st.markdown("Twelve models were trained on the same training images and tested once on the same 941 unseen "
            "test images. **Classic machine learning** needs hand-crafted features (raw pixels or HOG edge "
            "directions); **deep learning** learns its own features from the photos.")

exp = u.experiments().copy()
exp["Model"] = exp["model"].map(u.pretty_model)
exp["Type"] = exp["type"].map({"classic_ml": "Classic ML", "deep_learning": "Deep learning"})
exp = exp.sort_values("test_accuracy", ascending=False)

tab1, tab2, tab3, tab4 = st.tabs(["🥇 Leaderboard", "⚖️ Size vs speed", "🔀 Confusions & signs", "📉 Training curves"])

# ---------------------------------------------------------------- leaderboard
with tab1:
    show = st.multiselect("Show", ["Deep learning", "Classic ML"], default=["Deep learning", "Classic ML"])
    d = exp[exp["Type"].isin(show)].sort_values("test_accuracy")
    fig = px.bar(d, x=d["test_accuracy"] * 100, y="Model", color="Type", orientation="h",
                 color_discrete_map={"Deep learning": u.GREEN, "Classic ML": u.GREY},
                 text=[f"{v:.1%}" for v in d["test_accuracy"]],
                 hover_data={"test_macro_f1": ":.3f", "val_accuracy": ":.3f", "train_accuracy": ":.3f"},
                 labels={"x": "Test accuracy (%)", "Model": ""}, title="Test accuracy of every model")
    fig.add_vline(x=100 / 9, line_dash="dash", line_color=u.RED, annotation_text="random guess 11.1%",
                  annotation_position="bottom right")
    fig.update_layout(height=520, xaxis_range=[0, 100], legend_title="")
    st.plotly_chart(fig, width="stretch")

    head = u.table("06_headline_metrics")
    c1, c2 = st.columns(2)
    for col, (_, r) in zip([c1, c2], head.iterrows()):
        col.markdown(f"<div class='card'><b>{r['model']}</b><br>"
                     f"Accuracy <b>{r['accuracy']:.1%}</b> (95% CI {r['accuracy_95ci']})<br>"
                     f"Macro-F1 <b>{r['macro_f1']:.3f}</b> (95% CI {r['macro_f1_95ci']})</div>",
                     unsafe_allow_html=True)
    st.caption("95% confidence intervals from 2,000 bootstrap resamples of the test set. They do not overlap, "
               "so the gap between deep learning and classic ML is not due to chance. (The deep model's score "
               "is 85.0–86.0% depending on the JPEG decoder used.)")

    st.markdown("#### Overfitting check")
    over = exp[["Model", "Type", "train_accuracy", "test_accuracy"]].melt(
        id_vars=["Model", "Type"], var_name="set", value_name="accuracy")
    over["set"] = over["set"].map({"train_accuracy": "Training", "test_accuracy": "Test"})
    fig = px.bar(over, x="Model", y="accuracy", color="set", barmode="group",
                 color_discrete_map={"Training": "#9CC9BD", "Test": u.GREEN},
                 labels={"accuracy": "Accuracy", "Model": ""})
    fig.update_layout(height=420, legend_title="", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption("SVM and Random Forest score 100% on training photos but 52–60% on new ones: they memorise. "
               "Deep models with data augmentation keep training and test accuracy much closer.")

# ---------------------------------------------------------------- size vs speed
with tab2:
    dl = exp[exp["Type"] == "Deep learning"].copy()
    fig = px.scatter(dl, x="n_params_millions", y=dl["test_accuracy"] * 100, size="fit_time_s",
                     color="Model", text="Model", size_max=55,
                     labels={"n_params_millions": "Model size (million weights)", "y": "Test accuracy (%)"},
                     hover_data={"fit_time_s": ":.0f", "predict_ms_per_image": ":.1f"},
                     title="Deep models: accuracy vs size (bubble = training time)")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    tbl = exp[["Model", "Type", "test_accuracy", "test_macro_f1", "n_params_millions", "fit_time_s",
               "predict_ms_per_image"]].rename(columns={
        "test_accuracy": "Test accuracy", "test_macro_f1": "Macro-F1", "n_params_millions": "Weights (M)",
        "fit_time_s": "Training time (s)", "predict_ms_per_image": "ms per image"})
    st.dataframe(tbl.style.format({"Test accuracy": "{:.1%}", "Macro-F1": "{:.3f}", "Weights (M)": "{:.2f}",
                                   "Training time (s)": "{:.0f}", "ms per image": "{:.2f}"}, na_rep="–"),
                 hide_index=True, width="stretch")
    st.markdown("- **CNN from scratch:** smallest (1.2M weights) and nearly as accurate, but took ~42 minutes to train.\n"
                "- **MobileNetV2:** pretrained, trained in ~7 minutes, built for phones.\n"
                "- **EfficientNetB0 (extended fine-tuning):** most accurate; pretrained ImageNet features adapted in three stages.")

# ---------------------------------------------------------------- confusions & signs
with tab3:
    per = u.table("06_per_sign")
    long = per.melt(id_vars="sign", value_vars=["f1_classic", "f1_dl"], var_name="model", value_name="F1")
    long["model"] = long["model"].map({"f1_classic": "Classic (HOG + SVM)", "f1_dl": "Deep (EfficientNetB0)"})
    fig = px.bar(long, x="sign", y="F1", color="model", barmode="group",
                 color_discrete_map={"Classic (HOG + SVM)": u.GREY, "Deep (EfficientNetB0)": u.GREEN},
                 title="F1-score per sign", labels={"sign": ""})
    fig.update_layout(height=430, legend_title="", yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.markdown("Deep learning improves **every** sign. The hardest remain the look-alike pairs: *Friend vs Love* "
                "(crossed arms, open hands vs fists) and *Church vs Mosque* (one hand over the other).")
    fig = u.figure("06_confusion_classic_vs_dl")
    if fig:
        st.image(fig, width="stretch")
    conf = u.table("06_top_confusions")
    conf.columns = ["True → predicted", "Deep model", "Classic model"]
    st.dataframe(conf.style.format({"Deep model": "{:.1%}", "Classic model": "{:.1%}"}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- training curves
with tab4:
    hist_files = {u.pretty_model(p.stem.replace("history_", "")): p for p in sorted(u.LOGS.glob("history_*.csv"))}
    if hist_files:
        choice = st.selectbox("Model", list(hist_files), index=len(hist_files) - 1)
        h = pd.read_csv(hist_files[choice])
        metric = st.radio("Metric", ["accuracy", "loss"], horizontal=True)
        fig = go.Figure()
        fig.add_scatter(y=h[metric], name="training", line_color="#9CC9BD")
        fig.add_scatter(y=h[f"val_{metric}"], name="validation", line_color=u.GREEN)
        fig.update_layout(title=f"{choice}: {metric} per epoch", xaxis_title="epoch", yaxis_title=metric,
                          height=430)
        st.plotly_chart(fig, width="stretch")
        st.caption("Transfer models first train only a new final layer (base frozen), then fine-tune the top layers "
                   "of the pretrained network with a very small learning rate, hence the jump part-way through.")
    else:
        st.info("No training histories found in assets/outputs/logs.")
