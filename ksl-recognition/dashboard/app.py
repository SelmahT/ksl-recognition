"""
KSL Recognition dashboard: entry point.
Run with:  streamlit run app.py

Defines the sidebar menu (page names and icons). Keeping emojis here, not in file
names, avoids broken characters when the project is zipped or moved between computers.
"""
import streamlit as st

st.set_page_config(page_title="KSL Recognition", page_icon="🤟", layout="wide")

pages = [
    st.Page("home.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/1_Dataset.py", title="Dataset", icon="📊"),
    st.Page("pages/2_Model_showdown.py", title="Model showdown", icon="🏆"),
    st.Page("pages/3_Try_it_live.py", title="Try it live", icon="📷"),
    st.Page("pages/4_Stress_test.py", title="Stress test", icon="🌪️"),
]
st.navigation(pages).run()
