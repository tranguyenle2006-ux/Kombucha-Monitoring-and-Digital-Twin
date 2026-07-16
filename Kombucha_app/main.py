import streamlit as st
from pages import f1_dashboard, f2_dashboard, history

st.set_page_config(page_title="Kombucha Digital Twin", layout="wide")

page = st.sidebar.radio("Go to", ["First Fermentation (F1)", "Second Fermentation (F2)", "History"])

if page == "First Fermentation (F1)":
    f1_dashboard.show()
elif page == "Second Fermentation (F2)":
    f2_dashboard.show()
elif page == "History":
    history.show()