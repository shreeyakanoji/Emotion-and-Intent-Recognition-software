"""
theme.py

unified_app.py imports `apply_theme` from this module, but it wasn't
present in any of your uploaded files — this would have crashed the app
on launch with a ModuleNotFoundError. Added a minimal version so the app
runs; replace the CSS below with whatever styling you actually want.
"""

import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; }
        </style>
        """,
        unsafe_allow_html=True,
    )
