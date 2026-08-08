"""
theme.py

Shared visual theme for the app: pastel-neon pink/blue/yellow palette, an
animated swirling gradient background, and glassy glowing panels. Import
and call apply_theme() once near the top of any Streamlit page.
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ---- animated swirl background ---- */
.stApp {
    background: linear-gradient(-45deg, #ffd6ef, #d6ecff, #fff6c2, #ecd6ff, #ffd6ef);
    background-size: 400% 400%;
    animation: gradientSwirl 20s ease infinite;
    position: relative;
}

@keyframes gradientSwirl {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp::after {
    content: "";
    position: fixed;
    top: 50%; left: 50%;
    width: 160vmax; height: 160vmax;
    transform: translate(-50%, -50%);
    background: conic-gradient(from 0deg, #ffb3e6, #b3ddff, #fff2a8, #d9b3ff, #ffb3e6);
    filter: blur(140px) opacity(0.45);
    animation: spinSwirl 50s linear infinite;
    z-index: 0;
    pointer-events: none;
}

@keyframes spinSwirl {
    to { transform: translate(-50%, -50%) rotate(360deg); }
}

.stApp::before {
    content: "";
    position: fixed;
    top: -20%; left: -20%;
    width: 140%; height: 140%;
    background:
        radial-gradient(circle at 20% 30%, rgba(255,110,199,0.30), transparent 40%),
        radial-gradient(circle at 80% 20%, rgba(120,200,255,0.30), transparent 40%),
        radial-gradient(circle at 50% 85%, rgba(255,235,120,0.30), transparent 40%);
    filter: blur(70px);
    animation: blobMove 22s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}

@keyframes blobMove {
    0%   { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(4%, 6%) rotate(12deg); }
}

/* ---- lift real content above the swirl ---- */
.main .block-container {
    position: relative;
    z-index: 1;
    background: rgba(255, 255, 255, 0.42);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 28px;
    padding: 2.2rem 2.6rem 3rem 2.6rem;
    box-shadow: 0 0 40px rgba(255, 140, 220, 0.15);
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ---- headings ---- */
h1 {
    color: #c2007f;
    font-weight: 800 !important;
    text-shadow: 0 0 14px rgba(255, 110, 199, 0.55);
    letter-spacing: 0.5px;
}
h2, h3 {
    color: #a10a6f;
    font-weight: 700 !important;
}

/* ---- buttons ---- */
.stButton button, [data-testid="stDownloadButton"] button, [data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #ff7fd4, #8fd6ff);
    color: #2b0030;
    border: none;
    border-radius: 999px;
    padding: 0.55em 1.5em;
    font-weight: 700;
    box-shadow: 0 0 16px rgba(255, 110, 199, 0.55);
    transition: all 0.2s ease;
}
.stButton button:hover, [data-testid="stDownloadButton"] button:hover {
    box-shadow: 0 0 26px rgba(255, 110, 199, 0.9), 0 0 20px rgba(140, 210, 255, 0.6);
    transform: translateY(-2px);
    color: #2b0030;
}

/* ---- metrics ---- */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255, 110, 199, 0.45);
    box-shadow: 0 0 22px rgba(255, 110, 199, 0.22);
}
[data-testid="stMetricValue"] {
    color: #c2007f;
}

/* ---- bordered containers (used for the live-scan panel) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 22px !important;
    border: 2px solid rgba(255, 110, 199, 0.55) !important;
    background: rgba(255, 255, 255, 0.5);
    animation: pulseGlow 2.4s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 12px rgba(255, 110, 199, 0.35); }
    50%      { box-shadow: 0 0 30px rgba(255, 110, 199, 0.85); }
}

/* ---- expanders ---- */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.55);
    border-radius: 16px;
    border: 1px solid rgba(120, 200, 255, 0.5);
}

/* ---- file uploader ---- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 240, 180, 0.4);
    border: 2px dashed #ff7fd4 !important;
    border-radius: 18px;
}

/* ---- radio / select pills ---- */
[data-testid="stRadio"] label, [data-testid="stSelectbox"] {
    font-weight: 500;
}

/* ---- divider ---- */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #ff9fdf, #9fd8ff, #fff2a8);
    border-radius: 2px;
}

/* ---- scrollbar, just for polish ---- */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #ff9fdf, #9fd8ff);
    border-radius: 10px;
}
</style>
"""


def apply_theme():
    """Injects the pastel-neon swirl theme. Call once near the top of a page."""
    st.markdown(CSS, unsafe_allow_html=True)
