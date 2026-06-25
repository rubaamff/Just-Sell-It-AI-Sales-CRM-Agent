# macOS OpenMP workaround — faiss-cpu and numpy can each ship their own
# libomp.dylib; without this env var the second initializer aborts the
# whole process with "OMP: Error #15 ... already initialized". Must be
# set BEFORE any import that pulls in numpy / faiss / torch.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from agents.campaign_agent import CampaignAgent
from agents.opportunity_agent import (
    evaluate_single_opportunity,
    evaluate_opportunity_auto,
    evaluate_opportunity_custom,
)
from agents.email_agent import (
    is_gmail_configured,
    fetch_unread_emails,
    classify_and_draft,
    send_email as gmail_send,
    is_signed_in,
    get_authenticated_email,
    sign_in_gmail,
    sign_out_gmail,
)
from agents.proposal_agent import build_agent as build_proposal_agent
from utils.scorer import score_in_batches
from utils.email_finder import find_email_for_company
from utils.rag import initialize_rag
from utils.proposal_pdf import generate_proposal_pdf

st.set_page_config(
    page_title="Just Sell It | AI Sales & CRM Agent",
    page_icon="🇸🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Share+Tech+Mono&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ─── Hide a narrow, safe set of Streamlit chrome ────────────────
   Anything broader (header height, button[kind="header"], the whole
   toolbar) breaks the sidebar toggle and its widgets. */
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }
.viewerBadge_link__qRIco, .viewerBadge_container__r5tak { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ─── Page layout rhythm ────────────────────────────────────────── */
.stMainBlockContainer, .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px !important;
}
section[data-testid="stSidebar"] { padding-top: 0.5rem !important; }

.stApp {
    background-color: #050b18;
    background-image:
        linear-gradient(rgba(13,60,120,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(13,60,120,0.07) 1px, transparent 1px);
    background-size: 40px 40px;
}

body, p, div, span, label { font-family: 'Inter', sans-serif; color: #c8deff; }

div[data-testid="stSidebar"] {
    background: #040a14;
    border-right: 1px solid #0d3060;
}
div[data-testid="stSidebar"] * { color: #c8deff; }

/* ─── Branded topbar ────────────────────────────────────────────── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 22px 28px;
    margin-bottom: 28px;
    background: linear-gradient(135deg, rgba(13,58,138,0.18) 0%, rgba(8,20,40,0.6) 60%, rgba(8,20,40,0.4) 100%);
    border: 1px solid #1a3a6a;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 32px rgba(0,0,0,0.4);
}
.topbar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(77,159,255,0.7), transparent);
}
.topbar::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 600px 100px at 50% 0%, rgba(77,159,255,0.10), transparent 70%);
    pointer-events: none;
}
.brand {
    display: flex;
    flex-direction: column;
    gap: 4px;
    position: relative;
    z-index: 1;
}
.header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f6ff;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin: 0;
}
.header-title span {
    background: linear-gradient(135deg, #4d9fff 0%, #7cc1ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-sub {
    font-family: 'Share Tech Mono', monospace;
    color: #8fb6dc;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    margin: 0;
    text-transform: uppercase;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: rgba(61,214,140,0.10);
    border: 1px solid rgba(61,214,140,0.35);
    border-radius: 100px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #6ee5a8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
    white-space: nowrap;
}
.topbar-status .dot {
    width: 8px; height: 8px;
    background: #3dd68c;
    border-radius: 50%;
    box-shadow: 0 0 10px #3dd68c, 0 0 4px #3dd68c;
    animation: pulse 2.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.92); }
}

.metric-card {
    background: linear-gradient(160deg, #0c1830 0%, #0a1428 100%);
    border: 1px solid #1a3a6a;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 4px 16px rgba(0,0,0,0.3);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 12%; right: 12%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(77,159,255,0.6), transparent);
}
.metric-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(77,159,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.metric-card:hover {
    border-color: #2a7adf;
    box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 0 28px rgba(77,159,255,0.18), 0 8px 24px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}
.metric-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #5db0ff;
    letter-spacing: -0.03em;
    line-height: 1;
    text-shadow: 0 0 24px rgba(77,159,255,0.55);
}
.metric-lbl {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #a8c4e0;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 500;
}

.filter-header {
    font-family: 'Share Tech Mono', monospace;
    color: #a8c4e0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
    margin-top: 22px;
    border-left: 2px solid #4d9fff;
    padding-left: 10px;
    font-weight: 500;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    margin-right: 6px;
    letter-spacing: 0.05em;
    font-weight: 500;
}
.badge-sector { background: rgba(77,159,255,0.14); color: #7cc1ff; border: 1px solid rgba(77,159,255,0.35); }
.badge-startup { background: rgba(61,214,140,0.14); color: #6ee5a8; border: 1px solid rgba(61,214,140,0.35); }
.badge-emp { background: rgba(167,139,250,0.14); color: #c0aaff; border: 1px solid rgba(167,139,250,0.35); }

div.stButton > button {
    background: linear-gradient(145deg, #0a1428, #0e1c34) !important;
    border: 1px solid #1a3a6a !important;
    border-radius: 10px !important;
    color: #d4e6ff !important;
    text-align: left !important;
    padding: 14px 16px !important;
    transition: all 0.2s ease !important;
    font-size: 0.86rem !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
}
div.stButton > button:hover {
    border-color: #4d9fff !important;
    background: linear-gradient(145deg, #0e1f3a, #122849) !important;
    box-shadow: 0 0 20px rgba(77,159,255,0.18) !important;
    transform: translateY(-1px) !important;
    color: #f0f6ff !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a5dbb 0%, #2a7adf 100%) !important;
    border: 1px solid #4d9fff !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    text-align: center !important;
    box-shadow: 0 4px 14px rgba(77,159,255,0.28), 0 1px 0 rgba(255,255,255,0.15) inset !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2a7adf 0%, #4d9fff 100%) !important;
    box-shadow: 0 6px 22px rgba(77,159,255,0.45), 0 1px 0 rgba(255,255,255,0.25) inset !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}
div.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 10px rgba(77,159,255,0.3), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, #0a1428 0%, #0c1830 100%) !important;
    border: 1px solid #1a3a6a !important;
    border-radius: 14px !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 4px 16px rgba(0,0,0,0.28) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: '';
    position: absolute;
    top: 0; left: 8%; right: 8%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(77,159,255,0.45), transparent);
    pointer-events: none;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #2a5a9a !important;
    box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 6px 22px rgba(0,0,0,0.36) !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #081428 !important;
    border: 1px solid #1a3a6a !important;
    color: #f0f6ff !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #8fb6dc !important;
    opacity: 0.7;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #4d9fff !important;
    box-shadow: 0 0 14px rgba(77,159,255,0.22) !important;
    outline: none !important;
}

div[data-testid="stTabs"] { border-bottom: 1px solid #1a3a6a; }
div[data-testid="stTabs"] button {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #a8c4e0 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 12px 14px !important;
}
div[data-testid="stTabs"] button:hover {
    color: #d4e6ff !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #5db0ff !important;
    border-bottom: 2px solid #4d9fff !important;
    background: transparent !important;
    text-shadow: 0 0 12px rgba(77,159,255,0.35);
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #050b18; }
::-webkit-scrollbar-thumb { background: #1a5dbb; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4d9fff; }

.no-results {
    text-align: center;
    padding: 60px 20px;
    font-family: 'Share Tech Mono', monospace;
    color: #6fa0d4;
    font-size: 0.95rem;
    letter-spacing: 0.1em;
}
hr { border-color: #1a4a85 !important; }
.stCaption, .stCaption * { color: #a8c4e0 !important; font-size: 0.8rem !important; }
p { color: #b6d0ec; }
h1,h2,h3,h4 { color: #e8f2ff !important; font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

div[data-testid="stMarkdownContainer"] code {
    background: rgba(77,159,255,0.14) !important;
    color: #7cc1ff !important;
    border: 1px solid rgba(77,159,255,0.32) !important;
    border-radius: 5px !important;
    font-family: 'Share Tech Mono', monospace !important;
    padding: 2px 7px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

.detail-box {
    background: linear-gradient(145deg, #060e20, #091525);
    border: 1px solid #0d2a50;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.detail-title { font-size: 1.3rem; font-weight: 700; color: #e8f2ff; margin-bottom: 6px; font-family: 'Space Grotesk', sans-serif; }
.detail-desc { color: #b6d0ec; line-height: 1.8; margin: 12px 0; }
.detail-key { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #8fb6dc; text-transform: uppercase; letter-spacing: 0.12em; }
.detail-val { font-size: 0.95rem; color: #d4e6ff; font-weight: 500; margin-top: 3px; }

/* ─── Native Streamlit widgets ──────────────────────────────────── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #081428 !important;
    border-color: #1a3a6a !important;
    border-radius: 8px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: #2a5a9a !important;
}
div[data-baseweb="select"] [aria-expanded="true"] {
    border-color: #4d9fff !important;
    box-shadow: 0 0 14px rgba(77,159,255,0.22) !important;
}
div[data-baseweb="select"] [data-baseweb="tag"] {
    background: rgba(77,159,255,0.16) !important;
    border: 1px solid rgba(77,159,255,0.4) !important;
    color: #c8e2ff !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
div[data-baseweb="popover"] ul {
    background: #0c1830 !important;
    border: 1px solid #1a3a6a !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"] li {
    color: #d4e6ff !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.15s ease !important;
}
div[data-baseweb="popover"] li:hover {
    background: rgba(77,159,255,0.14) !important;
    color: #f0f6ff !important;
}

/* Radio + checkbox */
label[data-baseweb="radio"], label[data-baseweb="checkbox"] {
    background: transparent !important;
}
label[data-baseweb="radio"] > div:first-child,
label[data-baseweb="checkbox"] > div:first-child {
    border-color: #2a5a9a !important;
    background: #081428 !important;
}
label[data-baseweb="radio"][data-checked="true"] > div:first-child,
label[data-baseweb="checkbox"][data-checked="true"] > div:first-child {
    background: #4d9fff !important;
    border-color: #4d9fff !important;
}

/* Toggle (st.toggle) */
[role="switch"] {
    background: #1a3a6a !important;
}
[role="switch"][aria-checked="true"] {
    background: #4d9fff !important;
}

/* Expander */
details[data-testid="stExpander"], div[data-testid="stExpander"] {
    background: linear-gradient(160deg, #0a1428 0%, #0c1830 100%) !important;
    border: 1px solid #1a3a6a !important;
    border-radius: 12px !important;
}
details[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary {
    color: #d4e6ff !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
}

/* Alerts (success / warning / error / info) — softer SaaS feel */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-width: 1px !important;
    border-style: solid !important;
    backdrop-filter: blur(8px) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    padding: 12px 16px !important;
}
div[data-baseweb="notification"][kind="info"],
div[role="alert"][data-baseweb="notification"][kind="info"] {
    background: rgba(77,159,255,0.10) !important;
    border-color: rgba(77,159,255,0.4) !important;
    color: #c8e2ff !important;
}
div[data-testid="stAlertContentSuccess"],
div[data-baseweb="notification"][kind="positive"] {
    background: rgba(61,214,140,0.10) !important;
    border-color: rgba(61,214,140,0.4) !important;
    color: #b8f5d3 !important;
}
div[data-testid="stAlertContentWarning"],
div[data-baseweb="notification"][kind="warning"] {
    background: rgba(250,204,21,0.10) !important;
    border-color: rgba(250,204,21,0.4) !important;
    color: #fce97a !important;
}
div[data-testid="stAlertContentError"],
div[data-baseweb="notification"][kind="negative"] {
    background: rgba(248,113,113,0.10) !important;
    border-color: rgba(248,113,113,0.4) !important;
    color: #fdb7b7 !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #4d9fff 0%, #7cc1ff 100%) !important;
    border-radius: 100px !important;
}
div[data-testid="stProgress"] > div > div {
    background: #1a3a6a !important;
    border-radius: 100px !important;
}

/* Download button — should feel secondary */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(145deg, #0e1f3a, #122849) !important;
    border: 1px solid #2a5a9a !important;
    border-radius: 10px !important;
    color: #d4e6ff !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    border-color: #4d9fff !important;
    box-shadow: 0 0 20px rgba(77,159,255,0.18) !important;
    color: #f0f6ff !important;
    transform: translateY(-1px) !important;
}

/* Sidebar polish */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #040a14 0%, #06101e 100%) !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 0.95rem !important;
    color: #d4e6ff !important;
    letter-spacing: 0.02em !important;
    margin-top: 14px !important;
}

/* Plotly chart container */
div[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #1a3a6a !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.28) !important;
}

/* Tooltip / hover hints on tabs already styled; just bump spacing */
div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    with open('companies.json', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

df = load_data()

# ── Branded topbar ──────────────────────────────────────────────────
st.markdown(
    f'''
    <div class="topbar">
      <div class="brand">
        <div class="header-title">🚀 Just <span>Sell</span> It</div>
        <div class="header-sub">AI Sales & CRM Agent · Saudi Market · {len(df)} entities indexed</div>
      </div>
      <div class="topbar-status"><span class="dot"></span><span>System Online</span></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Sidebar Filters ─────────────────────────────────────────────────
with st.sidebar:
    # ── Gmail sign-in ───────────────────────────────────────────────
    st.markdown("## 📬 Gmail Account")

    if "gmail_email" not in st.session_state:
        st.session_state.gmail_email = get_authenticated_email()

    current_email = st.session_state.gmail_email

    if current_email:
        st.success(f"✓ Connected\n\n`{current_email}`")
        if st.button("🚪 Sign out", key="gmail_signout", use_container_width=True):
            sign_out_gmail()
            st.session_state.gmail_email = None
            st.rerun()
    else:
        st.warning("Not connected — sending and inbox features are disabled.")
        if st.button("🔐 Sign in to Gmail", key="gmail_signin",
                     type="primary", use_container_width=True):
            try:
                with st.spinner("Opening browser for Gmail consent…"):
                    result = sign_in_gmail()
                st.session_state.gmail_email = result.get("email")
                st.success(f"Connected as {result.get('email')}")
                st.rerun()
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.caption("Put your OAuth `credentials.json` (Google Cloud → "
                           "OAuth client ID → Desktop app) in the project root.")
            except Exception as exc:
                st.error(f"Sign-in failed: {exc}")

    st.markdown("---")

    st.markdown("## 🔍 Filter Companies")

    st.markdown('<div class="filter-header">📂 Sector — pick one or more</div>', unsafe_allow_html=True)
    all_sectors = sorted(df['sector'].dropna().unique())
    selected_sectors = st.multiselect("Sector", all_sectors, default=[], key="sectors", label_visibility="collapsed", placeholder="All sectors (default)")

    st.markdown('<div class="filter-header">📍 City — pick one or more</div>', unsafe_allow_html=True)
    all_cities = sorted([c for c in df['city_clean'].dropna().unique() if c not in ('Unknown', '')])
    selected_cities = st.multiselect("City", all_cities, key="cities", label_visibility="collapsed", placeholder="All cities (default)")

    st.markdown('<div class="filter-header">👥 Company Size — pick one or more</div>', unsafe_allow_html=True)
    emp_order = ['51-200', '200-1,000', '1,000-5,000', '5,000-10,000', '10,000+']
    selected_emp = st.multiselect("Company Size", emp_order, default=[], key="emp", label_visibility="collapsed", placeholder="All sizes (default)")

    st.markdown('<div class="filter-header">🚀 Company Type</div>', unsafe_allow_html=True)
    company_type = st.radio("Company Type", ['All', 'Startups Only', 'Established Only'], key="type", label_visibility="collapsed")

    st.markdown('<div class="filter-header">🔎 Search</div>', unsafe_allow_html=True)
    search = st.text_input("Search", placeholder="Company ", key="search", label_visibility="collapsed")

# ── Apply Filters ────────────────────────────────────────────────────
filtered = df.copy()

if selected_sectors:  # empty = show all
    filtered = filtered[filtered['sector'].isin(selected_sectors)]

if selected_cities:  # empty = show all
    filtered = filtered[filtered['city_clean'].isin(selected_cities)]

if selected_emp:  # empty = show all
    filtered = filtered[filtered['emp_bucket'].isin(selected_emp)]

if company_type == 'Startups Only':
    filtered = filtered[filtered['is_startup'] == True]
elif company_type == 'Established Only':
    filtered = filtered[filtered['is_startup'] == False]

if search:
    mask = (
        filtered['name'].str.contains(search, case=False, na=False) |
        filtered['description'].str.contains(search, case=False, na=False) |
        filtered['sub_sector'].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

filtered = filtered.sort_values('name').reset_index(drop=True)

# ── Metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, len(filtered), "Companies Found"),
    (c2, filtered['sector'].nunique(), "Sectors"),
    (c3, int(filtered['is_startup'].sum()), "🚀 Startups"),
    (c4, filtered[filtered['city_clean'] != 'Unknown']['city_clean'].nunique(), "Cities"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{num}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── State ────────────────────────────────────────────────────────────
if 'selected' not in st.session_state:
    st.session_state.selected = None
if 'score_results' not in st.session_state:
    st.session_state.score_results = None
if 'score_criteria' not in st.session_state:
    st.session_state.score_criteria = ""
if 'scored_count' not in st.session_state:
    st.session_state.scored_count = 0
if 'campaign_emails' not in st.session_state:
    st.session_state.campaign_emails = None
if 'opportunity_results' not in st.session_state:
    st.session_state.opportunity_results = {}
if 'proposal_pdf_path' not in st.session_state:
    st.session_state.proposal_pdf_path = None
if 'proposal_text' not in st.session_state:
    st.session_state.proposal_text = None
if 'sent_emails' not in st.session_state:
    st.session_state.sent_emails = {}        # {company_name: "sent" | "error msg"}
if 'found_emails' not in st.session_state:
    st.session_state.found_emails = {}       # {company_name: email_address}
if 'email_inbox' not in st.session_state:
    st.session_state.email_inbox = None
if 'email_ai_results' not in st.session_state:
    st.session_state.email_ai_results = {}   # {email_id: AI result dict}
if 'proposal_workflow' not in st.session_state:
    st.session_state.proposal_workflow = None  # {thread_id, preview, draft}
if 'send_all_confirm' not in st.session_state:
    st.session_state.send_all_confirm = False
if 'all_proposals' not in st.session_state:
    st.session_state.all_proposals = {}      # {company_name: {text, pdf_path}}
if 'auto_poll_enabled' not in st.session_state:
    st.session_state.auto_poll_enabled = False
if 'auto_poll_hours' not in st.session_state:
    st.session_state.auto_poll_hours = 0
if 'auto_poll_minutes' not in st.session_state:
    st.session_state.auto_poll_minutes = 10
if 'last_polled_at' not in st.session_state:
    st.session_state.last_polled_at = ""

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Companies List", "📊 Analytics", "🎯 Lead Scoring",
    "📧 Campaign", "📨 Email Agent", "🤝 Opportunity", "📄 Proposal"
])

with tab1:
    if filtered.empty:
        st.markdown('<div class="no-results">😕 No companies match your current filters.</div>', unsafe_allow_html=True)
    else:
        # Detail panel
        if st.session_state.selected is not None:
            idx = st.session_state.selected
            if idx < len(filtered):
                row = filtered.iloc[idx]
                website = str(row.get('website', '') or '').strip()
                website_clean = website.replace('www.', '').strip()

                with st.container(border=True):
                    title_col, close_col = st.columns([5, 1])
                    with title_col:
                        st.markdown(f"### 🏢 {row.get('name', '')}")
                    with close_col:
                        if st.button("✕ Close", key="close"):
                            st.session_state.selected = None
                            st.rerun()
                    badges = f"`{row.get('sector', '')}`"
                    if row.get('is_startup'):
                        badges += "  `🚀 Startup`"
                    badges += f"  `👥 {row.get('employees', 'N/A')}`"
                    st.markdown(badges)
                    st.markdown(f"> {row.get('description', 'No description available.')}")
                    d1, d2, d3, d4 = st.columns(4)
                    with d1:
                        st.markdown("**📍 City**")
                        st.write(row.get('city_clean', 'N/A'))
                    with d2:
                        st.markdown("**🏷️ Sub-sector**")
                        st.write(row.get('sub_sector', 'N/A'))
                    with d3:
                        st.markdown("**👥 Employees**")
                        st.write(row.get('employees', 'N/A'))
                    with d4:
                        st.markdown("**🔗 Website**")
                        if website_clean and website_clean not in ('N/A', 'nan', ''):
                            st.markdown(f"[{website_clean}](https://{website_clean})")
                        else:
                            st.write("N/A")

                st.markdown("---")

        # Grid
        cols_per_row = 2
        rows = [filtered.iloc[i:i+cols_per_row] for i in range(0, len(filtered), cols_per_row)]

        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col_idx, (_, company) in enumerate(row_df.iterrows()):
                abs_idx = filtered.index.get_loc(company.name)
                with cols[col_idx]:
                    startup_tag = "🚀 " if company.get('is_startup') else ""
                    sub = str(company.get('sub_sector', '') or '')[:45]
                    btn_label = f"{startup_tag}**{company['name']}**\n📍 {company.get('city_clean','?')}  ·  👥 {company.get('emp_bucket','?')}\n🏷️ {sub}"
                    if st.button(btn_label, key=f"btn_{abs_idx}", use_container_width=True):
                        st.session_state.selected = abs_idx
                        st.rerun()

with tab2:
    if filtered.empty:
        st.warning("No data to display.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            sector_counts = filtered['sector'].value_counts().reset_index()
            sector_counts.columns = ['Sector', 'Count']
            fig1 = px.bar(sector_counts, x='Count', y='Sector', orientation='h',
                         title='Companies by Sector', color='Count',
                         color_continuous_scale=['#1a3a2a', '#00c853'],
                         template='plotly_dark')
            fig1.update_layout(showlegend=False, height=380, plot_bgcolor='#1a1d27', paper_bgcolor='#1a1d27')
            st.plotly_chart(fig1, width='stretch')

        with col_b:
            city_df = filtered[filtered['city_clean'].notna() & (filtered['city_clean'] != 'Unknown')]
            if not city_df.empty:
                city_counts = city_df['city_clean'].value_counts().reset_index()
                city_counts.columns = ['City', 'Count']
                fig2 = px.pie(city_counts, values='Count', names='City',
                             title='Distribution by City', hole=0.45,
                             template='plotly_dark',
                             color_discrete_sequence=px.colors.qualitative.Safe)
                fig2.update_layout(height=380, paper_bgcolor='#1a1d27')
                st.plotly_chart(fig2, width='stretch')

        emp_counts = filtered[
            filtered['emp_bucket'].notna() & (filtered['emp_bucket'] != 'Unknown')
        ]['emp_bucket'].value_counts().reset_index()
        emp_counts.columns = ['Size', 'Count']
        emp_order_f = [e for e in emp_order if e in emp_counts['Size'].values]
        emp_counts['Size'] = pd.Categorical(emp_counts['Size'], categories=emp_order_f, ordered=True)
        emp_counts = emp_counts.sort_values('Size')
        fig3 = px.bar(emp_counts, x='Size', y='Count', title='Companies by Employee Count',
                     color='Count', color_continuous_scale=['#1e3a5f', '#60a5fa'],
                     template='plotly_dark')
        fig3.update_layout(showlegend=False, plot_bgcolor='#1a1d27', paper_bgcolor='#1a1d27')
        st.plotly_chart(fig3, width='stretch')


with tab3:
    st.markdown("### 🎯 Lead Scoring")
    st.markdown(f"Scoring will run on the **{len(filtered)} companies** from your current filters.")

    if filtered.empty:
        st.warning("No companies to score. Adjust your filters first.")
    else:
        st.markdown("---")

        # Criteria selection
        use_defaults = st.checkbox(
            "✅ Use BeamData default criteria (IT, Fintech, Telecom, Healthcare — 200+ employees — Riyadh priority)",
            value=True
        )

        custom_criteria = ""
        if not use_defaults:
            custom_criteria = st.text_area(
                "✍️ Write your own criteria:",
                placeholder="e.g. I want companies in healthcare with 500+ employees that are likely to invest in AI automation...",
                height=120
            )

        st.markdown("---")

        # Agent mode toggle
        use_agent = st.toggle(
            "🤖 Agent Mode — searches the web for each company (slower but more accurate)",
            value=False
        )
        if use_agent:
            st.info(f"⚠️ Agent mode will do {len(filtered)} web searches — recommended max 10 companies.")

        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            score_btn = st.button(
                f"🎯 Score {len(filtered)} Companies",
                use_container_width=True,
                type="primary"
            )
        with col_info:
            if use_agent:
                st.caption("🤖 Agent will search the web for each company then score it.")
            else:
                st.caption("⚡ Fast mode: scores based on existing data.")

        if score_btn:
            if not use_defaults and not custom_criteria.strip():
                st.error("Please write your criteria or use BeamData defaults.")
            else:
                companies_list = filtered.to_dict(orient='records')

                if use_agent:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(current, total, company_name):
                        progress_bar.progress(current / total)
                        status_text.text(f"🔍 Researching {current}/{total}: {company_name}")

                    try:
                        results = score_in_batches(
                            companies_list,
                            criteria=custom_criteria,
                            use_beamdata_defaults=use_defaults,
                            use_agent=True,
                            progress_callback=update_progress
                        )
                        progress_bar.progress(1.0)
                        status_text.text("✅ Done!")
                        st.session_state.score_results = results
                        st.session_state.scored_count = len(results)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    with st.spinner(f"⚡ Scoring {len(companies_list)} companies..."):
                        try:
                            results = score_in_batches(
                                companies_list,
                                criteria=custom_criteria,
                                use_beamdata_defaults=use_defaults,
                                batch_size=15,
                                use_agent=False
                            )
                            st.session_state.score_results = results
                            st.session_state.scored_count = len(results)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        # Show results
        if st.session_state.score_results:
            results = st.session_state.score_results

            # Summary metrics
            high = sum(1 for r in results if r.get('grade') == 'High')
            medium = sum(1 for r in results if r.get('grade') == 'Medium')
            low = sum(1 for r in results if r.get('grade') == 'Low')

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-num">{len(results)}</div><div class="metric-lbl">Scored</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4ade80">{high}</div><div class="metric-lbl">🟢 High</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#facc15">{medium}</div><div class="metric-lbl">🟡 Medium</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#f87171">{low}</div><div class="metric-lbl">🔴 Low</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Grade filter
            grade_filter = st.radio("Show:", ["All", "🟢 High", "🟡 Medium", "🔴 Low"], horizontal=True)

            filtered_results = results
            if grade_filter == "🟢 High":
                filtered_results = [r for r in results if r.get('grade') == 'High']
            elif grade_filter == "🟡 Medium":
                filtered_results = [r for r in results if r.get('grade') == 'Medium']
            elif grade_filter == "🔴 Low":
                filtered_results = [r for r in results if r.get('grade') == 'Low']

            st.markdown("---")

            # Results table
            for i, r in enumerate(filtered_results):
                grade = r.get('grade', '')
                score = r.get('score', 0)
                emoji = "🟢" if grade == "High" else "🟡" if grade == "Medium" else "🔴"

                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{i+1}. {r.get('name', 'N/A')}**")
                        st.caption(f"📍 {r.get('city_clean', 'N/A')}  ·  🏷️ {r.get('sector', 'N/A')}  ·  👥 {r.get('employees', 'N/A')}")
                    with c2:
                        st.markdown(f"### {emoji} {score}")
                    with c3:
                        st.markdown(f"`{grade}`")
                    st.markdown(f"💬 *{r.get('reason', '')}*")
                    if r.get('research'):
                        with st.expander("🔍 Web Research"):
                            st.caption(r.get('research', ''))

            st.markdown("---")

            # Export CSV
            import csv, io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['name', 'city_clean', 'sector', 'employees', 'score', 'grade', 'reason', 'website'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'name': r.get('name', ''),
                    'city_clean': r.get('city_clean', ''),
                    'sector': r.get('sector', ''),
                    'employees': r.get('employees', ''),
                    'score': r.get('score', ''),
                    'grade': r.get('grade', ''),
                    'reason': r.get('reason', ''),
                    'website': r.get('website', ''),
                })
            csv_data = output.getvalue()

            st.download_button(
                label="⬇️ Export Results as CSV",
                data=csv_data,
                file_name="lead_scoring_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            if st.button("🔄 Clear Results & Re-score", use_container_width=True):
                st.session_state.score_results = None
                st.rerun()

with tab4:
    st.markdown("### 📧 Campaign — Generate Cold Emails")

    if not st.session_state.score_results:
        st.warning("⚠️ Run Lead Scoring first (in the 🎯 Lead Scoring tab), then come back here.")
    else:
        st.markdown("Pick the companies you want to send a personalized email to:")

        results = st.session_state.score_results

        # ── Bulk select controls ────────────────────────────────────
        col_sa, col_ds, _ = st.columns([1, 1, 3])
        with col_sa:
            if st.button("✅ Select All", key="campaign_select_all", use_container_width=True):
                for i in range(len(results)):
                    st.session_state[f"campaign_pick_{i}"] = True
                st.rerun()
        with col_ds:
            if st.button("🔲 Deselect All", key="campaign_deselect_all", use_container_width=True):
                for i in range(len(results)):
                    st.session_state[f"campaign_pick_{i}"] = False
                st.rerun()

        # ── Checkbox picker ─────────────────────────────────────────
        selected_names = []
        for i, r in enumerate(results):
            grade = r.get('grade', '')
            emoji = "🟢" if grade == "High" else "🟡" if grade == "Medium" else "🔴"
            label = f"{emoji} {r.get('name', 'N/A')} — {r.get('sector', 'N/A')} ({r.get('score', 0)})"
            checked = st.checkbox(label, key=f"campaign_pick_{i}")
            if checked:
                selected_names.append(i)

        st.markdown("---")
        st.caption(f"✅ Selected: {len(selected_names)} companies")

        generate_btn = st.button(
            f"📧 Generate Emails for {len(selected_names)} Selected Companies",
            type="primary",
            disabled=(len(selected_names) == 0)
        )

        if generate_btn:
            companies_to_email = [results[i] for i in selected_names]

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_campaign_progress(current, total, name):
                progress_bar.progress(current / total)
                status_text.text(f"✍️ Writing email {current}/{total}: {name}")

            try:
                agent = CampaignAgent()
                emails = agent.run_on_companies(
                    companies_to_email,
                    progress_callback=update_campaign_progress
                )
                progress_bar.progress(1.0)
                status_text.text("✅ Done!")
                st.session_state.campaign_emails = emails
                st.rerun()
            except Exception as e:
                st.error(f"Error generating emails: {e}")

        # ── Show generated emails ───────────────────────────────────
        if st.session_state.campaign_emails:
            emails = st.session_state.campaign_emails
            st.markdown("---")
            st.markdown(f"#### ✉️ {len(emails)} Emails Generated")

            # Gmail status banner
            if is_gmail_configured():
                st.success(
                    f"✅ Signed in as **{st.session_state.get('gmail_email', '')}** — "
                    "you can send emails directly from here."
                )
            else:
                st.warning(
                    "🔐 Gmail not connected. Use **Sign in to Gmail** in the left "
                    "sidebar to enable sending."
                )

            # ── Send All (batch HITL) ───────────────────────────────
            if is_gmail_configured() and len(emails) > 0:
                if not st.session_state.send_all_confirm:
                    if st.button(f"📤 Send All {len(emails)} Emails",
                                 key="send_all_btn", use_container_width=True):
                        st.session_state.send_all_confirm = True
                        st.rerun()
                else:
                    valid_sends, skipped = [], []
                    for i, e in enumerate(emails):
                        cn = e.get('company_name', 'N/A')
                        if st.session_state.sent_emails.get(cn) == "sent":
                            continue
                        rec = (st.session_state.get(f"recipient_{i}", "")
                               or e.get('email', '')
                               or st.session_state.found_emails.get(cn, ''))
                        if rec:
                            valid_sends.append((i, cn, rec, e))
                        else:
                            skipped.append(cn)

                    st.warning(
                        f"⚠️ About to send **{len(valid_sends)}** emails. "
                        f"{len(skipped)} skipped (no recipient)."
                    )
                    y_col, n_col = st.columns(2)
                    with y_col:
                        if st.button("✅ Yes, Send All", key="send_all_yes",
                                     type="primary", use_container_width=True):
                            sent, failed = 0, 0
                            prog = st.progress(0)
                            stat = st.empty()
                            for k, (i, cn, rec, e) in enumerate(valid_sends):
                                stat.text(f"📤 Sending {k+1}/{len(valid_sends)}: {cn}")
                                try:
                                    gmail_send(
                                        to_address=rec,
                                        subject=e.get('email_subject', 'BeamData Introduction'),
                                        body=e.get('email_body', ''),
                                    )
                                    st.session_state.sent_emails[cn] = "sent"
                                    sent += 1
                                except Exception as ex:
                                    st.session_state.sent_emails[cn] = str(ex)
                                    failed += 1
                                prog.progress((k + 1) / max(1, len(valid_sends)))
                            stat.empty()
                            prog.empty()
                            st.session_state.send_all_confirm = False
                            st.success(f"✅ Sent {sent}. Failed {failed}.")
                            st.rerun()
                    with n_col:
                        if st.button("❌ Cancel", key="send_all_no",
                                     use_container_width=True):
                            st.session_state.send_all_confirm = False
                            st.rerun()

            for i, e in enumerate(emails):
                company_name = e.get('company_name', 'N/A')
                with st.container(border=True):
                    st.markdown(f"**{i+1}. {company_name}**")
                    st.caption(f"📍 {e.get('city', 'N/A')}  ·  🏷️ {e.get('sector', 'N/A')}")
                    st.markdown(f"**Subject:** {e.get('email_subject', '')}")
                    st.text_area(
                        "Body",
                        value=e.get('email_body', ''),
                        height=160,
                        key=f"email_body_{i}",
                        label_visibility="collapsed"
                    )
                    st.caption(f"🎯 Goal: {e.get('campaign_goal', '')}  ·  💡 Suggested: {e.get('suggested_service', '')}")
                    if e.get("matched_projects"):
                        st.caption(f"📁 Matched Past Projects: {e.get('matched_projects', '')}")

                    # ── Email sending row ──────────────────────────
                    st.markdown("---")
                    existing_email = (
                        e.get('email', '')
                        or st.session_state.found_emails.get(company_name, '')
                    )
                    send_col1, send_col2, send_col3 = st.columns([3, 1, 1])
                    with send_col1:
                        recipient = st.text_input(
                            "📧 Recipient email",
                            value=existing_email,
                            key=f"recipient_{i}",
                            placeholder="Enter or auto-find email address",
                            label_visibility="collapsed",
                        )
                    with send_col2:
                        if st.button("🔍 Auto-find", key=f"find_{i}", use_container_width=True):
                            with st.spinner("Searching..."):
                                try:
                                    found = find_email_for_company(
                                        company_name,
                                        website=e.get('website', ''),
                                        sector=e.get('sector', ''),
                                    )
                                    if found:
                                        st.session_state.found_emails[company_name] = found
                                        st.rerun()
                                    else:
                                        st.warning("No email found.")
                                except Exception as ex:
                                    st.error(str(ex))
                    with send_col3:
                        sent_status = st.session_state.sent_emails.get(company_name)
                        if sent_status == "sent":
                            st.success("✅ Sent!")
                        else:
                            send_disabled = not (is_gmail_configured() and recipient)
                            if st.button("📤 Send", key=f"send_{i}", use_container_width=True, disabled=send_disabled):
                                try:
                                    gmail_send(
                                        to_address=recipient,
                                        subject=e.get('email_subject', 'BeamData Introduction'),
                                        body=e.get('email_body', ''),
                                    )
                                    st.session_state.sent_emails[company_name] = "sent"
                                except Exception as ex:
                                    st.session_state.sent_emails[company_name] = str(ex)
                                st.rerun()

            st.markdown("---")

            # Export CSV
            import csv as csv_module, io as io_module
            output = io_module.StringIO()
            fieldnames = [
                "company_name", "arabic_name", "sector", "sub_sector", "city",
                "country", "website", "email", "phone", "linkedin_url",
                "employees", "founded_year", "description", "tags",
                "email_subject", "email_body", "campaign_goal", "suggested_service",
                "matched_projects",
            ]
            writer = csv_module.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for e in emails:
                writer.writerow(e)
            csv_data = output.getvalue()

            st.download_button(
                label="⬇️ Export Emails as CSV",
                data=csv_data,
                file_name="campaign_emails.csv",
                mime="text/csv",
                use_container_width=True
            )

            if st.button("🔄 Clear Emails", use_container_width=True):
                st.session_state.campaign_emails = None
                st.rerun()

with tab5:
    st.markdown("### 📨 Email Agent — Inbox & Replies")
    st.markdown(
        "Read incoming replies to your campaign, let the AI classify and draft responses, "
        "then approve and send with one click."
    )

    if not is_gmail_configured():
        st.warning(
            "🔐 Gmail not connected. Use **Sign in to Gmail** in the left sidebar to enable inbox and replies."
        )
    else:
        st.success(f"✅ Connected as: **{st.session_state.get('gmail_email', '')}**")

        # ── Auto-poll controls ──────────────────────────────────────
        with st.expander("⚙️ Auto-poll inbox (background fetch)", expanded=False):
            st.caption("While this tab is open, the inbox can be refreshed automatically. New emails are also auto-classified.")
            ap_enable_col, ap_h_col, ap_m_col = st.columns([2, 1, 1])
            with ap_enable_col:
                st.session_state.auto_poll_enabled = st.checkbox(
                    "Enable auto-poll", value=st.session_state.auto_poll_enabled,
                    key="auto_poll_enabled_chk",
                )
            with ap_h_col:
                st.session_state.auto_poll_hours = st.number_input(
                    "Hours", min_value=0, max_value=24,
                    value=int(st.session_state.auto_poll_hours), step=1,
                    key="auto_poll_hours_in",
                )
            with ap_m_col:
                st.session_state.auto_poll_minutes = st.number_input(
                    "Minutes", min_value=0, max_value=59,
                    value=int(st.session_state.auto_poll_minutes), step=1,
                    key="auto_poll_minutes_in",
                )
            poll_seconds = (st.session_state.auto_poll_hours * 3600
                            + st.session_state.auto_poll_minutes * 60)
            if st.session_state.auto_poll_enabled and poll_seconds < 60:
                st.warning("Minimum interval is 1 minute — raising to 60 seconds.")
                poll_seconds = 60
            if st.session_state.auto_poll_enabled:
                st.caption(
                    f"⏱️ Polling every {st.session_state.auto_poll_hours}h "
                    f"{st.session_state.auto_poll_minutes}m "
                    f"({poll_seconds}s)."
                    + (f" Last poll: **{st.session_state.last_polled_at}**."
                       if st.session_state.last_polled_at else "")
                )

        # ── Auto-poll fragment (runs on schedule) ───────────────────
        if st.session_state.auto_poll_enabled and poll_seconds >= 60:
            @st.fragment(run_every=poll_seconds)
            def _auto_poll_fragment():
                from datetime import datetime
                if not is_gmail_configured():
                    return
                try:
                    inbox = fetch_unread_emails(limit=15)
                    existing_ids = {em["id"] for em in (st.session_state.email_inbox or [])}
                    new_ids = {em["id"] for em in inbox} - existing_ids

                    st.session_state.email_inbox = inbox
                    new_count = 0
                    for em in inbox:
                        if em["id"] not in st.session_state.email_ai_results:
                            try:
                                st.session_state.email_ai_results[em["id"]] = classify_and_draft(em)
                                new_count += 1
                            except Exception:
                                pass
                    st.session_state.last_polled_at = datetime.now().strftime("%H:%M:%S")

                    if new_ids or new_count > 0:
                        # New mail arrived — rerun the whole app so the cards below refresh.
                        st.rerun(scope="app")
                    st.caption(
                        f"🔄 Auto-polled at {st.session_state.last_polled_at} "
                        f"— {len(inbox)} unread, no new since last poll."
                    )
                except Exception as ex:
                    st.caption(f"⚠️ Auto-poll failed: {ex}")

            _auto_poll_fragment()

        st.markdown("---")

        col_fetch, col_clear = st.columns([2, 1])
        with col_fetch:
            if st.button("📥 Fetch Unread Emails", type="primary", use_container_width=True):
                with st.spinner("Connecting to Gmail inbox..."):
                    try:
                        st.session_state.email_inbox = fetch_unread_emails(limit=15)
                        st.session_state.email_ai_results = {}
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Could not connect to Gmail: {ex}")
        with col_clear:
            if st.button("🔄 Clear Inbox", use_container_width=True):
                st.session_state.email_inbox = None
                st.session_state.email_ai_results = {}
                st.rerun()

        if st.session_state.email_inbox is not None:
            inbox = st.session_state.email_inbox
            if not inbox:
                st.info("📭 No unread emails in your inbox.")
            else:
                st.markdown(f"#### 📬 {len(inbox)} Unread Email{'s' if len(inbox) > 1 else ''}")
                st.markdown("---")

                for em in inbox:
                    eid = em["id"]
                    priority_color = {"High": "#f87171", "Medium": "#facc15", "Low": "#4ade80"}.get(
                        st.session_state.email_ai_results.get(eid, {}).get("priority", ""), "#888"
                    )
                    with st.container(border=True):
                        hdr_col, btn_col = st.columns([4, 1])
                        with hdr_col:
                            st.markdown(f"**From:** {em['from']}")
                            st.markdown(f"**Subject:** {em['subject']}")
                            st.caption(f"📅 {em['date']}")
                        with btn_col:
                            if st.button("🤖 Analyze", key=f"analyze_{eid}", use_container_width=True):
                                with st.spinner("AI is analyzing..."):
                                    result = classify_and_draft(em)
                                    st.session_state.email_ai_results[eid] = result
                                    st.rerun()

                        with st.expander("📄 Email Body"):
                            st.text(em["body"][:800])

                        ai = st.session_state.email_ai_results.get(eid)
                        if ai:
                            st.markdown("---")
                            p_col, c_col, s_col = st.columns(3)
                            with p_col:
                                st.markdown(f"**🏷️ Classification:** `{ai.get('classification', '')}`")
                            with c_col:
                                priority = ai.get('priority', 'Medium')
                                p_emoji = "🔴" if priority == "High" else "🟡" if priority == "Medium" else "🟢"
                                st.markdown(f"**{p_emoji} Priority:** `{priority}`")
                            with s_col:
                                st.markdown(f"**💡 Signals:** {ai.get('signals', '')}")

                            st.markdown("**✏️ AI Draft Reply** (edit before sending):")
                            draft_body = st.text_area(
                                "Draft reply",
                                value=ai.get("draft_body", ""),
                                height=140,
                                key=f"draft_{eid}",
                                label_visibility="collapsed",
                            )
                            draft_subject = ai.get("draft_subject", f"Re: {em['subject']}")

                            # Extract reply-to address
                            reply_to = em["from"]
                            import re as _re
                            found_addr = _re.findall(r"<(.+?)>", reply_to)
                            reply_addr = found_addr[0] if found_addr else reply_to.strip()

                            send_reply_col, _ = st.columns([1, 3])
                            with send_reply_col:
                                if st.button("📤 Send Reply", key=f"send_reply_{eid}", type="primary", use_container_width=True):
                                    try:
                                        gmail_send(reply_addr, draft_subject, draft_body)
                                        st.success(f"✅ Reply sent to {reply_addr}")
                                    except Exception as ex:
                                        st.error(f"Send failed: {ex}")
                                        st.rerun()

                            # ── Routing: send the lead downstream based on classification ──
                            ROUTABLE = {"Interested", "Pricing Request", "Meeting Request"}
                            classification = ai.get("classification", "")
                            campaign_list = st.session_state.campaign_emails or []
                            if classification in ROUTABLE and campaign_list:
                                st.markdown("---")
                                st.markdown(f"**🔀 Route this lead** — classification *{classification}* suggests sending it downstream.")

                                # Auto-match sender to a known campaign company
                                sender_addr = reply_addr.lower()
                                sender_domain = sender_addr.split("@")[-1] if "@" in sender_addr else ""
                                matched_company = None
                                for ce in campaign_list:
                                    cn = ce.get("company_name", "")
                                    if (ce.get("email", "").lower() == sender_addr
                                            or st.session_state.found_emails.get(cn, "").lower() == sender_addr):
                                        matched_company = cn
                                        break
                                    if sender_domain and sender_domain in ce.get("website", "").lower():
                                        matched_company = cn
                                        break

                                company_options = [ce.get("company_name", "") for ce in campaign_list]
                                default_idx = (company_options.index(matched_company)
                                               if matched_company in company_options else 0)
                                if matched_company:
                                    st.caption(f"🔗 Auto-matched sender to **{matched_company}** by email/domain.")
                                selected_company = st.selectbox(
                                    "Company this lead belongs to:",
                                    company_options,
                                    index=default_idx,
                                    key=f"route_company_{eid}",
                                )
                                selected_ce = next(
                                    (ce for ce in campaign_list if ce.get("company_name") == selected_company),
                                    {},
                                )

                                qual_col, prop_col = st.columns(2)
                                with qual_col:
                                    if st.button("🤝 Qualify Opportunity",
                                                 key=f"route_qual_{eid}", use_container_width=True):
                                        with st.spinner(f"Qualifying {selected_company}…"):
                                            try:
                                                ctx = {
                                                    "company_name": selected_company,
                                                    "sector": selected_ce.get("sector", ""),
                                                    "sub_sector": selected_ce.get("sub_sector", ""),
                                                    "employees": selected_ce.get("employees", ""),
                                                    "city": selected_ce.get("city", ""),
                                                    "description": selected_ce.get("description", ""),
                                                    "tags": selected_ce.get("tags", ""),
                                                    "founded_year": selected_ce.get("founded_year", ""),
                                                    "email_subject": selected_ce.get("email_subject", ""),
                                                }
                                                r = evaluate_opportunity_auto(ctx, em.get("body", ""))
                                                r["company_name"] = selected_company
                                                r["email_reply"] = em.get("body", "")[:500]
                                                r["assessment_mode"] = "🎯 Auto Assessment (BANT)"
                                                st.session_state.opportunity_results[selected_company] = r
                                                st.rerun()
                                            except Exception as ex:
                                                st.error(f"Qualification failed: {ex}")

                                # If qualified, surface a one-click proposal action
                                opp = st.session_state.opportunity_results.get(selected_company)
                                if opp:
                                    status = opp.get("status", "")
                                    emoji = "🟢" if status == "Qualified" else "🔴" if status == "Not Qualified" else "⚪"
                                    st.info(f"{emoji} **{status}** — {opp.get('opportunity_score', 0)}/100. "
                                            f"{opp.get('reason', '')}")
                                    with prop_col:
                                        prop_disabled = status != "Qualified"
                                        if st.button("📄 Generate Proposal",
                                                     key=f"route_prop_{eid}",
                                                     disabled=prop_disabled,
                                                     use_container_width=True):
                                            try:
                                                with st.spinner("Building knowledge base…"):
                                                    retriever = initialize_rag()
                                                    proposal_executor = build_proposal_agent(retriever)
                                                with st.spinner(f"Drafting proposal for {selected_company}…"):
                                                    start_res = proposal_executor.start({
                                                        "company_name": selected_company,
                                                        "agreed_price": "Not specified — please estimate",
                                                    })
                                                    if start_res["state"] == "awaiting_approval":
                                                        final = proposal_executor.resume(start_res["thread_id"], approved=True)
                                                    else:
                                                        final = start_res
                                                if final.get("state") == "finalized" and final.get("output"):
                                                    text = final["output"]
                                                    pdf_path = f"/tmp/proposal_{selected_company.replace(' ', '_')}.pdf"
                                                    generate_proposal_pdf(selected_company, text, pdf_path)
                                                    st.session_state.all_proposals[selected_company] = {
                                                        "text": text, "pdf_path": pdf_path,
                                                    }
                                                    st.success(f"✅ Proposal ready for {selected_company} — see the 📄 Proposal tab.")
                                                    st.rerun()
                                                else:
                                                    st.error(f"Could not finalize proposal: {final.get('error', 'unknown')}")
                                            except FileNotFoundError as ex:
                                                st.error(str(ex))
                                            except Exception as ex:
                                                st.error(f"Error: {ex}")

with tab6:
    st.markdown("### 🤝 Opportunity Qualification — AI Assessment")
    st.markdown(
        "Select a company from your campaign, paste their reply (or leave blank for profile-only assessment), "
        "and the AI will automatically qualify the opportunity — no manual form filling required."
    )

    if not st.session_state.campaign_emails:
        st.warning("⚠️ Generate campaign emails first (in the 📧 Campaign tab), then come back here.")
    else:
        emails = st.session_state.campaign_emails
        st.markdown("---")

        company_options = [e.get("company_name", f"Company {i}") for i, e in enumerate(emails)]
        picked_name = st.selectbox("Select a company:", company_options, key="opp_company_pick")
        picked_idx = company_options.index(picked_name)
        picked_email = emails[picked_idx]

        # ── Company info card ──────────────────────────────────────
        with st.container(border=True):
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                st.markdown(f"**🏢 {picked_email.get('company_name', '')}**")
                st.caption(f"🏷️ {picked_email.get('sector', '')} · {picked_email.get('sub_sector', '')}")
            with c_b:
                st.caption(f"📍 {picked_email.get('city', '')}")
                st.caption(f"👥 {picked_email.get('employees', 'N/A')} employees")
            with c_c:
                st.caption(f"📧 Subject sent: {picked_email.get('email_subject', '')[:60]}...")

        st.markdown("---")

        # ── Assessment mode selector ───────────────────────────────
        assessment_mode = st.radio(
            "**Choose assessment mode:**",
            ["🎯 Auto Assessment (BANT)", "✍️ Custom Criteria"],
            horizontal=True,
            key="opp_mode"
        )

        email_reply = st.text_area(
            "✉️ Their email reply (paste it here — or leave blank to assess from company profile only)",
            key="opp_reply",
            height=110,
            placeholder="Dear BeamData team, thank you for reaching out. We are currently exploring AI solutions for our operations and would love to learn more..."
        )

        custom_criteria_text = ""
        if assessment_mode == "✍️ Custom Criteria":
            st.markdown("**Define your scoring criteria** — describe what makes an ideal client for BeamData:")
            custom_criteria_text = st.text_area(
                "Custom criteria",
                key="opp_custom_criteria",
                height=130,
                label_visibility="collapsed",
                placeholder=(
                    "Example:\n"
                    "- Company must have 200+ employees (weight: high)\n"
                    "- Sector should be Fintech, Healthcare, or Telecom (weight: high)\n"
                    "- Company should show interest in AI/data transformation (weight: medium)\n"
                    "- Located in Riyadh or Jeddah (weight: low)"
                )
            )

        eval_col_one, eval_col_all = st.columns(2)
        with eval_col_one:
            evaluate_btn = st.button("🤖 Evaluate This Opportunity", type="primary", use_container_width=True)
        with eval_col_all:
            evaluate_all_btn = st.button(f"🤖 Evaluate All {len(emails)} Companies", use_container_width=True)
        st.caption("ℹ️ Evaluate All uses the reply field above for **every** company — leave it empty for profile-only batch assessment.")

        if evaluate_all_btn:
            if assessment_mode == "✍️ Custom Criteria" and not custom_criteria_text.strip():
                st.error("Please define your scoring criteria before evaluating.")
            else:
                prog = st.progress(0)
                stat = st.empty()
                ok, fail = 0, 0
                for idx, e in enumerate(emails):
                    cn = e.get("company_name", "")
                    stat.text(f"🤖 Evaluating {idx+1}/{len(emails)}: {cn}")
                    try:
                        ctx = {
                            "company_name": cn,
                            "sector": e.get("sector", ""),
                            "sub_sector": e.get("sub_sector", ""),
                            "employees": e.get("employees", ""),
                            "city": e.get("city", ""),
                            "description": e.get("description", ""),
                            "tags": e.get("tags", ""),
                            "founded_year": e.get("founded_year", ""),
                            "email_subject": e.get("email_subject", ""),
                        }
                        if assessment_mode == "🎯 Auto Assessment (BANT)":
                            r = evaluate_opportunity_auto(ctx, email_reply)
                        else:
                            r = evaluate_opportunity_custom(ctx, custom_criteria_text, email_reply)
                        r["company_name"] = cn
                        r["email_reply"] = email_reply
                        r["assessment_mode"] = assessment_mode
                        st.session_state.opportunity_results[cn] = r
                        ok += 1
                    except Exception as ex:
                        st.error(f"Failed for {cn}: {ex}")
                        fail += 1
                    prog.progress((idx + 1) / max(1, len(emails)))
                stat.empty()
                prog.empty()
                st.success(f"✅ Evaluated {ok}. Failed {fail}.")
                st.rerun()

        if evaluate_btn:
            if assessment_mode == "✍️ Custom Criteria" and not custom_criteria_text.strip():
                st.error("Please define your scoring criteria before evaluating.")
            else:
                with st.spinner("🤖 AI is evaluating the opportunity..."):
                    try:
                        company_ctx = {
                            "company_name": picked_email.get("company_name", ""),
                            "sector": picked_email.get("sector", ""),
                            "sub_sector": picked_email.get("sub_sector", ""),
                            "employees": picked_email.get("employees", ""),
                            "city": picked_email.get("city", ""),
                            "description": picked_email.get("description", ""),
                            "tags": picked_email.get("tags", ""),
                            "founded_year": picked_email.get("founded_year", ""),
                            "email_subject": picked_email.get("email_subject", ""),
                        }

                        if assessment_mode == "🎯 Auto Assessment (BANT)":
                            result = evaluate_opportunity_auto(company_ctx, email_reply)
                        else:
                            result = evaluate_opportunity_custom(company_ctx, custom_criteria_text, email_reply)

                        result["company_name"] = picked_email.get("company_name", "")
                        result["email_reply"] = email_reply
                        result["assessment_mode"] = assessment_mode
                        st.session_state.opportunity_results[picked_email.get("company_name", "")] = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # ── Show result for selected company ───────────────────────
        existing = st.session_state.opportunity_results.get(picked_email.get("company_name", ""))
        if existing:
            st.markdown("---")
            score = existing.get("opportunity_score", 0)
            status = existing.get("status", "")
            mode_label = existing.get("assessment_mode", "")
            emoji = "🟢" if status == "Qualified" else "🔴" if status == "Not Qualified" else "⚪"

            with st.container(border=True):
                st.markdown(f"### {emoji} Score: **{score}/100** — `{status}`")
                st.caption(f"Assessment mode: {mode_label}")
                st.markdown(f"💬 **Summary:** {existing.get('reason', '')}")
                st.markdown(f"➡️ **Next step:** {existing.get('recommended_next_step', '')}")

                # BANT breakdown
                if existing.get("bant"):
                    st.markdown("---")
                    st.markdown("#### 📊 BANT Breakdown")
                    bant = existing["bant"]
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        b_score = bant.get("budget_score", 0)
                        st.markdown(f'<div class="metric-card"><div class="metric-num">{b_score}/25</div><div class="metric-lbl">💰 Budget</div></div>', unsafe_allow_html=True)
                        st.caption(bant.get("budget_reason", ""))
                    with b2:
                        a_score = bant.get("authority_score", 0)
                        st.markdown(f'<div class="metric-card"><div class="metric-num">{a_score}/25</div><div class="metric-lbl">🧑‍💼 Authority</div></div>', unsafe_allow_html=True)
                        st.caption(bant.get("authority_reason", ""))
                    with b3:
                        n_score = bant.get("need_score", 0)
                        st.markdown(f'<div class="metric-card"><div class="metric-num">{n_score}/25</div><div class="metric-lbl">🎯 Need</div></div>', unsafe_allow_html=True)
                        st.caption(bant.get("need_reason", ""))
                    with b4:
                        t_score = bant.get("timeline_score", 0)
                        st.markdown(f'<div class="metric-card"><div class="metric-num">{t_score}/25</div><div class="metric-lbl">⏱️ Timeline</div></div>', unsafe_allow_html=True)
                        st.caption(bant.get("timeline_reason", ""))

                # Custom criteria breakdown
                if existing.get("criteria_scores"):
                    st.markdown("---")
                    st.markdown("#### 📋 Criteria Breakdown")
                    for c in existing["criteria_scores"]:
                        c_score = c.get("score", 0)
                        color = "#4ade80" if c_score >= 70 else "#facc15" if c_score >= 40 else "#f87171"
                        with st.container(border=True):
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.markdown(f"**{c.get('criterion', '')}**")
                                st.caption(c.get("reason", ""))
                            with cols[1]:
                                st.markdown(f'<div style="text-align:center;font-size:1.4rem;font-weight:800;color:{color}">{c_score}</div>', unsafe_allow_html=True)

        # ── Summary table of all evaluated opportunities ───────────
        if st.session_state.opportunity_results:
            st.markdown("---")
            st.markdown("#### 📊 All Evaluated Opportunities")

            all_results = list(st.session_state.opportunity_results.values())
            qualified_count = sum(1 for r in all_results if r.get("status") == "Qualified")

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-num">{len(all_results)}</div><div class="metric-lbl">Evaluated</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4ade80">{qualified_count}</div><div class="metric-lbl">🟢 Qualified — Proposal Ready</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in sorted(all_results, key=lambda x: x.get("opportunity_score", 0), reverse=True):
                status = r.get("status", "")
                emoji = "🟢" if status == "Qualified" else "🔴" if status == "Not Qualified" else "⚪"
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{r.get('company_name', 'N/A')}**")
                        st.caption(f"💬 {r.get('reason', '')}")
                        st.caption(f"Mode: {r.get('assessment_mode', '')}")
                    with c2:
                        st.markdown(f"### {emoji} {r.get('opportunity_score', 0)}")

            st.markdown("---")

            import csv as csv_module2, io as io_module2
            output2 = io_module2.StringIO()
            fieldnames2 = [
                "company_name", "assessment_mode", "email_reply",
                "opportunity_score", "status", "reason", "recommended_next_step"
            ]
            writer2 = csv_module2.DictWriter(output2, fieldnames=fieldnames2, extrasaction='ignore')
            writer2.writeheader()
            for r in all_results:
                writer2.writerow(r)
            csv_data2 = output2.getvalue()

            st.download_button(
                label="⬇️ Export Opportunities as CSV",
                data=csv_data2,
                file_name="qualified_opportunities.csv",
                mime="text/csv",
                use_container_width=True
            )

with tab7:
    st.markdown("### 📄 Proposal Generator")
    st.markdown(
        "Pick a **Qualified** opportunity, confirm the agreed price, and the "
        "agent will research the company online and in Beam Data's knowledge "
        "base to write a tailored proposal — exported as a PDF."
    )

    qualified = {
        name: r for name, r in st.session_state.opportunity_results.items()
        if r.get("status") == "Qualified"
    }

    if not qualified:
        st.warning("⚠️ No qualified opportunities yet. Evaluate one in the 🤝 Opportunity tab first.")
    else:
        st.markdown("---")
        company_options2 = list(qualified.keys())
        picked_company = st.selectbox("Select a qualified company:", company_options2, key="proposal_company_pick")
        picked_opp = qualified[picked_company]

        with st.container(border=True):
            st.markdown(f"**🏢 {picked_company}**")
            st.caption(f"💬 {picked_opp.get('reason', '')}")

            agreed_price = st.text_input(
                "💰 Agreed price (optional — leave blank to let the agent estimate)",
                placeholder="e.g. $15,000 for a 3-month engagement",
                key="proposal_price_input"
            )

            gen_col_one, gen_col_all = st.columns(2)
            with gen_col_one:
                generate_proposal_btn = st.button("📄 Generate Proposal", type="primary", use_container_width=True)
            with gen_col_all:
                generate_all_btn = st.button(f"📄 Generate All {len(qualified)} Proposals", use_container_width=True)
            st.caption("ℹ️ Generate All auto-approves the HITL gate per proposal — the single click here is your batch approval.")

        if generate_proposal_btn:
            try:
                with st.spinner("Building knowledge base (first run may take a minute)…"):
                    retriever = initialize_rag()
                    proposal_executor = build_proposal_agent(retriever)

                with st.spinner(f"Researching **{picked_company}** and drafting the proposal…"):
                    start_res = proposal_executor.start({
                        "company_name": picked_company,
                        "agreed_price": agreed_price.strip() if agreed_price.strip() else "Not specified — please estimate",
                    })

                if start_res["state"] == "error":
                    st.error(f"Guardrail blocked proposal: {start_res.get('error','')}")
                elif start_res["state"] == "awaiting_approval":
                    st.session_state.proposal_workflow = {
                        "executor": proposal_executor,
                        "thread_id": start_res["thread_id"],
                        "preview": start_res.get("preview", {}),
                        "company_name": picked_company,
                    }
                    st.rerun()
                elif start_res["state"] == "finalized":
                    # Content-safety / guardrail short-circuited HITL — accept output directly.
                    full_output = start_res.get("output", "")
                    if full_output:
                        st.session_state.proposal_text = full_output
                        pdf_path = f"/tmp/proposal_{picked_company.replace(' ', '_')}.pdf"
                        generate_proposal_pdf(picked_company, full_output, pdf_path)
                        st.session_state.proposal_pdf_path = pdf_path
                        st.rerun()
                    else:
                        st.error("Proposal generation produced no output.")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

        # ── HITL gate: graph paused after draft, awaiting human approval ──
        if st.session_state.proposal_workflow:
            wf = st.session_state.proposal_workflow
            preview = wf.get("preview", {})
            st.markdown("---")
            st.markdown("#### 🤝 Review Draft Before PDF Generation")
            with st.container(border=True):
                st.caption(f"Draft length: {preview.get('length', 0)} characters")
                st.markdown("**Preview (first 600 chars):**")
                st.text(preview.get("preview", ""))

                a_col, r_col = st.columns(2)
                with a_col:
                    if st.button("✅ Approve — Generate PDF", type="primary", use_container_width=True, key="proposal_approve"):
                        res = wf["executor"].resume(wf["thread_id"], approved=True)
                        if res["state"] == "finalized" and res.get("output"):
                            full_output = res["output"]
                            st.session_state.proposal_text = full_output
                            pdf_path = f"/tmp/proposal_{wf['company_name'].replace(' ', '_')}.pdf"
                            generate_proposal_pdf(wf["company_name"], full_output, pdf_path)
                            st.session_state.proposal_pdf_path = pdf_path
                        else:
                            st.error(f"Could not finalize: {res.get('error','unknown')}")
                        st.session_state.proposal_workflow = None
                        st.rerun()
                with r_col:
                    if st.button("❌ Reject Draft", use_container_width=True, key="proposal_reject"):
                        wf["executor"].resume(wf["thread_id"], approved=False)
                        st.session_state.proposal_workflow = None
                        st.info("Draft rejected — no PDF generated.")
                        st.rerun()

        # ── Generate All (batch HITL — single approval via the button click) ──
        if generate_all_btn:
            try:
                with st.spinner("Building knowledge base (first run may take a minute)…"):
                    retriever = initialize_rag()
                    proposal_executor = build_proposal_agent(retriever)

                prog = st.progress(0)
                stat = st.empty()
                ok, fail = 0, 0
                price_arg = (agreed_price.strip()
                             if agreed_price.strip()
                             else "Not specified — please estimate")

                for idx, cn in enumerate(qualified.keys()):
                    stat.text(f"📄 Drafting {idx+1}/{len(qualified)}: {cn}")
                    try:
                        start_res = proposal_executor.start({
                            "company_name": cn,
                            "agreed_price": price_arg,
                        })
                        if start_res["state"] == "awaiting_approval":
                            final = proposal_executor.resume(start_res["thread_id"], approved=True)
                        else:
                            final = start_res
                        if final.get("state") == "finalized" and final.get("output"):
                            output = final["output"]
                            pdf_path = f"/tmp/proposal_{cn.replace(' ', '_')}.pdf"
                            generate_proposal_pdf(cn, output, pdf_path)
                            st.session_state.all_proposals[cn] = {
                                "text": output, "pdf_path": pdf_path,
                            }
                            ok += 1
                        else:
                            fail += 1
                    except Exception as ex:
                        st.error(f"Failed for {cn}: {ex}")
                        fail += 1
                    prog.progress((idx + 1) / max(1, len(qualified)))
                stat.empty()
                prog.empty()
                st.success(f"✅ Generated {ok}. Failed {fail}.")
                st.rerun()
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

        # ── Display batch-generated proposals ──
        if st.session_state.all_proposals:
            st.markdown("---")
            st.markdown(f"#### 📚 {len(st.session_state.all_proposals)} Proposals Ready")
            for cn, prop in st.session_state.all_proposals.items():
                with st.container(border=True):
                    st.markdown(f"**🏢 {cn}**")
                    with st.expander("View proposal text"):
                        st.markdown(prop["text"])
                    try:
                        with open(prop["pdf_path"], "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download {cn} (PDF)",
                                data=f.read(),
                                file_name=prop["pdf_path"].split("/")[-1],
                                mime="application/pdf",
                                key=f"dl_all_{cn}",
                                use_container_width=True,
                            )
                    except FileNotFoundError:
                        st.warning("PDF file no longer available on disk.")
            if st.button("🔄 Clear All Proposals", key="clear_all_proposals", use_container_width=True):
                st.session_state.all_proposals = {}
                st.rerun()

        if st.session_state.proposal_text:
            st.markdown("---")
            st.markdown(f"#### ✅ Proposal Ready")
            with st.container(border=True):
                st.markdown(st.session_state.proposal_text)

            if st.session_state.proposal_pdf_path:
                with open(st.session_state.proposal_pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Proposal (PDF)",
                        data=f.read(),
                        file_name=st.session_state.proposal_pdf_path.split("/")[-1],
                        mime="application/pdf",
                        use_container_width=True
                    )

            if st.button("🔄 Clear Proposal", use_container_width=True):
                st.session_state.proposal_text = None
                st.session_state.proposal_pdf_path = None
                st.rerun()

st.markdown("---")
st.caption("Just Sell It • AI Sales & CRM Agent • Built with Streamlit & Plotly")