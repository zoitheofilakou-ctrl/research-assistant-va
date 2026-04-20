import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv
from project_paths import FILTERED_PAPERS_PATH, RAG_STORE_DIR

load_dotenv()

try:
    from data_acquisition.scraper import fetch_rehabilitation_papers
    from llm.rag_generator import generate_rag_answer
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()


def semantic_label(value):
    if value is None:
        return None
    if value >= 0.50:
        return "High"
    elif value >= 0.40:
        return "Moderate"
    else:
        return "Low"


def format_text_source(value):
    """Translate internal text_source values to researcher-facing labels."""
    if value == "fulltext":
        return "full paper"
    if value == "abstract":
        return "abstract only"
    return value or ""


st.set_page_config(
    page_title="HyBreDe — Evidence Retrieval System",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg:      #f4efe8;
    --surface: #faf7f2;
    --surface2:#f0ece4;
    --surface3:#ede8df;
    --ink:     #0d1b2a;
    --ink2:    #1e3448;
    --muted:   #4a6377;
    --pale:    #e8f2f9;
    --cyan:    #0077aa;
    --teal:    #007c6e;
    --gold:    #c08000;
    --red:     #c0392b;
    --green:   #1a7a4a;
    --border:  rgba(13,27,42,0.12);
    --border2: rgba(13,27,42,0.22);
    --mono:    'DM Mono', monospace;
    --sans:    'Inter', sans-serif;
    --serif:   'DM Serif Display', serif;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"],
.stApp, .main, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    font-family: var(--sans) !important;
    background-color: #f4efe8 !important;
    color: #0d1b2a !important;
}
.main .block-container {
    padding-top: 0 !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
    background-color: #f4efe8 !important;
}

/* ── Hide Streamlit toolbar / deploy button / top bar ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
header[data-testid="stHeader"],
.stDeployButton,
[data-testid="stAppDeployButton"] {
    display: none !important;
    visibility: hidden !important;
}

/* iframe fix */
iframe { background: transparent !important; color-scheme: light !important; }
[data-testid="stCustomComponentV1"],
[data-testid="stCustomComponentV1"] > div,
[data-testid="stCustomComponentV1"] iframe { background: transparent !important; }
.element-container iframe { background-color: #f4efe8 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--surface2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1rem 1.1rem 1rem !important;
}
section[data-testid="stSidebar"] * { color: var(--muted) !important; font-family: var(--sans) !important; }
section[data-testid="stSidebar"] h3 {
    font-family: var(--mono) !important;
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--ink) !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.5rem !important;
    margin: 1.2rem 0 0.8rem !important;
}
section[data-testid="stSidebar"] label { font-size: 0.78rem !important; font-weight: 500 !important; color: var(--ink2) !important; }
section[data-testid="stSidebar"] iframe { background-color: #faf7f2 !important; }
section[data-testid="stSidebar"] hr { margin: 0.9rem 0 !important; }
section[data-testid="stSidebar"] .stSuccess,
section[data-testid="stSidebar"] .stWarning,
section[data-testid="stSidebar"] .stError {
    padding: 0.5rem 0.8rem !important;
    font-size: 0.68rem !important;
    margin-bottom: 0.5rem !important;
}

/* Tabs */
.stTabs { margin-left: 0 !important; margin-right: 0 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important; padding: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 1rem 2rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom: 2px solid var(--ink) !important; background: transparent !important; }
.stTabs [data-baseweb="tab"]:hover { color: var(--ink2) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 2rem !important; padding-bottom: 2.5rem !important; }

/* Text inputs */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; font-style: italic !important; opacity: 1 !important; }
.stTextInput > div > div > input:focus { border-color: var(--cyan) !important; box-shadow: 0 0 0 3px rgba(0,119,170,0.1) !important; outline: none !important; }

/* Chat input */
[data-testid="stBottom"] {
    background: #f4efe8 !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
    border-top: 1px solid var(--border) !important;
    box-sizing: border-box !important;
}
.stChatInputContainer { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; padding: 0.5rem 0.8rem !important; }
.stChatInputContainer textarea { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 2px !important; color: var(--ink) !important; font-family: var(--sans) !important; font-size: 0.9rem !important; }
.stChatInputContainer textarea:focus { border-color: var(--cyan) !important; box-shadow: 0 0 0 3px rgba(0,119,170,0.1) !important; }

/* Buttons */
.stButton > button {
    background: var(--ink) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 2px !important;
    color: #faf7f2 !important;
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}
.stButton > button:hover { background: var(--ink2) !important; border-color: var(--ink2) !important; box-shadow: 0 2px 8px rgba(13,27,42,0.18) !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* Link buttons */
.stLinkButton > a {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    border-radius: 2px !important;
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.35rem 0.8rem !important;
    text-decoration: none !important;
    transition: all 0.15s !important;
}
.stLinkButton > a:hover { background: rgba(0,119,170,0.06) !important; border-color: var(--cyan) !important; }

/* Chat messages */
[data-testid="stChatMessage"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 1.2rem !important; padding: 1rem 1.3rem !important; }
[data-testid="stChatMessage"] p { color: var(--ink) !important; font-family: var(--sans) !important; font-size: 0.9rem !important; line-height: 1.75 !important; }
[data-testid="stChatMessage"] li { color: var(--ink2) !important; font-size: 0.88rem !important; line-height: 1.7 !important; }
[data-testid="stChatMessage"] strong { color: var(--ink) !important; font-weight: 600 !important; }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { font-family: var(--mono) !important; font-size: 0.62rem !important; font-weight: 500 !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: var(--muted) !important; margin-top: 1.4rem !important; margin-bottom: 0.6rem !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0.4rem !important; }

/* Markdown */
.stMarkdown p { color: var(--ink) !important; font-size: 0.9rem !important; line-height: 1.75 !important; margin-bottom: 0.5rem !important; }
.stMarkdown li { color: var(--ink2) !important; font-size: 0.88rem !important; line-height: 1.7 !important; margin-bottom: 0.2rem !important; }
.stMarkdown strong { color: var(--ink) !important; font-weight: 600 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { font-family: var(--mono) !important; font-size: 0.62rem !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0.4rem !important; margin-top: 1.4rem !important; }

/* Metrics */
[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; padding: 1.1rem 1.4rem !important; transition: border-color 0.2s, box-shadow 0.2s !important; }
[data-testid="stMetric"]:hover { border-color: var(--border2) !important; box-shadow: 0 2px 12px rgba(13,27,42,0.08) !important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] span { font-family: var(--mono) !important; font-size: 0.58rem !important; font-weight: 500 !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: var(--muted) !important; }
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] span,
[data-testid="stMetricValue"] > div { font-family: var(--mono) !important; font-size: 1.8rem !important; font-weight: 500 !important; color: var(--cyan) !important; letter-spacing: -0.02em !important; }

/* Expander */
.streamlit-expanderHeader { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; color: var(--ink2) !important; font-family: var(--mono) !important; font-size: 0.62rem !important; font-weight: 500 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; padding: 0.95rem 1.3rem !important; transition: background 0.2s !important; }
.streamlit-expanderHeader:hover { background: var(--surface3) !important; }
.streamlit-expanderContent { background: var(--surface) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 2px 2px !important; padding: 1.5rem 1.6rem 1.6rem !important; }

/* Containers — scoped to tabs only to avoid polluting sidebar/page wrappers */
.stTabs [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    transition: border-color 0.2s, box-shadow 0.15s !important;
}
.stTabs [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border2) !important;
    box-shadow: 0 2px 12px rgba(13,27,42,0.08) !important;
}

/* Selectbox */
.stSelectbox > div > div { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 2px !important; color: var(--ink) !important; font-family: var(--sans) !important; font-size: 0.88rem !important; }

/* Slider */
.stSlider > div > div > div > div { background: var(--cyan) !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { font-family: var(--mono) !important; font-size: 0.6rem !important; color: var(--muted) !important; }

/* Status */
.stSuccess { background: rgba(26,122,74,0.07) !important; border: 1px solid rgba(26,122,74,0.25) !important; border-radius: 2px !important; color: #1a7a4a !important; font-family: var(--mono) !important; font-size: 0.72rem !important; padding: 0.65rem 1rem !important; }
.stWarning { background: rgba(192,128,0,0.07) !important; border: 1px solid rgba(192,128,0,0.25) !important; border-radius: 2px !important; color: var(--gold) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; padding: 0.65rem 1rem !important; }
.stError { background: rgba(192,57,43,0.07) !important; border: 1px solid rgba(192,57,43,0.25) !important; border-radius: 2px !important; color: var(--red) !important; padding: 0.65rem 1rem !important; }
.stInfo { padding: 0.65rem 1rem !important; border-radius: 2px !important; }

/* Caption */
.stCaption, small { font-family: var(--mono) !important; font-size: 0.62rem !important; font-weight: 400 !important; color: var(--muted) !important; letter-spacing: 0.04em !important; line-height: 1.65 !important; margin-bottom: 0.25rem !important; }

/* HR / Spinner */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* Checkbox */
.stCheckbox > label > span { color: var(--ink2) !important; font-size: 0.82rem !important; }

/* Footer */
footer, [data-testid="stFooter"] { background: transparent !important; color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙ Configuration")

    provider = st.selectbox("LLM Provider", ["openai", "ollama"])
    ollama_model = None
    if provider == "ollama":
        ollama_model = st.text_input(
            "Ollama model",
            value=os.getenv("OLLAMA_MODEL", "phi"),
            help="Example: qwen3:14b",
        )
    k_articles = st.slider("Retrieved papers (k)", 1, 10, 5)

    st.markdown("---")
    st.markdown("### ◈ Pipeline Status")

    if os.path.exists(RAG_STORE_DIR):
        st.success("✓  Vector index ready")
    else:
        st.warning("⚠  No vector index")

    if os.path.exists(FILTERED_PAPERS_PATH):
        st.success("✓  Screened corpus available")
    else:
        st.warning("⚠  No screened corpus")

    st.markdown("---")
    components.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
    html, body { margin:0; padding:0.2rem 0.8rem 0 0.8rem; background:#faf7f2 !important; background-color:#faf7f2 !important; }
    .sb-footer { font-family:'DM Mono',monospace; font-size:0.62rem; color:#4a6377; line-height:2.3; padding:0.6rem 0.2rem 0.4rem; }
    .sb-footer .divider { height:1px; background:rgba(13,27,42,0.12); margin-bottom:0.7rem; }
    .sb-footer .brand { color:#0d1b2a; font-weight:500; }
    .sb-footer .hl    { color:#0077aa; }
    .sb-footer .warn  { color:#c08000; font-size:0.58rem; }
    </style>
    <div class="sb-footer">
        <div class="divider"></div>
        <span class="brand">HyBreDe</span> · Turku UAS ICT · 2026<br>
        Governance-Aware RAG Pipeline<br>
        <span class="hl">Evidence-grounded</span> · <span class="warn">No clinical use</span>
    </div>
    """, height=100)

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
html, body { background:#0d1b2a !important; background-color:#0d1b2a !important; overflow:hidden; }
body { padding:2.2rem 2.5rem 1.8rem; }

.header { position:relative; padding-bottom:1.8rem; }
.header::after { content:''; position:absolute; bottom:0; left:0; right:0; height:1px; background:rgba(250,247,242,0.12); }

.project-label {
    font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.24em;
    text-transform:uppercase; color:rgba(250,247,242,0.35); margin-bottom:1rem;
    opacity:0; animation:fadeUp 0.5s 0.05s forwards;
}

.title {
    font-family:'DM Serif Display',serif; font-size:4.8rem; font-weight:400;
    color:#faf7f2; letter-spacing:-0.02em; line-height:1;
    opacity:0; animation:fadeUp 0.6s 0.15s forwards;
}
.title em { font-style:italic; color:#7ecfea; }

.header-bottom {
    display:flex; justify-content:space-between; align-items:center;
    margin-top:1rem;
}

.subtitle {
    font-family:'Inter',sans-serif; font-size:0.78rem; color:rgba(250,247,242,0.5);
    letter-spacing:0.04em; display:flex; gap:1.6rem; flex-wrap:wrap;
    opacity:0; animation:fadeUp 0.6s 0.25s forwards;
}
.subtitle-item { display:flex; align-items:center; gap:0.45rem; transition:color 0.2s; cursor:default; }
.subtitle-item::before { content:'·'; color:rgba(126,207,234,0.5); }
.subtitle-item:hover { color:rgba(250,247,242,0.8); }

.logo-wrap {
    opacity:0; animation:fadeUp 0.6s 0.3s forwards;
    flex-shrink:0;
}
.logo-wrap img {
    height:44px; width:auto;
    display:block;
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
</style>
<div class="header">
    <div class="project-label">Healthcare Literature Pre-Screening System &nbsp;·&nbsp; v2026</div>
    <div class="title">Hyb<em>Re</em>De</div>
    <div class="header-bottom">
        <div class="subtitle">
            <div class="subtitle-item">Governance-Aware RAG Pipeline</div>
            <div class="subtitle-item">AI-Assisted Evidence Synthesis</div>
            <div class="subtitle-item">Human Oversight Preserved</div>
            <div class="subtitle-item">Turku UAS · ICT 2026</div>
        </div>
    </div>
</div>
""", height=235)


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_ask, tab_search = st.tabs(["◉  Ask the Assistant", "◇  External Search (Semantic Scholar)"])


def section_header(tag, desc):
    st.markdown(f"""
    <div style="padding-top:0.5rem;margin-bottom:2rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;font-weight:500;
        letter-spacing:0.2em;text-transform:uppercase;color:#4a6377;margin-bottom:0.65rem;
        display:flex;align-items:center;gap:0.5rem;">
            <span style="display:inline-block;width:16px;height:1px;background:rgba(13,27,42,0.2);"></span>
            {tag}
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#4a6377;line-height:1.8;max-width:72ch;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── TAB 1 ───────────────────────────────────────────────────────────────────
with tab_ask:

    section_header(
        "Evidence Synthesis Engine",
        "The assistant retrieves evidence exclusively from the pre-screened corpus and synthesizes a structured response. "
        "It does not use external knowledge and does not provide clinical recommendations. "
        "All responses are grounded in retrieved literature with traceable citations."
    )

    if not os.path.exists(RAG_STORE_DIR):
        st.warning("No vector index found. Build the index before asking questions.")

    user_input = st.chat_input("Ask a research question about the pre-screened corpus...")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                sources = msg["sources"]
                with st.expander(f"◈ Evidence Sources · {len(sources)} Top-k Retrieved Results"):
                    st.caption(
                        "Top-k results after hybrid retrieval (embedding + lexical + reranking). "
                        "Signals reflect retrieval alignment (semantic and lexical), "
                        "not evidence quality, methodological rigor, or clinical validity."
                    )
                    st.caption(
                        "Ranking reflects retrieval relevance, not strength of evidence, study quality, or clinical validity."
                    )
                    st.caption("Not all retrieved papers are necessarily used in the final answer; only those selected during generation are cited.")
                    st.caption("Corpus composition: 27 full-text papers · 80 abstract-only papers")

                    for idx, s in enumerate(sources, start=1):
                        paper_num       = s.get("rank", idx)
                        title_s         = s.get("title", "Unknown")
                        year_s          = s.get("year", "")
                        url_s           = s.get("url", "")
                        pid             = s.get("paperId", "")
                        sem             = s.get("semantic_similarity")
                        keyword_overlap = s.get("keyword_overlap")
                        ce              = s.get("cross_encoder_score")
                        src             = format_text_source(s.get("text_source", ""))

                        if not s.get("text", "").strip():
                            st.warning("No textual evidence available for this result.")
                            continue

                        st.markdown(f"**Paper {paper_num} — {title_s}**")
                        st.caption(f"{year_s} · {src}")

                        if src == "abstract only":
                            st.warning("Abstract-level evidence: limited detail and lower evidential strength.")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Position in results",
                                str(paper_num),
                                help="Retrieval rank — lower is more relevant. Not a measure of evidence quality."
                            )

                        with col2:
                            if sem is not None:
                                label = semantic_label(sem)
                                st.metric(
                                    "Semantic similarity",
                                    f"{sem:.2f} ({label})",
                                    help="How closely this text's meaning matches your query. High ≥ 0.50 · Moderate ≥ 0.40 · Low < 0.40."
                                )

                        with col3:
                            st.metric(
                                "Keyword overlap",
                                keyword_overlap if keyword_overlap else "N/A",
                                help="How many of your query terms appear in this text. Shared vocabulary, not guaranteed relevance."
                            )

                        with col4:
                            st.metric(
                                "Relevance (reranker)",
                                f"{ce:.2f}" if ce else "N/A",
                                help="Fine-grained relevance score from the reranker — the most reliable of the three signals. Still a retrieval metric, not a quality judgment."
                            )

                        if pid:
                            st.caption(f"ID: {pid}")
                        if url_s:
                            st.link_button("↗ Open", url_s)

    if user_input:

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence · Generating response..."):
                try:
                    result = generate_rag_answer(
                        user_input,
                        provider=provider,
                        k=k_articles,
                        answer_template="structured",
                        output_mode="text",
                        **({"model": ollama_model} if provider == "ollama" and ollama_model else {}),
                    )

                    answer                = result.get("answer", "No answer generated.")
                    sources               = result.get("sources", [])
                    insufficient_evidence = result.get("insufficient_evidence", False)

                    # ── ANSWER LOGIC ─────────────────────────────
                    if insufficient_evidence:
                        st.error(
                            "No sufficiently relevant evidence found in the screened corpus. "
                            "The system abstains from generating a grounded answer."
                        )

                        if sources:
                            st.caption(
                                "Retrieved results are shown below for transparency, "
                                "but none were considered sufficiently relevant for synthesis."
                            )
                        else:
                            st.caption(
                                "No relevant results were retrieved for this query."
                            )
                    else:
                        st.markdown(answer)

                    # ── ALWAYS SHOW SOURCES ─────────────────────
                    if sources:
                        with st.expander(f"◈ Evidence Sources · {len(sources)} Top-k Retrieved Results"):

                            st.caption(
                                "Top-k results after hybrid retrieval (embedding + lexical + reranking). "
                                "Signals reflect retrieval alignment (semantic and lexical), "
                                "not evidence quality, methodological rigor, or clinical validity."
                            )
                            st.caption(
                                "Ranking reflects retrieval relevance, not strength of evidence, study quality, or clinical validity."
                            )
                            st.caption("Not all retrieved papers are necessarily used in the final answer; only those selected during generation are cited.")
                            st.caption("Corpus composition: 27 full-text papers · 80 abstract-only papers")

                            for idx, s in enumerate(sources, start=1):
                                paper_num       = s.get("rank", idx)
                                title_s         = s.get("title", "Unknown")
                                year_s          = s.get("year", "")
                                url_s           = s.get("url", "")
                                pid             = s.get("paperId", "")
                                sem             = s.get("semantic_similarity")
                                keyword_overlap = s.get("keyword_overlap")
                                ce              = s.get("cross_encoder_score")
                                src             = format_text_source(s.get("text_source", ""))

                                if not s.get("text", "").strip():
                                    st.warning("No textual evidence available for this result.")
                                    continue

                                st.markdown(f"**Paper {paper_num} — {title_s}**")
                                st.caption(f"{year_s} · {src}")

                                if src == "abstract only":
                                    st.warning("Abstract-level evidence: limited detail and lower evidential strength.")

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric(
                                        "Position in results",
                                        str(paper_num),
                                        help="Retrieval rank — lower is more relevant. Not a measure of evidence quality."
                                    )

                                with col2:
                                    if sem is not None:
                                        label = semantic_label(sem)
                                        st.metric(
                                            "Semantic similarity",
                                            f"{sem:.2f} ({label})",
                                            help="How closely this text's meaning matches your query. High ≥ 0.50 · Moderate ≥ 0.40 · Low < 0.40."
                                        )

                                with col3:
                                    st.metric(
                                        "Keyword overlap",
                                        keyword_overlap if keyword_overlap else "N/A",
                                        help="How many of your query terms appear in this text. Shared vocabulary, not guaranteed relevance."
                                    )

                                with col4:
                                    st.metric(
                                        "Relevance (reranker)",
                                        f"{ce:.2f}" if ce else "N/A",
                                        help="Fine-grained relevance score from the reranker — the most reliable of the three signals. Still a retrieval metric, not a quality judgment."
                                    )

                                if pid:
                                    st.caption(f"ID: {pid}")
                                if url_s:
                                    st.link_button("↗ Open", url_s)

                    # ── SAVE CHAT ONLY IF ANSWER EXISTS ─────────
                    if not insufficient_evidence:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                except Exception as e:
                    st.error(f"Generation error: {e}")

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ─── TAB 2 ───────────────────────────────────────────────────────────────────
with tab_search:

    st.info(
        "This tab queries Semantic Scholar directly and is not connected to the HyBreDe pipeline. "
        "Results are unscreened and cannot be used as grounded evidence by the system."
    )

    section_header(
        "External Search · Semantic Scholar API",
        "External search for background reading and orientation. "
        "Results are not screened, not indexed, and not used for evidence synthesis. "
        "This tool operates independently of the HyBreDe pipeline."
    )

    col_input, col_btn = st.columns([5, 1])

    with col_input:
        search_query = st.text_input("", key="search_query", placeholder="e.g. human oversight automation bias AI clinical decision support", label_visibility="collapsed")

    with col_btn:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        if not search_query.strip():
            st.warning("Enter a search topic first.")
        else:
            with st.spinner("Querying Semantic Scholar..."):
                results = fetch_rehabilitation_papers(search_query, result_limit=k_articles)
                st.session_state.search_results = results

    if st.session_state.search_results:
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:0.6rem;
        font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;
        text-transform:uppercase;color:#4a6377;padding:0.4rem 0.9rem;
        background:#faf7f2;border:1px solid rgba(13,27,42,0.12);border-radius:2px;margin:0.8rem 0;">
            <span style="color:#0077aa;font-size:0.9rem;font-weight:500;">{len(st.session_state.search_results)}</span>
            papers retrieved
        </div>
        """, unsafe_allow_html=True)

        for i, paper in enumerate(st.session_state.search_results):
            title      = paper.get("title", "No title")
            abstract   = paper.get("abstract", "No abstract available.")
            year       = paper.get("year", "N/A")
            url        = paper.get("url", "")
            authors    = paper.get("authors", [])
            author_str = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_str += " et al."

            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"{year}  ·  {author_str if author_str else 'Unknown authors'}")
                with st.expander("Abstract"):
                    st.write(abstract)
                if url:
                    st.link_button("↗ Open paper", url)

    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
            letter-spacing:0.2em;color:rgba(13,27,42,0.2);margin-bottom:0.6rem;">NO RESULTS</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#4a6377;">
            Enter a research topic above and click Search</div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:1rem 1.5rem;margin-top:2.5rem;
border-top:1px solid rgba(13,27,42,0.12);
font-family:'DM Mono',monospace;font-size:0.6rem;color:#4a6377;letter-spacing:0.06em;">
    <div style="color:#1e3448;">HyBreDe · Turku University of Applied Sciences · ICT 2026</div>
    <div>
        <span style="color:#0077aa;">Evidence-grounded</span>
        <span style="color:rgba(13,27,42,0.3);">&nbsp;·&nbsp; No clinical recommendations &nbsp;·&nbsp;</span>
        <span style="color:#007c6e;">Human oversight preserved</span>
    </div>
</div>
""", unsafe_allow_html=True)
