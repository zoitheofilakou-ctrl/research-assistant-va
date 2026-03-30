import streamlit as st
import streamlit.components.v1 as components
import json
import os
from dotenv import load_dotenv
from project_paths import FILTERED_PAPERS_PATH, RAG_STORE_DIR, ensure_parent_dir

load_dotenv()

try:
    from data_acquisition.scraper import fetch_rehabilitation_papers
    from llm.rag_generator import generate_rag_answer
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()


def format_score(value):
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return None


st.set_page_config(
    page_title="HyBreDe — Research Assistant",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL CSS ────────────────────────────────────────────────────────────────
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
    max-width: 100% !important;
    background-color: #f4efe8 !important;
}

/* ── Hide Streamlit toolbar / deploy button / top bar ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,

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

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0 !important; padding: 0 !important;
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
.stChatInputContainer { background: var(--surface2) !important; border-top: 1px solid var(--border) !important; padding: 1rem !important; }
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
[data-testid="stChatMessage"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 0.8rem !important; padding: 0.4rem !important; }
[data-testid="stChatMessage"] p { color: var(--ink) !important; font-family: var(--sans) !important; font-size: 0.9rem !important; line-height: 1.75 !important; }
[data-testid="stChatMessage"] li { color: var(--ink2) !important; font-size: 0.88rem !important; line-height: 1.7 !important; }
[data-testid="stChatMessage"] strong { color: var(--ink) !important; font-weight: 600 !important; }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { font-family: var(--mono) !important; font-size: 0.62rem !important; font-weight: 500 !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: var(--muted) !important; margin-top: 1.4rem !important; margin-bottom: 0.6rem !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0.4rem !important; }

/* Markdown */
.stMarkdown p { color: var(--ink) !important; font-size: 0.9rem !important; line-height: 1.75 !important; }
.stMarkdown li { color: var(--ink2) !important; font-size: 0.88rem !important; line-height: 1.7 !important; }
.stMarkdown strong { color: var(--ink) !important; font-weight: 600 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { font-family: var(--mono) !important; font-size: 0.62rem !important; letter-spacing: 0.16em !important; text-transform: uppercase !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; padding-bottom: 0.4rem !important; margin-top: 1.4rem !important; }

/* Metrics */
[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; padding: 1rem 1.2rem !important; transition: border-color 0.2s, box-shadow 0.2s !important; }
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
.streamlit-expanderHeader { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; color: var(--ink2) !important; font-family: var(--mono) !important; font-size: 0.62rem !important; font-weight: 500 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; padding: 0.85rem 1.1rem !important; transition: background 0.2s !important; }
.streamlit-expanderHeader:hover { background: var(--surface3) !important; }
.streamlit-expanderContent { background: var(--surface) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 2px 2px !important; padding: 1.2rem !important; }

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; transition: border-color 0.2s, box-shadow 0.15s !important; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--border2) !important; box-shadow: 0 2px 12px rgba(13,27,42,0.08) !important; }

/* Selectbox */
.stSelectbox > div > div { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 2px !important; color: var(--ink) !important; font-family: var(--sans) !important; font-size: 0.88rem !important; }

/* Slider */
.stSlider > div > div > div > div { background: var(--cyan) !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { font-family: var(--mono) !important; font-size: 0.6rem !important; color: var(--muted) !important; }

/* Status */
.stSuccess { background: rgba(26,122,74,0.07) !important; border: 1px solid rgba(26,122,74,0.25) !important; border-radius: 2px !important; color: #1a7a4a !important; font-family: var(--mono) !important; font-size: 0.72rem !important; }
.stWarning { background: rgba(192,128,0,0.07) !important; border: 1px solid rgba(192,128,0,0.25) !important; border-radius: 2px !important; color: var(--gold) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; }
.stError { background: rgba(192,57,43,0.07) !important; border: 1px solid rgba(192,57,43,0.25) !important; border-radius: 2px !important; color: var(--red) !important; }

/* Caption */
.stCaption, small { font-family: var(--mono) !important; font-size: 0.62rem !important; font-weight: 400 !important; color: var(--muted) !important; letter-spacing: 0.04em !important; }

/* HR / Spinner */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* Checkbox */
.stCheckbox > label > span { color: var(--ink2) !important; font-size: 0.82rem !important; }

/* Footer */
footer, [data-testid="stFooter"] { background: transparent !important; color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)



# Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state: st.session_state.results = []
if "selected_ids" not in st.session_state: st.session_state.selected_ids = set()
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    provider = st.selectbox("LLM Provider", ["openai", "ollama"])

    # Ollama model selector
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

    # Use project_paths constants
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
    html, body { margin:0; padding:0; background:#faf7f2 !important; background-color:#faf7f2 !important; }
    .sb-footer { font-family:'DM Mono',monospace; font-size:0.62rem; color:#4a6377; line-height:2.2; padding:0.5rem 0 0.25rem; }
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

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
html, body { background:#0d1b2a !important; background-color:#0d1b2a !important; }
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
.logo-wrap img { height:44px; width:auto; display:block; }

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
        <div class="logo-wrap">
            <img src="data:image/png;base64,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" alt="Turku AMK" />
        </div>
    </div>
</div>
""", height=235)

# ────────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────────
tab_search, tab_ask = st.tabs(["◈  Search & Select Papers", "◉  Ask the Assistant"])


def section_header(tag, desc):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;font-weight:500;
        letter-spacing:0.2em;text-transform:uppercase;color:#4a6377;margin-bottom:0.5rem;
        display:flex;align-items:center;gap:0.5rem;">
            <span style="display:inline-block;width:16px;height:1px;background:rgba(13,27,42,0.2);"></span>
            {tag}
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#4a6377;line-height:1.75;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


# TAB 1 — Search ────────────────────────────────────────────────────────────────
with tab_search:
    section_header(
        "Semantic Scholar API",
        "Search the academic literature. Select papers to include in the pre-screened retrieval corpus. "
        "Selected papers form the governance-filtered evidence base used during retrieval and generation."
    )

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "",
            placeholder="e.g. human oversight automation bias AI clinical decision support",
            label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.warning("Enter a search topic first.")
        else:
            with st.spinner("Querying Semantic Scholar..."):
                results = fetch_rehabilitation_papers(query, result_limit=k_articles)
                st.session_state.results = results
                st.session_state.selected_ids = set()

    if st.session_state.results:
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:0.6rem;
        font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.1em;
        text-transform:uppercase;color:#4a6377;padding:0.4rem 0.9rem;
        background:#faf7f2;border:1px solid rgba(13,27,42,0.12);border-radius:2px;margin:0.8rem 0;">
            <span style="color:#0077aa;font-size:0.9rem;font-weight:500;">{len(st.session_state.results)}</span>
            papers retrieved
        </div>
        """, unsafe_allow_html=True)

        for i, paper in enumerate(st.session_state.results):
            paper_id   = paper.get("paperId", str(i))
            title      = paper.get("title", "No title")
            abstract   = paper.get("abstract", "No abstract available.")
            year       = paper.get("year", "N/A")
            url        = paper.get("url", "")
            authors    = paper.get("authors", [])
            # Show up to 4 authors
            
            author_str = ", ".join([a.get("name", "") for a in authors[:4]])
            if len(authors) > 4: author_str += " et al."
            selected = paper_id in st.session_state.selected_ids

            with st.container(border=True):
                col_chk, col_content = st.columns([1, 18])
                with col_chk:
                    if st.checkbox("", value=selected, key=f"chk_{paper_id}"):
                        st.session_state.selected_ids.add(paper_id)
                    else:
                        st.session_state.selected_ids.discard(paper_id)
                with col_content:
                    st.markdown(f"**{title}**")
                    st.caption(f"{year}  ·  {author_str if author_str else 'Unknown authors'}")
                    with st.expander("Abstract"):
                        st.write(abstract)
                    if url:
                        st.link_button("↗ Open paper", url)

        st.markdown("---")
        n_selected = len(st.session_state.selected_ids)
        col_save, col_clear, col_count = st.columns([2, 2, 4])
        with col_save:
            if st.button("Add to corpus", use_container_width=True):
                selected_papers = [
                    p for p in st.session_state.results
                    if p.get("paperId", str(id(p))) in st.session_state.selected_ids
                ]
                if not selected_papers:
                    st.warning("Select at least one paper first.")
                else:
                    # Use project_paths helper
                    ensure_parent_dir(FILTERED_PAPERS_PATH)
                    with open(FILTERED_PAPERS_PATH, "w", encoding="utf-8") as f:
                        json.dump(selected_papers, f, indent=2)
                    st.success(f"✓  {len(selected_papers)} papers added to corpus.")
        with col_clear:
            if st.button("Clear selection", use_container_width=True):
                st.session_state.selected_ids = set()
                st.rerun()
        with col_count:
            st.caption(f"{n_selected} paper{'s' if n_selected != 1 else ''} selected")
    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
            letter-spacing:0.2em;color:rgba(13,27,42,0.2);margin-bottom:0.6rem;">NO RESULTS</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.85rem;color:#4a6377;">
            Enter a research topic above and click Search</div>
        </div>
        """, unsafe_allow_html=True)


# TAB 2 — Chat ────────────────────────────────────────────────────────────
with tab_ask:
    section_header(
        "Evidence Synthesis Engine",
        "The assistant retrieves evidence exclusively from the pre-screened corpus and synthesizes a structured response. "
        "It does not use external knowledge and does not provide clinical recommendations. "
        "All responses are grounded in retrieved literature with traceable citations."
    )

    if not os.path.exists(RAG_STORE_DIR):
        st.warning("No vector index found. Build the index before asking questions.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a research question about the pre-screened corpus...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence · Generating response..."):
                try:
                    # Pass ollama_model when applicable
                    result = generate_rag_answer(
                        user_input,
                        provider=provider,
                        k=k_articles,
                        answer_template="structured",
                        output_mode="text",
                        **({"model": ollama_model} if provider == "ollama" and ollama_model else {}),
                    )
                    answer               = result.get("answer", "No answer generated.")
                    sources              = result.get("sources", [])
                    insufficient_evidence = result.get("insufficient_evidence", False)

                    # Show warning for insufficient evidence
                    if insufficient_evidence:
                        st.warning(answer)
                    else:
                        st.markdown(answer)

                    if sources:
                        with st.expander(f"◈  Evidence Sources  ·  {len(sources)} papers retrieved"):
                            components.html("""
                            <style>
                            @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
                            html, body { margin:0; padding:0; background:#faf7f2 !important; }
                            .disc {
                                background:rgba(0,119,170,0.05); border:1px solid rgba(0,119,170,0.2);
                                border-left:3px solid #0077aa; border-radius:2px;
                                padding:0.9rem 1.1rem; font-family:'DM Mono',monospace;
                                font-size:0.67rem; color:#4a6377; line-height:1.85; letter-spacing:0.02em;
                            }
                            .disc strong { color:#0d1b2a; font-weight:500; }
                            .disc em { font-style:italic; color:#0077aa; }
                            </style>
                            <div class="disc">
                                <strong>⚠ Note on retrieval scores:</strong>
                                Scores are retrieval relevance indicators — they reflect alignment between the
                                query and retrieved evidence. They are <em>not</em> statistical confidence measures.
                                Final evaluation of the evidence remains with the human researcher.
                            </div>
                            """, height=110)

                            for s in sources:
                                paper_num        = s.get("paper_number", "")
                                title_s          = s.get("title", "Unknown")
                                year_s           = s.get("year", "")
                                url_s            = s.get("url", "")
                                pid              = s.get("paperId", "")
                                src              = s.get("text_source", "")
                                # structured retrieval_scores + final_score
                                final_score      = s.get("final_score")
                                retrieval_scores = s.get("retrieval_scores") or {}
                                sem              = retrieval_scores.get("semantic_match") or s.get("semantic_match")
                                cross            = retrieval_scores.get("cross_encoder_score")

                                components.html(f"""
                                <style>
                                @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
                                html, body {{ margin:0; padding:0; background:#faf7f2 !important; }}
                                .paper-header {{ padding:1rem 0 0.6rem; border-top:1px solid rgba(13,27,42,0.1); margin-top:0.5rem; }}
                                .paper-num {{ font-family:'DM Mono',monospace; font-size:0.57rem; font-weight:500; letter-spacing:0.2em; text-transform:uppercase; color:#4a6377; margin-bottom:0.35rem; }}
                                .paper-title {{ font-family:'Inter',sans-serif; font-size:0.92rem; font-weight:600; color:#0d1b2a; line-height:1.4; margin-bottom:0.3rem; }}
                                .paper-meta {{ font-family:'DM Mono',monospace; font-size:0.6rem; color:#4a6377; letter-spacing:0.04em; }}
                                </style>
                                <div class="paper-header">
                                    <div class="paper-num">Paper {paper_num}</div>
                                    <div class="paper-title">{title_s}</div>
                                    <div class="paper-meta">{year_s} &nbsp;·&nbsp; {src}</div>
                                </div>
                                """, height=100)

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if sem is not None:
                                        st.metric("Semantic Match", format_score(sem),
                                            help="Cosine similarity between query and paper embeddings (0–1). Measures conceptual alignment.")
                                with col2:
                                    if cross is not None:
                                        st.metric("Cross-Encoder", format_score(cross),
                                            help="Cross-encoder reranking score (0–1). Second-stage validation of how directly the paper addresses the query.")
                                with col3:
                                    if final_score is not None:
                                        st.metric("Final Score", format_score(final_score),
                                            help="Composite final ranking score used for retrieval ordering.")

                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    if pid: st.caption(f"ID: {pid}")
                                with c2:
                                    if url_s: st.link_button("↗ Open", url_s)

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Generation error: {e}")

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

#Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:0.7rem 0;margin-top:0.5rem;
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