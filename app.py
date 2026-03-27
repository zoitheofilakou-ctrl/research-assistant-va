import streamlit as st
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HybReDe — AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ── Custom styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #06101a;
        color: #d6eeff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #091520;
        border-right: 1px solid rgba(77,216,255,0.1);
    }

    section[data-testid="stSidebar"] * {
        color: #9db5c9 !important;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        letter-spacing: -0.02em;
    }

    h1 { color: #4dd8ff !important; font-size: 1.6rem !important; }
    h2 { color: #d6eeff !important; font-size: 1.1rem !important; }
    h3 { color: #d6eeff !important; font-size: 0.95rem !important; }

    /* Text input */
    .stTextInput > div > div > input,
    .stChatInputContainer textarea {
        background-color: #0d1e2e !important;
        border: 1px solid rgba(77,216,255,0.2) !important;
        color: #d6eeff !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        border-radius: 6px !important;
    }

    .stTextInput > div > div > input:focus,
    .stChatInputContainer textarea:focus {
        border-color: rgba(77,216,255,0.5) !important;
        box-shadow: 0 0 0 2px rgba(77,216,255,0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: transparent !important;
        border: 1px solid rgba(77,216,255,0.3) !important;
        color: #4dd8ff !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em !important;
        border-radius: 4px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: rgba(77,216,255,0.08) !important;
        border-color: rgba(77,216,255,0.6) !important;
    }

    /* Container / card */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #091520 !important;
        border: 1px solid rgba(77,216,255,0.12) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0d1e2e !important;
        color: #5a8aaa !important;
        font-size: 0.82rem !important;
        border-radius: 4px !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #0d1e2e !important;
        border: 1px solid rgba(77,216,255,0.2) !important;
        color: #d6eeff !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #4dd8ff !important;
    }

    /* Caption / small text */
    .stCaption, small, caption {
        color: #5a8aaa !important;
        font-size: 0.78rem !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #0d1e2e !important;
        border: 1px solid rgba(77,216,255,0.08) !important;
        border-radius: 8px !important;
    }

    /* Success / warning / error */
    .stSuccess { background-color: rgba(46,232,184,0.08) !important; border-color: rgba(46,232,184,0.3) !important; }
    .stWarning { background-color: rgba(240,180,41,0.08) !important; }
    .stError   { background-color: rgba(255,107,107,0.08) !important; }

    /* Divider */
    hr { border-color: rgba(77,216,255,0.1) !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #4dd8ff !important; }

    /* Link buttons */
    .stLinkButton > a {
        background-color: transparent !important;
        border: 1px solid rgba(77,216,255,0.2) !important;
        color: #4dd8ff !important;
        font-size: 0.75rem !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []

if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    provider = st.selectbox("LLM Provider", ["openai", "ollama"])
    ollama_model = None
    if provider == "ollama":
        ollama_model = st.text_input(
            "Ollama model",
            value=os.getenv("OLLAMA_MODEL", "phi"),
            help="Example: qwen3:14b",
        )
    k_articles = st.slider("Papers to retrieve", 1, 10, 5)

    st.markdown("---")
    st.markdown("### Pipeline status")

    rag_store_ok = os.path.exists(RAG_STORE_DIR)
    filtered_ok  = os.path.exists(FILTERED_PAPERS_PATH)

    if rag_store_ok:
        st.success("Vector index ready")
    else:
        st.warning("No vector index — run indexing first")

    if filtered_ok:
        st.success("Filtered corpus available")
    else:
        st.warning("No filtered papers")

    st.markdown("---")
    st.caption("HybReDe · Turku UAS ICT · 2026")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("HybReDe — AI Research Assistant")
st.caption("Governance-aware retrieval-augmented pipeline for healthcare literature pre-screening")
st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_search, tab_ask = st.tabs(["🔎 Search & Select Papers", "💬 Ask the Assistant"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Search
# ════════════════════════════════════════════════════════════════════════════
with tab_search:

    st.subheader("Search Research Papers")
    st.caption("Search Semantic Scholar to discover papers. Select relevant ones to add to the RAG corpus.")

    col_input, col_btn = st.columns([5, 1])

    with col_input:
        query = st.text_input(
            "Research topic",
            placeholder="e.g. evidence-based practice barriers healthcare professionals",
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

    # Results
    if st.session_state.results:
        st.markdown(f"**{len(st.session_state.results)} papers found**")
        st.markdown("---")

        for i, paper in enumerate(st.session_state.results):
            paper_id   = paper.get("paperId", str(i))
            title      = paper.get("title", "No title")
            abstract   = paper.get("abstract", "No abstract available.")
            year       = paper.get("year", "N/A")
            url        = paper.get("url", "")
            authors    = paper.get("authors", [])
            author_str = ", ".join([a.get("name", "") for a in authors[:4]])
            if len(authors) > 4:
                author_str += " et al."

            selected = paper_id in st.session_state.selected_ids

            with st.container(border=True):
                col_chk, col_content = st.columns([1, 14])

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
                        st.link_button("Open paper", url)

        st.markdown("---")

        # Selection controls
        n_selected = len(st.session_state.selected_ids)
        col_save, col_clear, col_count = st.columns([2, 2, 4])

        with col_save:
            if st.button("Add to RAG corpus", use_container_width=True):
                selected_papers = [
                    p for p in st.session_state.results
                    if p.get("paperId", str(id(p))) in st.session_state.selected_ids
                ]
                if not selected_papers:
                    st.warning("Select at least one paper first.")
                else:
                    ensure_parent_dir(FILTERED_PAPERS_PATH)
                    with open(FILTERED_PAPERS_PATH, "w", encoding="utf-8") as f:
                        json.dump(selected_papers, f, indent=2)
                    st.success(f"{len(selected_papers)} papers saved to corpus.")

        with col_clear:
            if st.button("Clear selection", use_container_width=True):
                st.session_state.selected_ids = set()
                st.rerun()

        with col_count:
            st.caption(f"{n_selected} paper{'s' if n_selected != 1 else ''} selected")

    else:
        st.info("No results yet. Enter a topic and click Search.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat
# ════════════════════════════════════════════════════════════════════════════
with tab_ask:

    st.subheader("Ask the Research Assistant")
    st.caption(
        "The assistant retrieves evidence from the pre-screened corpus and generates a grounded response. "
        "It does not use external knowledge and does not provide clinical recommendations."
    )

    if not os.path.exists(RAG_STORE_DIR):
        st.warning("No vector index found. Build the index before asking questions.")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about the research corpus...")

    if user_input:

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating response..."):
                try:
                    result = generate_rag_answer(
                        user_input,
                        provider=provider,
                        k=k_articles,
                        answer_template="structured",
                        output_mode="text",
                        **({"model": ollama_model} if provider == "ollama" and ollama_model else {}),
                    )

                    answer  = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])
                    insufficient_evidence = result.get("insufficient_evidence", False)

                    if insufficient_evidence:
                        st.warning(answer)
                    else:
                        st.write(answer)

                    if sources:
                        with st.expander(f"Sources ({len(sources)} papers)"):
                            for s in sources:
                                title_s = s.get("title", "Unknown")
                                year_s  = s.get("year", "")
                                url_s   = s.get("url", "")
                                pid     = s.get("paperId", "")
                                final_score = s.get("final_score")
                                retrieval_scores = s.get("retrieval_scores") or {}
                                st.markdown(f"**{title_s}** ({year_s})")
                                meta_bits = []
                                if pid:
                                    meta_bits.append(f"paperId: `{pid}`")
                                formatted_final_score = format_score(final_score)
                                if formatted_final_score is not None:
                                    meta_bits.append(f"final_score: `{formatted_final_score}`")
                                if meta_bits:
                                    st.caption(" | ".join(meta_bits))
                                score_labels = [
                                    ("cross_encoder_score", "cross"),
                                ]
                                score_bits = []
                                for score_key, score_label in score_labels:
                                    formatted_score = format_score(retrieval_scores.get(score_key))
                                    if formatted_score is not None:
                                        score_bits.append(f"{score_label}: `{formatted_score}`")
                                if score_bits:
                                    st.caption("retrieval: " + " | ".join(score_bits))
                                if url_s:
                                    st.link_button("Open paper", url_s)
                                st.markdown("---")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Generation error: {e}")

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "HybReDe · Evidence-grounded · No clinical recommendations · Human oversight preserved · Turku UAS 2026"
)
