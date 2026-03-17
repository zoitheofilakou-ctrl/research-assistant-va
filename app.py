import streamlit as st
import json
import os
from dotenv import load_dotenv

st.write("DEBUG: App started")

st.write("DEBUG: Loading .env")
# Load environment variables
load_dotenv()

st.write("DEBUG: Importing")
# Backend imports
#from ingestion.scripts.updated_scraper import fetch_rehabilitation_papers
#from llm.rag_generator import generate_rag_answer

# Try/Except import version for debugging and error handling. Without try/except commented above ^^^
try:
    from ingestion.scripts.updated_scraper import fetch_rehabilitation_papers
    from llm.rag_generator import generate_rag_answer
    st.write("Imports successful")
except Exception as e:
    st.error(f"Import error: {e}")

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="HybReDe AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# -------------------------
# Session State Init
# -------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    provider = st.selectbox(
        "LLM Provider",
        ["ollama", "openai"]
    )

    k_articles = st.slider("Articles to retrieve", 1, 10, 5)

    st.markdown("---")

    st.subheader("RAG Status")

    if os.path.exists("data/processed/filtered_papers.json"):
        st.success("Filtered papers available")
    else:
        st.warning("No filtered papers yet")

    st.markdown("---")
    st.caption("HybReDe Research Assistant")

# -------------------------
# Header
# -------------------------
st.title("🔬 HybReDe AI Research Assistant")
st.caption("Search → Select → Build RAG → Ask Questions")

# -------------------------
# Search Section
# -------------------------
st.subheader("🔎 Search Research Papers")

query = st.text_input(
    "Enter your research topic",
    placeholder="Effects of fasting on metabolism"
)

if st.button("Search Papers"):

    with st.spinner("Searching Semantic Scholar..."):
        results = fetch_rehabilitation_papers(query, result_limit=k_articles)

        st.session_state.results = results
        st.session_state.selected_ids = set()

# -------------------------
# Results Section
# -------------------------
st.subheader("📄 Search Results")

if not st.session_state.results:
    st.info("No results yet. Run a search.")
else:

    for i, paper in enumerate(st.session_state.results):

        paper_id = paper.get("paperId", str(i))
        title = paper.get("title", "No title")
        abstract = paper.get("abstract", "No abstract")
        year = paper.get("year", "N/A")
        url = paper.get("url", "")
        authors = paper.get("authors", [])
        author_names = ", ".join([a.get("name", "") for a in authors[:5]])

        if len(authors) > 5:
             author_names += " et al."

        selected = paper_id in st.session_state.selected_ids

        with st.container(border=True):

            col1, col2 = st.columns([1, 10])

            # Checkbox
            with col1:
                if st.checkbox("", value=selected, key=f"chk_{paper_id}"):
                    st.session_state.selected_ids.add(paper_id)
                else:
                    st.session_state.selected_ids.discard(paper_id)

            # Paper content
            with col2:
                st.markdown(f"### {title}")
                st.caption(f"Year: {year}")
                st.caption(f"Authors: {author_names if author_names else 'Unknown'}")

                with st.expander("Abstract"):
                    st.write(abstract)

                if url:
                    st.link_button("Open Paper", url)

# -------------------------
# Selection Summary
# -------------------------
st.subheader("📚 Selected Papers")

selected_papers = [
    p for p in st.session_state.results
    if p.get("paperId", str(id(p))) in st.session_state.selected_ids
]

st.write(f"Selected: **{len(selected_papers)} papers**")

col1, col2 = st.columns(2)

# Save to RAG
with col1:
    if st.button("Add Selected to RAG"):

        if not selected_papers:
            st.warning("No papers selected")
        else:
            os.makedirs("data/processed", exist_ok=True)

            with open("data/processed/filtered_papers.json", "w", encoding="utf-8") as f:
                json.dump(selected_papers, f, indent=2)

            st.success("Saved to filtered_papers.json")

# Clear selection
with col2:
    if st.button("Clear Selection"):
        st.session_state.selected_ids = set()

# -------------------------
# RAG Debug Panel
# -------------------------
with st.expander("🧠 RAG Debug Panel"):

    st.write("### Selected Paper Titles")
    for p in selected_papers:
        st.write("-", p.get("title", "No title"))

    st.write("### Provider")
    st.write(provider)

# -------------------------
# Chat Section
# -------------------------
st.subheader("💬 Ask the Research Assistant")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question about your selected papers...")

if user_input:

    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):

            try:
                result = generate_rag_answer(
                    user_input,
                    provider=provider,
                    k=5
                )

                answer = result.get("answer", "No answer")
                sources = result.get("sources", [])

                st.write(answer)

                # Show sources
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"**{s.get('title','Unknown')}** ({s.get('year','')})")
                            if s.get("url"):
                                st.link_button("Open Source", s["url"])

                # Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("HybReDe AI Research Assistant – Functional UI Prototype")