import streamlit as st
import json
import os

st.write("DEBUG: App started")

#from ingestion.scripts.updated_scraper import fetch_rehabilitation_papers
#from llm.rag_generator import generate_rag_answer

try:
    from ingestion.scripts.updated_scraper import fetch_rehabilitation_papers
    from llm.rag_generator import generate_rag_answer
    st.write("Imports successful")
except Exception as e:
    st.error(f"Import error: {e}")

# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="HybReDe Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# -------------------------
# Session state
# -------------------------

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Title
# -------------------------

st.title("🔬 HybReDe AI Research Assistant")

st.write(
"""
Search for research papers, select useful ones, and build a RAG knowledge base.
"""
)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Settings")

model_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["openai", "ollama"]
)

k_articles = st.sidebar.slider(
    "Number of papers retrieved",
    1,
    10,
    5
)

# -------------------------
# Search
# -------------------------

st.subheader("Search Research Papers")

query = st.text_input(
    "Enter research topic",
    placeholder="Effects of intermittent fasting on metabolism"
)

if st.button("🔎 Search Papers"):

    with st.spinner("Searching Semantic Scholar..."):

        results = fetch_rehabilitation_papers(query, result_limit=k_articles)

        st.session_state.search_results = results
        st.session_state.selected_papers = []

# -------------------------
# Results
# -------------------------

st.subheader("Search Results")

if st.session_state.search_results:

    for i, paper in enumerate(st.session_state.search_results):

        title = paper.get("title", "No title")
        abstract = paper.get("abstract", "No abstract")
        year = paper.get("year", "N/A")

        with st.container(border=True):

            col1, col2 = st.columns([1,9])

            with col1:
                selected = st.checkbox("", key=f"paper_{i}")

                if selected:
                    if paper not in st.session_state.selected_papers:
                        st.session_state.selected_papers.append(paper)

            with col2:

                st.markdown(f"**{title}**")
                st.caption(f"Year: {year}")

                with st.expander("Abstract"):
                    st.write(abstract)

# -------------------------
# Selected papers
# -------------------------

st.subheader("Selected Papers")

st.write(f"{len(st.session_state.selected_papers)} papers selected.")

if st.button("📚 Add Selected to RAG"):

    if len(st.session_state.selected_papers) == 0:
        st.warning("Select papers first.")
    else:

        # save selected papers to filtered file
        path = "data/processed/filtered_papers.json"

        os.makedirs("data/processed", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.selected_papers, f, indent=2)

        st.success("Selected papers saved to filtered_papers.json")

# -------------------------
# AI Chat
# -------------------------

st.subheader("AI Research Assistant")

user_question = st.chat_input("Ask a question based on indexed papers...")

if user_question:

    st.chat_message("user").write(user_question)

    with st.spinner("Generating answer..."):

        result = generate_rag_answer(
            user_question,
            provider=model_provider,
            k=5
        )

        answer = result["answer"]
        sources = result["sources"]

    st.chat_message("assistant").write(answer)

    with st.expander("Sources"):

        for s in sources:
            st.markdown(f"**{s['title']}** ({s['year']})")
            st.write(s["url"])