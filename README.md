# HybReDe — Hybrid Retrieval-augmented Evidence System

A governance-aware RAG pipeline for AI-assisted literature pre-screening in healthcare research.
Built as a Turku UAS ICT Capstone 2026.

---

## Project Overview

HybReDe is a 6-stage pipeline that assists researchers in pre-screening scientific literature:

1. **Metadata Acquisition** — Semantic Scholar API
2. **LLM Pre-screening** — GPT-4o-mini with inclusion/exclusion criteria
3. **Full-text Harvesting** — PDF download and text extraction
4. **Hybrid Index Construction** — ChromaDB + BM25
5. **Hybrid Retrieval** — Vector search + cross-encoder reranking
6. **Evidence-grounded Generation** — Structured RAG answers

---

## Installation
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

---

## Running the Pipeline
```bash
# 1. Collect metadata
python data_acquisition/scraper.py

# 2. Screen papers
python screening/llm_screening.py

# 3. Build index
python Retrieval/retrieval.py index

# 4. Run UI
streamlit run app.py
```

---

## Important Notice

This system performs automated pre-screening only.
Final evaluation of the literature remains under human control.
This system does not perform clinical decision-making.