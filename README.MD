# Research Assistant VA

Research Assistant VA is a local RAG pipeline for healthcare-related literature.
It supports:
- metadata acquisition
- LLM-based screening
- PDF harvesting and PDF-to-text extraction
- vector indexing and retrieval
- optional answer generation with OpenAI or Ollama

## Current Standard Paths
All scripts are now aligned to this folder layout:

```text
data/
  hybrede_metadata_v3.json
  harvested_pdfs/                # created by PDFscraper.py
  v3_full_text/                  # created/used by pdf_to_text.py and Retrieval/retrieval.py
  processed/
    filtered_papers.json
    screening_log.json
    audit_log.json
rag_store/                       # created by Retrieval/retrieval.py
```

## Requirements
- Python 3.10+
- `pip`

Install dependencies:

```bash
pip install openai requests python-dotenv pdfplumber chromadb sentence-transformers streamlit
```

Optional (for local LLM mode):
- Ollama running locally at `http://localhost:11434`

## Environment Variables
Create a `.env` file in the project root (or use your shell environment):

```env
OPENAI_API_KEY=your_api_key_here
```

Notes:
- `screening/llm_screening.py` needs `OPENAI_API_KEY` for OpenAI API calls.
- `llm/interface.py` also uses `OPENAI_API_KEY` for OpenAI responses.
- `ingestion/scripts/updated_scraper.py` currently also reads `OPENAI_API_KEY` for Semantic Scholar requests (name is legacy in code).

## Recommended Run Order (from repo root)
1. Acquire or refresh metadata (`v3`):

```bash
python ingestion/scripts/updated_scraper.py
```

2. Run screening (writes `data/processed/` files):

```bash
python screening/llm_screening.py
```

3. Download open-access PDFs:

```bash
python PDFscraper.py
```

4. Convert PDFs to text files (`paperId.txt` in `data/v3_full_text`):

```bash
python pdf_to_text.py
```

5. Build vector index:

```bash
python Retrieval/retrieval.py index
```

## Retrieval Usage
Build index with explicit files:

```bash
python Retrieval/retrieval.py index --metadata data/hybrede_metadata_v3.json --filtered data/processed/filtered_papers.json
```

Run query:

```bash
python Retrieval/retrieval.py query "evidence-based practice instruments" --k 5
```

Get suggestions:

```bash
python Retrieval/retrieval.py suggest --n 5
```

## RAG Answer Generator
Generate answers from indexed evidence:

```bash
python llm/rag_generator.py "What are key trends in clinical decision support?" openai --k 5 --template structured
```

JSON output mode:

```bash
python llm/rag_generator.py "What are key trends in clinical decision support?" openai --output json
```

Using Ollama:

```bash
python llm/rag_generator.py "What are key trends in clinical decision support?" ollama
```

## Using the User interface
Open the UI

```bash
python -m streamlit run app.py
```
UI Available in browset at http://localhost:8501


## Important Behavior Notes
- Retrieval indexes only papers allowed by `data/processed/filtered_papers.json`.
- Retrieval prefers full text from `data/v3_full_text/{paperId}.txt`.
- If full text is missing, retrieval falls back to abstract from metadata.
- `cmd_query` output includes `text_source` for traceability (`fulltext` or `abstract`).

## Troubleshooting
- `ModuleNotFoundError: chromadb`
  - Install dependencies: `pip install chromadb sentence-transformers`
- `No index found. Run: python Retrieval/retrieval.py index`
  - Build the index first.
- `PDF folder ... data/harvested_pdfs not found`
  - Run `python PDFscraper.py` before `python pdf_to_text.py`.
- `OPENAI_API_KEY` errors
  - Ensure `.env` exists and the key is valid.

## Repository Scripts (Active)
- `ingestion/scripts/updated_scraper.py` - metadata acquisition (`v3`)
- `screening/llm_screening.py` - LLM screening and audit logs
- `PDFscraper.py` - PDF download
- `pdf_to_text.py` - PDF-to-text conversion by `paperId`
- `Retrieval/retrieval.py` - index/query/suggest
- `llm/interface.py` - LLM provider abstractions
- `llm/rag_generator.py` - end-to-end retrieval + answer generation
- `app.py` - User interface
