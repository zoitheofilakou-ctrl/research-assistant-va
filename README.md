# Research Assistant VA

Research Assistant VA is a local literature-search and RAG pipeline for healthcare-related papers. The repository currently contains:

- metadata search against Semantic Scholar
- LLM-based title/abstract screening
- PDF download and PDF-to-text conversion
- hybrid retrieval (`embeddings + BM25 + reranking + paper aggregation`)
- CLI answer generation with OpenAI or Ollama
- a Streamlit UI prototype

## Current Repository Layout

```text
research-assistant-va/
  app.py
  README.md
  data/
    hybrede_metadata_v3.json
    hybrede_metadata_v4.json
    harvested_pdfs/
    processed/
      filtered_papers.json
      screening_log.json
      audit_log.json
    v3_full_text/
  data_acquisition/
    PDFscraper.py
    pdf_to_text.py
  docs/
    ai_usage_disclosure.md
  ingestion/
    scripts/
      updated_scraper.py
  llm/
    interface.py
    rag_generator.py
  Retrieval/
    retrieval.py
  screening/
    llm_screening.py
  rag_store/
    ... created after indexing
```

## Requirements

- Python 3.10+
- `pip`
- Internet access for:
  - Semantic Scholar search
  - OpenAI API calls
  - first-time Hugging Face model download
  - PDF downloads

Optional:

- Ollama, if you want local answer generation

## Install Dependencies

From the repository root:

```powershell
python -m pip install --upgrade pip
python -m pip install openai requests python-dotenv pdfplumber chromadb sentence-transformers streamlit rapidfuzz langdetect
```

Notes:

- `sentence-transformers` will also pull the transformer stack used by retrieval.
- If your environment does not already have a compatible PyTorch build, install one separately according to the official PyTorch instructions.

## Environment Variables

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your_api_key_here
```

Important:

- `screening/llm_screening.py` needs `OPENAI_API_KEY` for OpenAI.
- `llm/interface.py` uses `OPENAI_API_KEY` for the OpenAI provider.
- `ingestion/scripts/updated_scraper.py` also reads `OPENAI_API_KEY` and sends it as the Semantic Scholar API key. The variable name is legacy in code.

## Optional Model Pre-Download

Retrieval uses local Hugging Face models. They are downloaded automatically on first use, but you can cache them manually.

Embedding model:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('EMBEDDING_MODEL_CACHED')"
```

Cross-encoder reranker:

```powershell
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('CROSS_ENCODER_CACHED')"
```

Offline cache check:

```powershell
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', local_files_only=True); print('MODEL_CACHED')"
```

If the offline check fails, the model is not cached yet.

## Important Path and Version Notes

There is currently a version mismatch in the codebase:

- `ingestion/scripts/updated_scraper.py` writes `data/hybrede_metadata_v3.json`
- `screening/llm_screening.py` reads `data/hybrede_metadata_v3.json`
- `Retrieval/retrieval.py` defaults to `data/hybrede_metadata_v3.json`
- `data_acquisition/PDFscraper.py` defaults to `../data/hybrede_metadata_v4.json`
- `data_acquisition/pdf_to_text.py` defaults to `../data/hybrede_metadata_v4.json`

That means the acquisition scripts are not automatically aligned with the retrieval pipeline unless your `v4` file contains the papers you want to process.

Also note:

- `data_acquisition/PDFscraper.py` and `data_acquisition/pdf_to_text.py` use relative paths and are safest to run from inside the `data_acquisition/` directory.
- `app.py` does not rebuild the retrieval index for you. If you change `data/processed/filtered_papers.json`, you must run indexing again manually.

## Recommended End-to-End Workflow

### 1. Fetch Metadata

Run from repository root:

```powershell
python ingestion/scripts/updated_scraper.py
```

What it does:

- queries Semantic Scholar
- writes metadata to `data/hybrede_metadata_v3.json`

Current limitation:

- when run directly, `updated_scraper.py` uses a hardcoded search query from the script body
- if you need custom queries without editing code, use the Streamlit UI or import `fetch_rehabilitation_papers()` manually

### 2. Screen Papers with OpenAI

Run from repository root:

```powershell
python screening/llm_screening.py
```

What it does:

- reads `data/hybrede_metadata_v3.json`
- screens papers with OpenAI
- validates each decision with a second OpenAI call
- writes:
  - `data/processed/filtered_papers.json`
  - `data/processed/screening_log.json`
  - `data/processed/audit_log.json`

Important behavior:

- this script resumes from existing logs if they already exist
- if you want a clean screening run, delete the files in `data/processed/` first

### 3. Download PDFs

Run from inside `data_acquisition/`:

```powershell
Push-Location data_acquisition
python PDFscraper.py
Pop-Location
```

What it does:

- reads metadata from `../data/hybrede_metadata_v4.json`
- downloads open-access PDFs to `data/harvested_pdfs/`
- validates that files are real English-language PDFs

### 4. Convert PDFs to Text

Run from inside `data_acquisition/`:

```powershell
Push-Location data_acquisition
python pdf_to_text.py
Pop-Location
```

What it does:

- reads metadata from `../data/hybrede_metadata_v4.json`
- matches downloaded PDFs back to `paperId`
- writes plain text files to `data/v3_full_text/`

### 5. Build the Hybrid Retrieval Index

Run from repository root:

```powershell
python Retrieval/retrieval.py index
```

Explicit files:

```powershell
python Retrieval/retrieval.py index --metadata data/hybrede_metadata_v3.json --filtered data/processed/filtered_papers.json
```

What it does:

- reads screened papers from `data/processed/filtered_papers.json`
- prefers full text from `data/v3_full_text/{paperId}.txt`
- falls back to abstracts if full text is missing
- builds a hybrid index in `rag_store/`
- writes a lexical sidecar file at `rag_store/lexical_index.json`

### 6. Query Retrieval Directly

Run from repository root:

```powershell
python Retrieval/retrieval.py query "evidence-based practice instruments" --k 5
```

You can also change the semantic filter:

```powershell
python Retrieval/retrieval.py query "clinical decision support" --k 5 --min-score 0.45
```

Paper suggestions:

```powershell
python Retrieval/retrieval.py suggest --n 5
```

What the retrieval layer currently does:

- embedding retrieval with ChromaDB
- lexical search with BM25-style scoring
- query expansion for some domain terms
- query-type aware weighting
- title and abstract boosting
- section-aware chunk scoring
- full-text vs abstract boosting
- paper-level aggregation
- optional cross-encoder reranking
- MMR diversification

### 7. Generate a RAG Answer from CLI

OpenAI:

```powershell
python llm/rag_generator.py "What are the main themes in evidence-based practice instruments?" openai --k 5 --template structured
```

OpenAI JSON output:

```powershell
python llm/rag_generator.py "What are the main themes in evidence-based practice instruments?" openai --output json
```

Ollama:

```powershell
python llm/rag_generator.py "What are the main themes in evidence-based practice instruments?" ollama
```

### 8. Run the Streamlit App

Run from repository root:

```powershell
streamlit run app.py
```

Current app behavior:

- search papers through `fetch_rehabilitation_papers()`
- let you select papers
- save selected papers to `data/processed/filtered_papers.json`
- use `generate_rag_answer()` for chat

Important limitation:

- after changing selected papers, you still need to rebuild the retrieval index manually:

```powershell
python Retrieval/retrieval.py index
```

## Quick Start Options

### Option A: Full pipeline

1. `python ingestion/scripts/updated_scraper.py`
2. `python screening/llm_screening.py`
3. `Push-Location data_acquisition`
4. `python PDFscraper.py`
5. `python pdf_to_text.py`
6. `Pop-Location`
7. `python Retrieval/retrieval.py index`
8. `python llm/rag_generator.py "your question" openai --k 5`

### Option B: Retrieval only from existing processed files

If `data/processed/filtered_papers.json` and `data/v3_full_text/` already exist:

```powershell
python Retrieval/retrieval.py index
python Retrieval/retrieval.py query "your topic" --k 5
```

### Option C: UI workflow

1. `streamlit run app.py`
2. Search and select papers
3. Click `Add Selected to RAG`
4. In a terminal, run `python Retrieval/retrieval.py index`
5. Return to the app and ask questions

## Ollama Setup

If you want local generation:

```powershell
ollama pull phi
ollama serve
```

Then use:

```powershell
python llm/rag_generator.py "your question" ollama
```

## Outputs and Artifacts

Main files produced by the pipeline:

- `data/hybrede_metadata_v3.json` - metadata fetched from Semantic Scholar
- `data/processed/filtered_papers.json` - screened or manually selected papers
- `data/processed/screening_log.json` - screening decisions
- `data/processed/audit_log.json` - screening audit trail
- `data/harvested_pdfs/` - downloaded PDF files
- `data/v3_full_text/` - extracted full text by `paperId`
- `rag_store/` - ChromaDB index and lexical sidecar index

## Troubleshooting

### `ModuleNotFoundError: chromadb`

Install retrieval dependencies:

```powershell
python -m pip install chromadb sentence-transformers
```

### `No index found. Run: python Retrieval/retrieval.py index`

Build the retrieval index first:

```powershell
python Retrieval/retrieval.py index
```

### `Import error` in Streamlit app

Run the app from the repository root:

```powershell
streamlit run app.py
```

### PDF scripts cannot find metadata or folders

Run them from inside `data_acquisition/`:

```powershell
Push-Location data_acquisition
python PDFscraper.py
python pdf_to_text.py
Pop-Location
```

### Cross-encoder is skipped during retrieval

That usually means the reranker model is not cached yet. Download it once:

```powershell
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('DOWNLOADED')"
```

### `OPENAI_API_KEY` errors

- check that `.env` exists in the repository root
- check that the key is valid
- restart the shell after editing environment variables if needed

## Active Scripts

- `ingestion/scripts/updated_scraper.py` - metadata acquisition
- `screening/llm_screening.py` - LLM screening and audit logging
- `data_acquisition/PDFscraper.py` - PDF download and validation
- `data_acquisition/pdf_to_text.py` - PDF-to-text conversion
- `Retrieval/retrieval.py` - hybrid retrieval index/query/suggest
- `llm/interface.py` - LLM provider abstraction
- `llm/rag_generator.py` - retrieval + answer generation
- `app.py` - Streamlit UI prototype
