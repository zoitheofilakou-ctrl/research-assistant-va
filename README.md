# Research Assistant VA

Research Assistant VA is a local healthcare-literature RAG project built around the HyBreDe pipeline.

Current repository scope:
- metadata acquisition from Semantic Scholar
- LLM-based title/abstract screening
- PDF harvesting and PDF-to-text extraction
- local hybrid retrieval over screened literature
- grounded answer generation with OpenAI or Ollama
- Streamlit UI for retrieval and external search

## Current Status

Primary entrypoints:
- `app.py` is the main Streamlit UI documented by this repository.
- `Retrieval/retrieval.py` is the main indexing and retrieval CLI.
- `llm/rag_generator.py` is the main retrieval-plus-generation CLI.

Important caveats in the current implementation:
- `app.py` imports the Semantic Scholar search module at startup, so `SEMANTIC_SCHOLAR_API_KEY` is currently required even if you only want to use the chat/RAG tab.
- The external search tab in `app.py` is not connected to the HyBreDe evidence pipeline. It is for browsing only.
- `screening/llm_screening.py` is a script-style pipeline, not a reusable library module.
- The test suite exists, but some tests are currently out of sync with the latest retrieval/RAG behavior.

## Actual Project Layout

```text
data/
  hybrede_metadata_v5.json
  harvested_pdfs/
  fulltext/
  processed/
    filtered_papers.json
    screening_log.json
    audit_log.json
    run_manifests/
rag_store/
data_acquisition/
  scraper.py
  PDFscraper.py
  pdf_to_text.py
screening/
  llm_screening.py
Retrieval/
  retrieval.py
llm/
  interface.py
  rag_generator.py
app.py
appV2.py
```

Canonical paths are defined in [project_paths.py](/C:/Dev/research-assistant-va/project_paths.py).

## How The Pipeline Works

1. `data_acquisition/scraper.py`
   Writes metadata to `data/hybrede_metadata_v5.json`.

2. `screening/llm_screening.py`
   Reads metadata, screens papers with OpenAI, and writes:
   - `data/processed/filtered_papers.json`
   - `data/processed/screening_log.json`
   - `data/processed/audit_log.json`

3. `data_acquisition/PDFscraper.py`
   Reads metadata and downloads open-access PDFs into `data/harvested_pdfs/`.

4. `data_acquisition/pdf_to_text.py`
   Extracts text from harvested PDFs and writes `data/fulltext/{paperId}.txt`.

5. `Retrieval/retrieval.py index`
   Builds the local Chroma index in `rag_store/` using:
   - metadata from `data/hybrede_metadata_v5.json`
   - the screened paper list from `data/processed/filtered_papers.json`
   - full text from `data/fulltext/` when available
   - abstract fallback when full text is missing

6. `llm/rag_generator.py`
   Queries the local retrieval index and synthesizes a grounded answer with OpenAI or Ollama.

7. `app.py`
   Exposes the current UI:
   - `Ask the Assistant`: grounded answers from the indexed screened corpus
   - `External Search (Semantic Scholar)`: unscreened browsing, not used for grounded synthesis

## Installation

Python:
- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

Current requirements are:
- `chromadb`
- `langdetect`
- `openai`
- `pdfplumber`
- `python-dotenv`
- `rapidfuzz`
- `requests`
- `sentence-transformers`
- `streamlit`

## Environment Variables

Create a `.env` file in the project root.

Required for OpenAI screening/generation:

```env
OPENAI_API_KEY=your_openai_key_here
```

Required for Semantic Scholar metadata acquisition and for starting `app.py`:

```env
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

Optional for local Ollama use:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TIMEOUT_SECONDS=180
```

Note:
- `.env.example` exists, but its formatting should be cleaned up before copying directly.

## Recommended Run Order

From the repository root:

1. Acquire or refresh metadata

```bash
python data_acquisition/scraper.py
```

2. Run screening

```bash
python screening/llm_screening.py
```

3. Download PDFs

```bash
python data_acquisition/PDFscraper.py
```

4. Convert PDFs to text

```bash
python data_acquisition/pdf_to_text.py
```

5. Build the retrieval index

```bash
python Retrieval/retrieval.py index
```

6. Ask retrieval questions directly

```bash
python Retrieval/retrieval.py query "clinical decision support implementation barriers" --k 5
```

7. Generate grounded answers

OpenAI:

```bash
python llm/rag_generator.py "What barriers to AI-enabled clinical decision support are reported?" openai --k 5 --template structured
```

Ollama:

```bash
python llm/rag_generator.py "What barriers to AI-enabled clinical decision support are reported?" ollama --model qwen2.5:7b
```

8. Start the main UI

```bash
python -m streamlit run app.py
```

## Retrieval And Generation Behavior

Current retrieval behavior:
- indexing is restricted to papers listed in `data/processed/filtered_papers.json`
- retrieval prefers `data/fulltext/{paperId}.txt`
- retrieval falls back to abstracts when full text is missing
- the retriever uses hybrid scoring: embeddings, lexical match, paper aggregation, cross-encoder reranking, and MMR diversification
- the current retrieval pipeline applies a minimum cross-encoder cutoff before final selection

Current RAG behavior:
- answers are grounded only in retrieved local evidence
- no external knowledge is allowed by the prompt
- no clinical recommendations are allowed by the prompt
- citations are currently normalized to numbered references like `[1]`, `[2]`
- returned source payloads include retrieval-facing signals such as semantic similarity and cross-encoder score

## UI Notes

`app.py`:
- main documented UI
- safe framing: the external search tab is explicitly separated from grounded evidence synthesis
- shows retrieval-facing signals for returned sources

`appV2.py`:
- alternative Streamlit UI still present in the repository
- not referenced by the current README or runtime wiring
- includes direct "add to corpus" behavior
- should be treated as an alternative or legacy UI until the project decides whether to keep or remove it

## Likely Legacy Or Auxiliary Files

These files are present but are not part of the main runtime path described above:
- `appV2.py`
- `tmp_retrieval_query.json`
- `technical_documentation.docx`
- `data/processed/invalid_papers.json`

Current runtime code does not rely on those files for the main pipeline.

## Useful Commands

CLI help:

```bash
python Retrieval/retrieval.py -h
python llm/rag_generator.py -h
```

## Testing

Tests currently live in:
- `tests/test_retrieval.py`
- `tests/test_rag_generator.py`

Repository status note:
- all Python files compile successfully
- the current `unittest` suite is partially stale relative to the latest retrieval and citation behavior
