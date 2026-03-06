#!/usr/bin/env python3
"""
retrieval.py - simple retrieval script for RAG.

What it does:
1) index  : takes paperId from filtered JSON, retrieves abstract+url from metadata JSON,
            splits text into chunks and stores them in ChromaDB with sentence-transformers embeddings.
2) query  : given a user query, returns top-k most similar chunks (with title/url/score/text).
3) suggest: returns N papers (e.g., most recent by year) for greeting links.

Installation:
pip install chromadb sentence-transformers

Usage:
python retrieval.py index
python retrieval.py index --metadata hybrede_metadata_v3.json --filtered filtered_papers_v3_test.json
python retrieval.py query "evidence-based practice instruments" --k 5
python retrieval.py suggest --n 5
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None


# Project-root-aware paths keep the script stable regardless of cwd.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------- Settings ----------
DEFAULT_FILTERED_FILE = os.path.join(BASE_DIR, "filtered_papers.json")
DEFAULT_METADATA_FILE = os.path.join(BASE_DIR, "data", "hybrede_metadata_v3.json")
FULLTEXT_DIR = os.path.join(BASE_DIR, "data", "v3_full_text")

# Directory where Chroma stores the index
CHROMA_DIR = os.path.join(BASE_DIR, "rag_store")
COLLECTION_NAME = "hybrede"

# Local embedding model (free, CPU-friendly)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings for full text / fallback abstracts
CHUNK_WORDS = 1000
CHUNK_OVERLAP = 150


def require_retrieval_dependencies():
    missing = []
    if chromadb is None:
        missing.append("chromadb")
    if SentenceTransformer is None:
        missing.append("sentence-transformers")
    if missing:
        pkg_hint = " ".join(missing)
        raise ModuleNotFoundError(
            f"Missing dependencies: {', '.join(missing)}. Install with: pip install {pkg_hint}"
        )


# --------- Utilities ----------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into chunks by words.
    Abstracts often produce only one chunk - that is normal.
    """
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_words - max(0, overlap))
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def build_metadata_index(metadata_records: List[dict]) -> Dict[str, dict]:
    """
    Builds a dictionary paperId -> record for fast abstract/url lookup.
    """
    idx = {}
    for rec in metadata_records:
        pid = rec.get("paperId")
        if pid:
            idx[pid] = rec
    return idx


def get_abstract_text(meta_record: dict) -> str:
    abstract = meta_record.get("abstract") or ""
    abstract = abstract.strip() if isinstance(abstract, str) else ""
    return abstract or "No abstract provided"


def load_full_text(paper_id: str, fulltext_dir: str = FULLTEXT_DIR) -> str:
    if not paper_id:
        return ""

    fulltext_path = os.path.join(fulltext_dir, f"{paper_id}.txt")
    if not os.path.exists(fulltext_path):
        return ""

    with open(fulltext_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_text_for_paper(meta_record: dict, fulltext_dir: str = FULLTEXT_DIR) -> Tuple[str, str, int, str]:
    """
    Returns (title, text_for_index, year, text_source).
    Prefer full text by paperId; fall back to abstract when missing.
    """
    title = (meta_record.get("title") or "").strip()
    paper_id = (meta_record.get("paperId") or "").strip()
    year = meta_record.get("year") or 0

    full_text = load_full_text(paper_id, fulltext_dir=fulltext_dir)
    if full_text:
        return title, full_text, year, "fulltext"

    return title, get_abstract_text(meta_record), year, "abstract"


def get_best_url(meta_record: dict) -> str:
    """
    Returns the most useful URL (PDF if available, otherwise Semantic Scholar page).
    """
    oap = meta_record.get("openAccessPdf") or {}
    pdf_url = (oap.get("url") or "").strip() if isinstance(oap, dict) else ""
    if pdf_url:
        return pdf_url

    url = (meta_record.get("url") or "").strip()
    return url


def ensure_files_exist(*paths: str):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")


# --------- Chroma helpers ----------
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def collection_count(collection) -> int:
    try:
        return collection.count()
    except Exception:
        return 0


# --------- Commands ----------
def cmd_index(metadata_file: str, filtered_file: str):
    require_retrieval_dependencies()
    ensure_files_exist(filtered_file, metadata_file)

    filtered = load_json(filtered_file)
    metadata = load_json(metadata_file)

    meta_index = build_metadata_index(metadata)

    # paperIds allowed by screening
    allowed_ids = [p.get("paperId") for p in filtered if p.get("paperId")]
    allowed_set = set(allowed_ids)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Recreate collection to avoid duplicates
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    docs: List[str] = []
    ids: List[str] = []
    metas: List[dict] = []

    missing_in_metadata = 0
    total_chunks = 0
    fulltext_papers = 0
    abstract_fallback_papers = 0

    for pid in allowed_ids:
        meta_rec = meta_index.get(pid)
        if not meta_rec:
            missing_in_metadata += 1
            continue

        title, text_for_index, year, text_source = get_text_for_paper(meta_rec)
        url = get_best_url(meta_rec)

        if text_source == "fulltext":
            fulltext_papers += 1
        else:
            abstract_fallback_papers += 1

        chunks = chunk_text(text_for_index, CHUNK_WORDS, CHUNK_OVERLAP)
        if not chunks:
            continue

        for i, ch in enumerate(chunks):
            chunk_id = f"{pid}:{i:04d}"
            docs.append(ch)
            ids.append(chunk_id)
            metas.append({
                "paperId": pid,
                "title": title,
                "url": url,
                "year": year,
                "chunk_index": i,
                "text_source": text_source
            })
            total_chunks += 1

    if total_chunks == 0:
        print("Nothing to index: no full text, no abstracts, or filtered file is empty")
        return

    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    print("=== INDEX DONE ===")
    print(f"Metadata input: {metadata_file}")
    print(f"Filtered input: {filtered_file}")
    print(f"Allowed papers (from filtered): {len(allowed_set)}")
    print(f"Missing paperId in metadata: {missing_in_metadata}")
    print(f"Papers indexed from full text: {fulltext_papers}")
    print(f"Papers indexed from abstract fallback: {abstract_fallback_papers}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Chroma collection size: {collection_count(collection)}")
    print(f"Stored at: {CHROMA_DIR}/ (collection: {COLLECTION_NAME})")


def cmd_query(user_query: str, k: int):
    require_retrieval_dependencies()

    if not user_query.strip():
        print("Empty query.")
        return

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError("No index found. Run: python retrieval.py index")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = get_chroma_collection()

    q_emb = model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True).tolist()[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = {"query": user_query, "results": []}
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        score = 1.0 / (1.0 + float(dist)) if dist is not None else None
        out["results"].append({
            "chunk_id": cid,
            "score": score,
            "paperId": meta.get("paperId"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "year": meta.get("year"),
            "text": doc
        })

    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_suggest(n: int, metadata_file: str, filtered_file: str):
    """
    Simple greeting suggestion:
    Takes unique filtered papers,
    sorts by year descending,
    returns top-N.
    """
    ensure_files_exist(filtered_file, metadata_file)

    filtered = load_json(filtered_file)
    metadata = load_json(metadata_file)
    meta_index = build_metadata_index(metadata)

    allowed_ids = [p.get("paperId") for p in filtered if p.get("paperId")]

    suggestions = []
    for pid in allowed_ids:
        rec = meta_index.get(pid)
        if not rec:
            continue
        title = (rec.get("title") or "").strip()
        year = rec.get("year") or 0
        url = get_best_url(rec)
        suggestions.append({
            "paperId": pid,
            "title": title,
            "year": year,
            "url": url
        })

    uniq = {s["paperId"]: s for s in suggestions}
    suggestions = list(uniq.values())

    suggestions.sort(key=lambda x: int(x.get("year") or 0), reverse=True)

    out = {
        "metadata_file": metadata_file,
        "filtered_file": filtered_file,
        "suggestions": suggestions[:n]
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="HybReDe RAG Retrieval Script (local)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build local vector index from filtered + metadata")
    p_index.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Path to metadata JSON")
    p_index.add_argument("--filtered", default=DEFAULT_FILTERED_FILE, help="Path to filtered JSON")

    p_query = sub.add_parser("query", help="Retrieve top-k relevant chunks for a user query")
    p_query.add_argument("text", type=str, help="User query text")
    p_query.add_argument("--k", type=int, default=5, help="Number of results (top-k)")

    p_suggest = sub.add_parser("suggest", help="Return N suggested papers for greeting")
    p_suggest.add_argument("--n", type=int, default=5, help="How many papers to suggest")
    p_suggest.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Path to metadata JSON")
    p_suggest.add_argument("--filtered", default=DEFAULT_FILTERED_FILE, help="Path to filtered JSON")

    args = parser.parse_args()

    if args.cmd == "index":
        cmd_index(args.metadata, args.filtered)
    elif args.cmd == "query":
        cmd_query(args.text, args.k)
    elif args.cmd == "suggest":
        cmd_suggest(args.n, args.metadata, args.filtered)


if __name__ == "__main__":
    main()
