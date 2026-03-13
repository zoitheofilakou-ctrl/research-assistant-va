#!/usr/bin/env python3
"""
retrieval.py - simple retrieval script for RAG.

What it does:
1) index  : takes paperIds from filtered JSON, retrieves metadata and text,
            prefers full text by paperId from FULLTEXT_DIR (fallback to abstract),
            splits text into chunks and stores them in ChromaDB with sentence-transformers embeddings.
2) query  : given a user query, returns top-k most similar chunks (with title/url/score/text).
3) suggest: returns N papers (e.g., most recent by year) for greeting links.

Installation:
pip install chromadb sentence-transformers

Usage:
python Retrieval/retrieval.py index
python Retrieval/retrieval.py index --metadata data/hybrede_metadata_v3.json --filtered data/processed/filtered_papers.json
python Retrieval/retrieval.py query "evidence-based practice instruments" --k 5
python Retrieval/retrieval.py suggest --n 5
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

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
DEFAULT_FILTERED_FILE = os.path.join(BASE_DIR, "data", "processed", "filtered_papers.json")
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

# Retrieval filtering settings
# score = 1 / (1 + distance); lower values are less relevant.
DEFAULT_MIN_SCORE = 0.45
QUERY_OVERSAMPLE_FACTOR = 3


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


def distance_to_score(distance: Optional[float]) -> Optional[float]:
    if distance is None:
        return None
    return 1.0 / (1.0 + float(distance))


def _filter_query_result(result: dict, min_score: Optional[float], limit: Optional[int] = None) -> dict:
    if min_score is None:
        return result

    ids_by_query = result.get("ids")
    if not isinstance(ids_by_query, list):
        return result

    query_count = len(ids_by_query)
    if query_count == 0:
        return result

    result_keys = ("ids", "documents", "metadatas", "distances", "embeddings", "uris", "data")
    query_keys = [
        key for key in result_keys
        if isinstance(result.get(key), list) and len(result.get(key)) == query_count
    ]
    distances_by_query = result.get("distances")
    has_distances = isinstance(distances_by_query, list) and len(distances_by_query) == query_count

    filtered = dict(result)
    for key in query_keys:
        filtered[key] = []

    for q_idx in range(query_count):
        row_ids = ids_by_query[q_idx] if isinstance(ids_by_query[q_idx], list) else []
        keep_indices = []
        for i in range(len(row_ids)):
            dist = None
            if has_distances:
                row_distances = distances_by_query[q_idx] if isinstance(distances_by_query[q_idx], list) else []
                if i < len(row_distances):
                    dist = row_distances[i]
            score = distance_to_score(dist)
            if score is None or score >= min_score:
                keep_indices.append(i)

        if limit is not None:
            keep_indices = keep_indices[:limit]

        for key in query_keys:
            row_values = result[key][q_idx]
            if isinstance(row_values, list):
                filtered[key].append([row_values[i] for i in keep_indices if i < len(row_values)])
            else:
                filtered[key].append(row_values)

    return filtered


class FilteredCollectionProxy:
    """
    Transparent proxy around Chroma collection that applies an optional
    relevance-score filter on query results.
    """

    def __init__(self, collection, min_score: Optional[float], oversample_factor: int = QUERY_OVERSAMPLE_FACTOR):
        self._collection = collection
        self._min_score = min_score if min_score is not None and min_score > 0 else None
        self._oversample_factor = max(1, int(oversample_factor))

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def query(self, *args, **kwargs):
        original_n = kwargs.get("n_results")

        if self._min_score is not None and isinstance(original_n, int) and original_n > 0:
            requested_n = max(original_n, original_n * self._oversample_factor)
            if requested_n != original_n:
                kwargs = dict(kwargs)
                kwargs["n_results"] = requested_n

        raw_result = self._collection.query(*args, **kwargs)
        limit = original_n if isinstance(original_n, int) and original_n > 0 else None
        return _filter_query_result(raw_result, min_score=self._min_score, limit=limit)


# --------- Chroma helpers ----------
def get_chroma_collection(min_score: Optional[float] = DEFAULT_MIN_SCORE, oversample_factor: int = QUERY_OVERSAMPLE_FACTOR):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return FilteredCollectionProxy(collection, min_score=min_score, oversample_factor=oversample_factor)


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

    # paperIds allowed by screening (deduplicated)
    allowed_set = {p.get("paperId") for p in filtered if p.get("paperId")}
    allowed_ids = list(allowed_set)

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


def cmd_query(user_query: str, k: int, min_score: Optional[float] = DEFAULT_MIN_SCORE):
    require_retrieval_dependencies()

    if not user_query.strip():
        print("Empty query.")
        return

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError("No index found. Run: python Retrieval/retrieval.py index")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = get_chroma_collection(min_score=min_score)

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

    out = {"query": user_query, "k": k, "min_score": min_score, "results": []}
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        score = distance_to_score(dist)
        out["results"].append({
            "chunk_id": cid,
            "score": score,
            "paperId": meta.get("paperId"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "year": meta.get("year"),
            "text_source": meta.get("text_source"),
            "text": doc
        })

    if not out["results"]:
        out["note"] = "No chunks passed the relevance filter. Try lowering --min-score."

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
    p_query.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum relevance score. Use <= 0 to disable filtering.",
    )

    p_suggest = sub.add_parser("suggest", help="Return N suggested papers for greeting")
    p_suggest.add_argument("--n", type=int, default=5, help="How many papers to suggest")
    p_suggest.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Path to metadata JSON")
    p_suggest.add_argument("--filtered", default=DEFAULT_FILTERED_FILE, help="Path to filtered JSON")

    args = parser.parse_args()

    if args.cmd == "index":
        cmd_index(args.metadata, args.filtered)
    elif args.cmd == "query":
        min_score = args.min_score if args.min_score > 0 else None
        cmd_query(args.text, args.k, min_score=min_score)
    elif args.cmd == "suggest":
        cmd_suggest(args.n, args.metadata, args.filtered)


if __name__ == "__main__":
    main()
