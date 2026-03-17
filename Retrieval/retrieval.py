#!/usr/bin/env python3
"""
retrieval.py - local hybrid retrieval for the HybReDe RAG pipeline.

What it does:
1) index  : indexes screened papers into Chroma and writes a lexical sidecar index.
2) query  : runs hybrid retrieval (embeddings + BM25 + rerank + paper aggregation).
3) suggest: returns simple suggested papers for greeting links.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ModuleNotFoundError:
    CrossEncoder = None
    SentenceTransformer = None


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_FILTERED_FILE = os.path.join(BASE_DIR, "data", "processed", "filtered_papers.json")
DEFAULT_METADATA_FILE = os.path.join(BASE_DIR, "data", "hybrede_metadata_v3.json")
FULLTEXT_DIR = os.path.join(BASE_DIR, "data", "v3_full_text")
CHROMA_DIR = os.path.join(BASE_DIR, "rag_store")
COLLECTION_NAME = "hybrede"
LEXICAL_INDEX_FILE = os.path.join(CHROMA_DIR, "lexical_index.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_WORDS = 400
CHUNK_OVERLAP = 80
ABSTRACT_WORD_LIMIT = 220

DEFAULT_MIN_SCORE = 0.55
QUERY_OVERSAMPLE_FACTOR = 3
HYBRID_CANDIDATE_POOL = 40
CROSS_ENCODER_TOP_N = 20
MMR_CANDIDATE_POOL = 15
MMR_LAMBDA = 0.78
BM25_K1 = 1.5
BM25_B = 0.75

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "of", "on", "or", "that", "the", "to", "what",
    "which", "with",
}

SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "limitations",
}

SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)?\s*(abstract|introduction|background|method|methods|"
    r"materials and methods|results|discussion|conclusion|conclusions|limitations)\s*$",
    re.IGNORECASE,
)

SECTION_BASE_BOOSTS = {
    "abstract": 0.12,
    "introduction": 0.02,
    "background": 0.02,
    "methods": 0.05,
    "results": 0.08,
    "discussion": 0.05,
    "conclusion": 0.10,
    "limitations": 0.01,
    "body": 0.03,
}

QUERY_TYPE_CONFIG = {
    "general": {
        "hybrid_weights": {"embedding": 0.60, "bm25": 0.40},
        "section_boosts": {"abstract": 0.03, "conclusion": 0.03},
        "intent_terms": [],
    },
    "instrument": {
        "hybrid_weights": {"embedding": 0.45, "bm25": 0.55},
        "section_boosts": {"abstract": 0.04, "methods": 0.08, "conclusion": 0.02},
        "intent_terms": [
            "instrument", "scale", "questionnaire", "survey", "measure",
            "measurement", "assessment", "tool", "inventory", "checklist",
            "validation", "psychometric",
        ],
    },
    "evidence": {
        "hybrid_weights": {"embedding": 0.55, "bm25": 0.45},
        "section_boosts": {"abstract": 0.05, "results": 0.08, "conclusion": 0.08},
        "intent_terms": [
            "effect", "effective", "effectiveness", "efficacy", "outcome",
            "trial", "randomized", "evidence", "impact", "comparison",
        ],
    },
    "qualitative": {
        "hybrid_weights": {"embedding": 0.50, "bm25": 0.50},
        "section_boosts": {"abstract": 0.05, "methods": 0.08, "discussion": 0.08},
        "intent_terms": [
            "qualitative", "interview", "interviews", "focus", "group",
            "perception", "perceptions", "experience", "experiences",
            "attitude", "attitudes", "acceptability", "feasibility", "theme",
        ],
    },
}

DOMAIN_EXPANSIONS = {
    "evidence-based practice": ["evidence based practice", "ebp"],
    "evidence-based": ["evidence based", "ebp"],
    "instrument": ["instruments", "scale", "scales", "questionnaire", "questionnaires", "measure", "measures", "tool", "tools"],
    "scale": ["scales", "instrument", "questionnaire", "survey", "measure"],
    "questionnaire": ["questionnaires", "survey", "instrument", "scale", "measure"],
    "patient-reported outcome": ["patient reported outcome", "prom", "proms"],
    "clinical decision support": ["decision support", "cds"],
    "qualitative": ["interview", "interviews", "focus group", "focus groups", "perceptions", "experiences"],
}

_CROSS_ENCODER_LOAD_ERROR = None


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


@lru_cache(maxsize=1)
def get_embedding_model():
    require_retrieval_dependencies()
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_cross_encoder_model():
    global _CROSS_ENCODER_LOAD_ERROR
    if CrossEncoder is None:
        _CROSS_ENCODER_LOAD_ERROR = "sentence-transformers CrossEncoder is not installed"
        return None
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    except Exception as exc:
        _CROSS_ENCODER_LOAD_ERROR = str(exc)
        return None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_files_exist(*paths: str):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")


def normalize_text_for_match(text: str) -> str:
    normalized = (text or "").lower().replace("/", " ")
    normalized = re.sub(r"[^a-z0-9\-\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_lexical(text: str) -> List[str]:
    normalized = normalize_text_for_match(text).replace("-", " ")
    return [tok for tok in re.findall(r"[a-z0-9]+", normalized) if tok]


def singularize_term(term: str) -> str:
    if term.endswith("ies") and len(term) > 4:
        return f"{term[:-3]}y"
    if term.endswith("s") and not term.endswith("ss") and len(term) > 4:
        return term[:-1]
    return term


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def distance_to_score(distance: Optional[float]) -> Optional[float]:
    if distance is None:
        return None
    return 1.0 / (1.0 + float(distance))


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if math.isclose(lo, hi):
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(score - lo) / (hi - lo) for score in scores]


def extract_query_terms(query_text: str) -> List[str]:
    normalized_query = normalize_text_for_match(query_text)
    raw_terms = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized_query)

    terms: List[str] = []
    for term in raw_terms:
        if len(term) < 3 or term in QUERY_STOPWORDS:
            continue
        variants = [term, singularize_term(term)]
        for variant in list(variants):
            if "-" in variant:
                variants.append(variant.replace("-", " "))
        terms.extend(variants)

    return dedupe_preserve_order(terms)


def contains_exact_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def normalize_section_name(name: str) -> str:
    normalized = normalize_text_for_match(name)
    return SECTION_ALIASES.get(normalized, "body")


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
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


def split_text_into_sections(text: str) -> List[Tuple[str, str]]:
    cleaned = (text or "").replace("\r\n", "\n")
    if not cleaned.strip():
        return []

    sections: List[Tuple[str, str]] = []
    current_name = "body"
    current_lines: List[str] = []
    matched_heading = False

    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            if current_lines:
                current_lines.append("")
            continue

        heading_match = SECTION_HEADING_RE.match(line)
        if heading_match:
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append((current_name, section_text))
            current_name = normalize_section_name(heading_match.group(1))
            current_lines = []
            matched_heading = True
            continue

        current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_name, section_text))

    if not matched_heading:
        return [("body", cleaned.strip())]

    return [(name, section_text) for name, section_text in sections if section_text]


def build_embedding_input(title: str, abstract_text: str, section: str, chunk_text_value: str) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract_text:
        parts.append(f"Abstract: {truncate_words(abstract_text, ABSTRACT_WORD_LIMIT)}")
    if section and section != "body":
        parts.append(f"Section: {section}")
    parts.append(f"Passage: {chunk_text_value}")
    return "\n".join(parts).strip()


def build_metadata_index(metadata_records: List[dict]) -> Dict[str, dict]:
    idx = {}
    for rec in metadata_records:
        pid = rec.get("paperId")
        if pid:
            idx[pid] = rec
    return idx


def get_abstract_text(meta_record: dict) -> str:
    abstract = meta_record.get("abstract") or ""
    abstract = abstract.strip() if isinstance(abstract, str) else ""
    return abstract


def load_full_text(paper_id: str, fulltext_dir: str = FULLTEXT_DIR) -> str:
    if not paper_id:
        return ""
    fulltext_path = os.path.join(fulltext_dir, f"{paper_id}.txt")
    if not os.path.exists(fulltext_path):
        return ""
    with open(fulltext_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_best_url(meta_record: dict) -> str:
    oap = meta_record.get("openAccessPdf") or {}
    pdf_url = (oap.get("url") or "").strip() if isinstance(oap, dict) else ""
    if pdf_url:
        return pdf_url
    return (meta_record.get("url") or "").strip()


def get_text_for_paper(meta_record: dict, fulltext_dir: str = FULLTEXT_DIR) -> Tuple[str, str, str, int, str]:
    title = (meta_record.get("title") or "").strip()
    paper_id = (meta_record.get("paperId") or "").strip()
    year = meta_record.get("year") or 0
    abstract_text = get_abstract_text(meta_record)

    full_text = load_full_text(paper_id, fulltext_dir=fulltext_dir)
    if full_text:
        return title, abstract_text, full_text, year, "fulltext"

    return title, abstract_text, abstract_text or "No abstract provided", year, "abstract"


def build_chunk_records(meta_record: dict, fulltext_dir: str = FULLTEXT_DIR) -> List[dict]:
    title, abstract_text, text_for_index, year, text_source = get_text_for_paper(meta_record, fulltext_dir=fulltext_dir)
    paper_id = (meta_record.get("paperId") or "").strip()
    url = get_best_url(meta_record)

    sections: List[Tuple[str, str]]
    if text_source == "fulltext":
        sections = split_text_into_sections(text_for_index)
        if not sections:
            sections = [("body", text_for_index)]
    else:
        sections = [("abstract", text_for_index)]

    records: List[dict] = []
    chunk_counter = 0
    for section_name, section_text in sections:
        for chunk_text_value in chunk_text(section_text, CHUNK_WORDS, CHUNK_OVERLAP):
            records.append({
                "chunk_id": f"{paper_id}:{chunk_counter:04d}",
                "paperId": paper_id,
                "title": title,
                "abstract_text": abstract_text,
                "url": url,
                "year": year,
                "text_source": text_source,
                "section": section_name,
                "text": chunk_text_value,
            })
            chunk_counter += 1
    return records


def build_query_analysis(query_text: str) -> dict:
    normalized_query = normalize_text_for_match(query_text)
    base_terms = extract_query_terms(query_text)
    expanded_terms = list(base_terms)

    for phrase, variants in DOMAIN_EXPANSIONS.items():
        normalized_phrase = normalize_text_for_match(phrase)
        if normalized_phrase in normalized_query or any(term == normalized_phrase for term in base_terms):
            expanded_terms.extend(extract_query_terms(" ".join([phrase] + variants)))

    detected_query_type = "general"
    if any(term in normalized_query for term in ("instrument", "questionnaire", "scale", "measure", "survey", "tool", "psychometric")):
        detected_query_type = "instrument"
    elif any(term in normalized_query for term in ("qualitative", "interview", "focus group", "perception", "experience", "acceptability", "feasibility")):
        detected_query_type = "qualitative"
    elif any(term in normalized_query for term in ("evidence", "effectiveness", "efficacy", "outcome", "trial", "effective")):
        detected_query_type = "evidence"

    expanded_terms.extend(QUERY_TYPE_CONFIG[detected_query_type]["intent_terms"])
    expanded_terms = dedupe_preserve_order(expanded_terms)

    expanded_query_text = " ".join(dedupe_preserve_order([query_text] + expanded_terms))

    return {
        "raw_query": query_text,
        "normalized_query": normalized_query,
        "query_type": detected_query_type,
        "base_terms": base_terms,
        "expanded_terms": expanded_terms,
        "expanded_query_text": expanded_query_text,
        "hybrid_weights": QUERY_TYPE_CONFIG[detected_query_type]["hybrid_weights"],
        "section_boosts": QUERY_TYPE_CONFIG[detected_query_type]["section_boosts"],
        "intent_terms": QUERY_TYPE_CONFIG[detected_query_type]["intent_terms"],
    }


def compute_field_match_score(query_analysis: dict, text: str, field_name: str) -> float:
    normalized_text = normalize_text_for_match(text)
    if not normalized_text:
        return 0.0

    terms = query_analysis["expanded_terms"]
    if not terms:
        return 0.0

    matched_terms = [
        term for term in terms
        if contains_exact_term(normalized_text, term)
    ]
    if not matched_terms:
        return 0.0

    field_weights = {"title": 0.34, "abstract": 0.22, "body": 0.14}
    phrase_weight = {"title": 0.20, "abstract": 0.14, "body": 0.08}
    hit_weight = {"title": 0.05, "abstract": 0.035, "body": 0.025}

    coverage = len(set(matched_terms)) / max(1, len(terms))
    score = field_weights.get(field_name, 0.14) * coverage
    score += hit_weight.get(field_name, 0.025) * len(matched_terms)

    normalized_query = query_analysis["normalized_query"]
    normalized_query_alt = normalized_query.replace("-", " ")
    if normalized_query and " " in normalized_query:
        if normalized_query in normalized_text:
            score += phrase_weight.get(field_name, 0.08)
        if normalized_query_alt != normalized_query and normalized_query_alt in normalized_text:
            score += phrase_weight.get(field_name, 0.08) * 0.7

    return score


def compute_query_type_boost(query_analysis: dict, record: dict) -> float:
    intent_terms = query_analysis["intent_terms"]
    if not intent_terms:
        return 0.0

    searchable = " ".join([
        record.get("title", ""),
        record.get("abstract_text", ""),
        record.get("text", ""),
    ])
    normalized_searchable = normalize_text_for_match(searchable)
    matches = sum(1 for term in intent_terms if contains_exact_term(normalized_searchable, term))
    if matches == 0:
        return 0.0
    return min(0.16, matches * 0.03)


def compute_section_boost(query_analysis: dict, section: str) -> float:
    normalized_section = normalize_section_name(section)
    return SECTION_BASE_BOOSTS.get(normalized_section, 0.03) + query_analysis["section_boosts"].get(normalized_section, 0.0)


def compute_source_boost(text_source: str) -> float:
    if text_source == "fulltext":
        return 0.06
    return 0.01


class LexicalIndex:
    def __init__(self, records: List[dict]):
        self.records = records
        self.record_by_id: Dict[str, dict] = {}
        self.search_tf: Dict[str, Counter] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freqs: Counter = Counter()
        self.avg_doc_length = 0.0

        total_doc_length = 0
        for record in records:
            chunk_id = record["chunk_id"]
            self.record_by_id[chunk_id] = record

            title_tokens = tokenize_lexical(record.get("title", ""))
            abstract_tokens = tokenize_lexical(record.get("abstract_text", ""))
            body_tokens = tokenize_lexical(record.get("text", ""))
            search_tokens = title_tokens + abstract_tokens + body_tokens

            tf = Counter(search_tokens)
            self.search_tf[chunk_id] = tf
            self.doc_lengths[chunk_id] = len(search_tokens)
            total_doc_length += len(search_tokens)

            for token in tf.keys():
                self.doc_freqs[token] += 1

        if records:
            self.avg_doc_length = total_doc_length / len(records)

    @classmethod
    def from_file(cls, path: str) -> Optional["LexicalIndex"]:
        if not os.path.exists(path):
            return None
        payload = load_json(path)
        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            return None
        return cls(records)

    def search(self, query_analysis: dict, limit: int) -> List[dict]:
        if not self.records or limit <= 0:
            return []

        query_terms = tokenize_lexical(" ".join(query_analysis["expanded_terms"]))
        query_terms = dedupe_preserve_order(query_terms)
        if not query_terms:
            return []

        candidates = []
        doc_count = len(self.records)
        avg_doc_length = self.avg_doc_length or 1.0

        for record in self.records:
            chunk_id = record["chunk_id"]
            tf = self.search_tf.get(chunk_id, Counter())
            if not tf:
                continue

            bm25_score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
                doc_len = self.doc_lengths.get(chunk_id, 0)
                denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / avg_doc_length))
                bm25_score += idf * ((freq * (BM25_K1 + 1)) / max(denom, 1e-9))

            title_score = compute_field_match_score(query_analysis, record.get("title", ""), "title")
            abstract_score = compute_field_match_score(query_analysis, record.get("abstract_text", ""), "abstract")
            body_score = compute_field_match_score(query_analysis, record.get("text", ""), "body")
            lexical_score = bm25_score + title_score + abstract_score + body_score

            if lexical_score <= 0:
                continue

            candidates.append({
                "chunk_id": chunk_id,
                "bm25_score": bm25_score,
                "lexical_score": lexical_score,
                "title_field_score": title_score,
                "abstract_field_score": abstract_score,
                "body_field_score": body_score,
            })

        candidates.sort(key=lambda item: (item["lexical_score"], item["bm25_score"]), reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["bm25_rank"] = rank
        return candidates[:limit]


class HybridCollectionProxy:
    def __init__(self, collection, lexical_index: Optional[LexicalIndex], min_score: Optional[float], oversample_factor: int = QUERY_OVERSAMPLE_FACTOR):
        self._collection = collection
        self._lexical_index = lexical_index
        self._min_score = min_score if min_score is not None and min_score > 0 else None
        self._oversample_factor = max(1, int(oversample_factor))

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def _fallback_vector_query(self, query_embeddings: List[List[float]], n_results: int, include: List[str], query_texts: List[str]):
        raw_result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
        )

        ids_by_query = raw_result.get("ids", [])
        documents_by_query = raw_result.get("documents", [])
        metadatas_by_query = raw_result.get("metadatas", [])
        distances_by_query = raw_result.get("distances", [])

        out = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "embedding_scores": [],
            "bm25_scores": [],
            "hybrid_scores": [],
            "paper_scores": [],
            "cross_encoder_scores": [],
            "mmr_scores": [],
            "final_scores": [],
            "query_analysis": [],
            "retrieval_notes": [],
        }

        for q_idx, query_text in enumerate(query_texts):
            row_ids = ids_by_query[q_idx] if q_idx < len(ids_by_query) else []
            row_docs = documents_by_query[q_idx] if q_idx < len(documents_by_query) else []
            row_metas = metadatas_by_query[q_idx] if q_idx < len(metadatas_by_query) else []
            row_distances = distances_by_query[q_idx] if q_idx < len(distances_by_query) else []
            analysis = build_query_analysis(query_text)

            keep = []
            for i, _chunk_id in enumerate(row_ids):
                distance = row_distances[i] if i < len(row_distances) else None
                embedding_score = distance_to_score(distance)
                if self._min_score is None or embedding_score is None or embedding_score >= self._min_score:
                    keep.append(i)

            row_ids = [row_ids[i] for i in keep]
            row_docs = [row_docs[i] for i in keep if i < len(row_docs)]
            row_metas = [row_metas[i] for i in keep if i < len(row_metas)]
            row_distances = [row_distances[i] for i in keep if i < len(row_distances)]

            out["ids"].append(row_ids[:n_results])
            out["documents"].append(row_docs[:n_results])
            out["metadatas"].append(row_metas[:n_results])
            out["distances"].append(row_distances[:n_results])
            embedding_scores = [distance_to_score(dist) or 0.0 for dist in row_distances[:n_results]]
            out["embedding_scores"].append(embedding_scores)
            out["bm25_scores"].append([0.0 for _ in embedding_scores])
            out["hybrid_scores"].append(list(embedding_scores))
            out["paper_scores"].append(list(embedding_scores))
            out["cross_encoder_scores"].append([None for _ in embedding_scores])
            out["mmr_scores"].append(list(embedding_scores))
            out["final_scores"].append(list(embedding_scores))
            out["query_analysis"].append(analysis)
            out["retrieval_notes"].append(["vector-only fallback"])

        return out

    def query(self, *args, **kwargs):
        query_embeddings = kwargs.get("query_embeddings")
        original_n = kwargs.get("n_results")
        include = kwargs.get("include") or ["documents", "metadatas", "distances"]
        query_text = kwargs.pop("query_text", None)

        if not isinstance(query_embeddings, list) or not isinstance(original_n, int) or original_n <= 0:
            return self._collection.query(*args, **kwargs)

        query_texts = query_text if isinstance(query_text, list) else [query_text or "" for _ in range(len(query_embeddings))]

        if self._lexical_index is None:
            return self._fallback_vector_query(query_embeddings, original_n, include, query_texts)

        candidate_pool = max(original_n * self._oversample_factor, HYBRID_CANDIDATE_POOL)
        raw_vector_result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=candidate_pool,
            include=["documents", "metadatas", "distances"],
        )

        return hybrid_query_result(
            collection=self._collection,
            lexical_index=self._lexical_index,
            raw_vector_result=raw_vector_result,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            k=original_n,
            min_score=self._min_score,
        )


def build_candidate_record(base_record: dict) -> dict:
    return {
        "chunk_id": base_record["chunk_id"],
        "paperId": base_record.get("paperId"),
        "title": base_record.get("title", ""),
        "abstract_text": base_record.get("abstract_text", ""),
        "url": base_record.get("url", ""),
        "year": base_record.get("year"),
        "text_source": base_record.get("text_source", "unknown"),
        "section": base_record.get("section", "body"),
        "text": base_record.get("text", ""),
        "distance": None,
        "embedding_score": 0.0,
        "bm25_score": 0.0,
        "lexical_score": 0.0,
        "title_field_score": 0.0,
        "abstract_field_score": 0.0,
        "body_field_score": 0.0,
        "field_score": 0.0,
        "section_boost": 0.0,
        "source_boost": 0.0,
        "query_type_boost": 0.0,
        "hybrid_score": 0.0,
        "chunk_score": 0.0,
        "paper_score": 0.0,
        "cross_encoder_score": None,
        "mmr_score": None,
    }


def aggregate_papers(candidates: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for candidate in candidates:
        paper_id = candidate.get("paperId") or candidate["chunk_id"]
        grouped[paper_id].append(candidate)

    paper_entries = []
    for paper_id, paper_candidates in grouped.items():
        paper_candidates.sort(key=lambda item: item["chunk_score"], reverse=True)
        best_candidate = paper_candidates[0]
        second_score = paper_candidates[1]["chunk_score"] if len(paper_candidates) > 1 else 0.0
        support_bonus = min(0.14, second_score * 0.18) + min(0.06, max(0, len(paper_candidates) - 1) * 0.02)
        paper_score = best_candidate["chunk_score"] + support_bonus

        representative = dict(best_candidate)
        representative["paper_score"] = paper_score
        representative["supporting_chunks"] = len(paper_candidates)
        representative["candidate_chunk_ids"] = [candidate["chunk_id"] for candidate in paper_candidates]
        paper_entries.append(representative)

    paper_entries.sort(key=lambda item: (item["paper_score"], item["chunk_score"]), reverse=True)
    return paper_entries


def apply_cross_encoder(query_analysis: dict, papers: List[dict]) -> str:
    model = get_cross_encoder_model()
    if model is None:
        return _CROSS_ENCODER_LOAD_ERROR or "cross-encoder unavailable"

    top_n = min(CROSS_ENCODER_TOP_N, len(papers))
    if top_n == 0:
        return "cross-encoder skipped: no candidates"

    pairs = []
    for paper in papers[:top_n]:
        pair_text = "\n".join([
            f"Title: {paper.get('title', '')}",
            f"Abstract: {truncate_words(paper.get('abstract_text', ''), 120)}",
            f"Section: {paper.get('section', 'body')}",
            f"Excerpt: {truncate_words(paper.get('text', ''), 180)}",
        ])
        pairs.append((query_analysis["raw_query"], pair_text))

    raw_scores = model.predict(pairs)
    raw_scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
    normalized_scores = normalize_scores([float(score) for score in raw_scores])

    for idx, normalized_score in enumerate(normalized_scores):
        papers[idx]["cross_encoder_score"] = normalized_score
        papers[idx]["paper_score"] += 0.24 * normalized_score

    papers.sort(key=lambda item: (item["paper_score"], item["chunk_score"]), reverse=True)
    return "cross-encoder applied"


def apply_mmr_selection(query_embedding: List[float], papers: List[dict], k: int) -> List[dict]:
    if not papers:
        return []

    candidate_pool = min(max(k * 3, MMR_CANDIDATE_POOL), len(papers))
    shortlisted = papers[:candidate_pool]
    if len(shortlisted) <= k:
        for paper in shortlisted:
            paper["mmr_score"] = paper["paper_score"]
        return shortlisted

    model = get_embedding_model()
    representative_texts = [
        "\n".join([
            f"Title: {paper.get('title', '')}",
            f"Abstract: {truncate_words(paper.get('abstract_text', ''), 120)}",
            f"Section: {paper.get('section', 'body')}",
            f"Excerpt: {truncate_words(paper.get('text', ''), 180)}",
        ])
        for paper in shortlisted
    ]
    embeddings = model.encode(representative_texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    relevance_scores = normalize_scores([paper["paper_score"] for paper in shortlisted])
    selected_indices: List[int] = []
    remaining = set(range(len(shortlisted)))

    while remaining and len(selected_indices) < k:
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = relevance_scores[idx]
            diversity_penalty = 0.0
            if selected_indices:
                diversity_penalty = max(
                    cosine_similarity(embeddings[idx], embeddings[selected_idx])
                    for selected_idx in selected_indices
                )
            mmr_score = (MMR_LAMBDA * relevance) - ((1 - MMR_LAMBDA) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        shortlisted[best_idx]["mmr_score"] = best_score
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    selected = [shortlisted[idx] for idx in selected_indices]
    selected.sort(key=lambda item: (item.get("mmr_score") or 0.0, item["paper_score"]), reverse=True)
    return selected


def build_result_row(selected_papers: List[dict], query_analysis: dict, retrieval_notes: List[str]) -> dict:
    return {
        "ids": [paper["chunk_id"] for paper in selected_papers],
        "documents": [paper["text"] for paper in selected_papers],
        "metadatas": [
            {
                "paperId": paper.get("paperId"),
                "title": paper.get("title"),
                "url": paper.get("url"),
                "year": paper.get("year"),
                "text_source": paper.get("text_source"),
                "section": paper.get("section"),
                "supporting_chunks": paper.get("supporting_chunks", 1),
            }
            for paper in selected_papers
        ],
        "distances": [paper.get("distance") for paper in selected_papers],
        "embedding_scores": [paper.get("embedding_score", 0.0) for paper in selected_papers],
        "bm25_scores": [paper.get("bm25_score", 0.0) for paper in selected_papers],
        "hybrid_scores": [paper.get("hybrid_score", 0.0) for paper in selected_papers],
        "paper_scores": [paper.get("paper_score", 0.0) for paper in selected_papers],
        "cross_encoder_scores": [paper.get("cross_encoder_score") for paper in selected_papers],
        "mmr_scores": [paper.get("mmr_score") for paper in selected_papers],
        "final_scores": [
            (paper.get("mmr_score") if paper.get("mmr_score") is not None else paper.get("paper_score", 0.0))
            for paper in selected_papers
        ],
        "query_analysis": query_analysis,
        "retrieval_notes": retrieval_notes,
    }


def hybrid_query_result(collection, lexical_index: LexicalIndex, raw_vector_result: dict, query_embeddings: List[List[float]], query_texts: List[str], k: int, min_score: Optional[float]) -> dict:
    out = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "distances": [],
        "embedding_scores": [],
        "bm25_scores": [],
        "hybrid_scores": [],
        "paper_scores": [],
        "cross_encoder_scores": [],
        "mmr_scores": [],
        "final_scores": [],
        "query_analysis": [],
        "retrieval_notes": [],
    }

    ids_by_query = raw_vector_result.get("ids", [])
    documents_by_query = raw_vector_result.get("documents", [])
    metadatas_by_query = raw_vector_result.get("metadatas", [])
    distances_by_query = raw_vector_result.get("distances", [])

    for q_idx, query_text in enumerate(query_texts):
        query_analysis = build_query_analysis(query_text)
        retrieval_notes = ["hybrid retrieval", f"query_type={query_analysis['query_type']}"]

        row_ids = ids_by_query[q_idx] if q_idx < len(ids_by_query) else []
        row_docs = documents_by_query[q_idx] if q_idx < len(documents_by_query) else []
        row_metas = metadatas_by_query[q_idx] if q_idx < len(metadatas_by_query) else []
        row_distances = distances_by_query[q_idx] if q_idx < len(distances_by_query) else []

        lexical_candidates = lexical_index.search(query_analysis, limit=max(k * QUERY_OVERSAMPLE_FACTOR, HYBRID_CANDIDATE_POOL))
        lexical_rank_by_id = {candidate["chunk_id"]: idx + 1 for idx, candidate in enumerate(lexical_candidates)}
        candidate_by_id: Dict[str, dict] = {}

        for vector_rank, chunk_id in enumerate(row_ids, start=1):
            base_meta = row_metas[vector_rank - 1] if vector_rank - 1 < len(row_metas) else {}
            base_record = lexical_index.record_by_id.get(chunk_id, {
                "chunk_id": chunk_id,
                "paperId": base_meta.get("paperId"),
                "title": base_meta.get("title", ""),
                "abstract_text": "",
                "url": base_meta.get("url", ""),
                "year": base_meta.get("year"),
                "text_source": base_meta.get("text_source", "unknown"),
                "section": base_meta.get("section", "body"),
                "text": row_docs[vector_rank - 1] if vector_rank - 1 < len(row_docs) else "",
            })
            candidate = build_candidate_record(base_record)
            candidate["distance"] = row_distances[vector_rank - 1] if vector_rank - 1 < len(row_distances) else None
            candidate["embedding_score"] = distance_to_score(candidate["distance"]) or 0.0
            candidate["vector_rank"] = vector_rank
            candidate_by_id[chunk_id] = candidate

        for lexical_candidate in lexical_candidates:
            chunk_id = lexical_candidate["chunk_id"]
            base_record = lexical_index.record_by_id.get(chunk_id)
            if not base_record:
                continue
            candidate = candidate_by_id.get(chunk_id) or build_candidate_record(base_record)
            candidate["bm25_score"] = lexical_candidate.get("bm25_score", 0.0)
            candidate["lexical_score"] = lexical_candidate.get("lexical_score", 0.0)
            candidate["title_field_score"] = lexical_candidate.get("title_field_score", 0.0)
            candidate["abstract_field_score"] = lexical_candidate.get("abstract_field_score", 0.0)
            candidate["body_field_score"] = lexical_candidate.get("body_field_score", 0.0)
            candidate["bm25_rank"] = lexical_rank_by_id.get(chunk_id)
            candidate_by_id[chunk_id] = candidate

        if not candidate_by_id:
            empty_row = build_result_row([], query_analysis, retrieval_notes + ["no candidates"])
            for key, value in empty_row.items():
                out[key].append(value)
            continue

        candidates = list(candidate_by_id.values())
        normalized_embedding_scores = normalize_scores([candidate["embedding_score"] for candidate in candidates])
        normalized_bm25_scores = normalize_scores([candidate["bm25_score"] for candidate in candidates])

        embedding_weight = query_analysis["hybrid_weights"]["embedding"]
        bm25_weight = query_analysis["hybrid_weights"]["bm25"]

        scored_candidates = []
        for idx, candidate in enumerate(candidates):
            candidate["field_score"] = candidate["title_field_score"] + candidate["abstract_field_score"] + candidate["body_field_score"]
            candidate["section_boost"] = compute_section_boost(query_analysis, candidate.get("section", "body"))
            candidate["source_boost"] = compute_source_boost(candidate.get("text_source", "unknown"))
            candidate["query_type_boost"] = compute_query_type_boost(query_analysis, candidate)
            candidate["hybrid_score"] = (
                (embedding_weight * normalized_embedding_scores[idx]) +
                (bm25_weight * normalized_bm25_scores[idx])
            )
            candidate["chunk_score"] = (
                candidate["hybrid_score"] +
                candidate["field_score"] +
                candidate["section_boost"] +
                candidate["source_boost"] +
                candidate["query_type_boost"]
            )
            if min_score is None or candidate["embedding_score"] >= min_score or candidate["bm25_score"] > 0:
                scored_candidates.append(candidate)

        if not scored_candidates:
            empty_row = build_result_row([], query_analysis, retrieval_notes + ["all candidates filtered by min_score"])
            for key, value in empty_row.items():
                out[key].append(value)
            continue

        scored_candidates.sort(key=lambda item: (item["chunk_score"], item["embedding_score"], item["bm25_score"]), reverse=True)
        paper_entries = aggregate_papers(scored_candidates)
        retrieval_notes.append("paper-level aggregation")

        cross_encoder_note = apply_cross_encoder(query_analysis, paper_entries)
        if cross_encoder_note:
            retrieval_notes.append(cross_encoder_note)

        selected_papers = apply_mmr_selection(query_embeddings[q_idx], paper_entries, k)
        retrieval_notes.append("mmr diversification")
        result_row = build_result_row(selected_papers, query_analysis, retrieval_notes)
        for key, value in result_row.items():
            out[key].append(value)

    return out


def get_chroma_collection(min_score: Optional[float] = DEFAULT_MIN_SCORE, oversample_factor: int = QUERY_OVERSAMPLE_FACTOR):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    lexical_index = LexicalIndex.from_file(LEXICAL_INDEX_FILE)
    return HybridCollectionProxy(collection, lexical_index=lexical_index, min_score=min_score, oversample_factor=oversample_factor)


def collection_count(collection) -> int:
    try:
        return collection.count()
    except Exception:
        return 0


def cmd_index(metadata_file: str, filtered_file: str):
    require_retrieval_dependencies()
    ensure_files_exist(filtered_file, metadata_file)

    filtered = load_json(filtered_file)
    metadata = load_json(metadata_file)
    meta_index = build_metadata_index(metadata)
    allowed_set = {p.get("paperId") for p in filtered if p.get("paperId")}
    allowed_ids = list(allowed_set)

    model = get_embedding_model()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    docs: List[str] = []
    ids: List[str] = []
    metas: List[dict] = []
    embedding_inputs: List[str] = []
    lexical_records: List[dict] = []

    missing_in_metadata = 0
    total_chunks = 0
    fulltext_papers = 0
    abstract_fallback_papers = 0

    for pid in allowed_ids:
        meta_rec = meta_index.get(pid)
        if not meta_rec:
            missing_in_metadata += 1
            continue

        text_source = "fulltext" if load_full_text(pid, fulltext_dir=FULLTEXT_DIR) else "abstract"
        if text_source == "fulltext":
            fulltext_papers += 1
        else:
            abstract_fallback_papers += 1

        records = build_chunk_records(meta_rec, fulltext_dir=FULLTEXT_DIR)
        for record in records:
            lexical_records.append(record)
            docs.append(record["text"])
            ids.append(record["chunk_id"])
            metas.append({
                "paperId": record["paperId"],
                "title": record["title"],
                "url": record["url"],
                "year": record["year"],
                "text_source": record["text_source"],
                "section": record["section"],
            })
            embedding_inputs.append(
                build_embedding_input(
                    title=record["title"],
                    abstract_text=record["abstract_text"],
                    section=record["section"],
                    chunk_text_value=record["text"],
                )
            )
            total_chunks += 1

    if total_chunks == 0:
        print("Nothing to index: no full text, no abstracts, or filtered file is empty")
        return

    embeddings = model.encode(
        embedding_inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings,
    )

    save_json(LEXICAL_INDEX_FILE, {
        "version": 2,
        "metadata_file": metadata_file,
        "filtered_file": filtered_file,
        "record_count": len(lexical_records),
        "records": lexical_records,
    })

    print("=== INDEX DONE ===")
    print(f"Metadata input: {metadata_file}")
    print(f"Filtered input: {filtered_file}")
    print(f"Allowed papers (from filtered): {len(allowed_set)}")
    print(f"Missing paperId in metadata: {missing_in_metadata}")
    print(f"Papers indexed from full text: {fulltext_papers}")
    print(f"Papers indexed from abstract fallback: {abstract_fallback_papers}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Lexical records written: {len(lexical_records)}")
    print(f"Chroma collection size: {collection_count(collection)}")
    print(f"Stored at: {CHROMA_DIR}/ (collection: {COLLECTION_NAME})")


def cmd_query(user_query: str, k: int, min_score: Optional[float] = DEFAULT_MIN_SCORE):
    require_retrieval_dependencies()

    if not user_query.strip():
        print("Empty query.")
        return

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError("No index found. Run: python Retrieval/retrieval.py index")

    model = get_embedding_model()
    collection = get_chroma_collection(min_score=min_score)
    q_emb = model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True).tolist()[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        query_text=user_query,
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    embedding_scores = res.get("embedding_scores", [[]])[0]
    bm25_scores = res.get("bm25_scores", [[]])[0]
    hybrid_scores = res.get("hybrid_scores", [[]])[0]
    paper_scores = res.get("paper_scores", [[]])[0]
    cross_encoder_scores = res.get("cross_encoder_scores", [[]])[0]
    mmr_scores = res.get("mmr_scores", [[]])[0]
    final_scores = res.get("final_scores", [[]])[0]
    query_analysis = res.get("query_analysis", [{}])[0]
    retrieval_notes = res.get("retrieval_notes", [[]])[0]

    out = {
        "query": user_query,
        "k": k,
        "min_score": min_score,
        "query_analysis": query_analysis,
        "retrieval_notes": retrieval_notes,
        "results": [],
    }
    for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas)):
        out["results"].append({
            "chunk_id": cid,
            "score": final_scores[i] if i < len(final_scores) else None,
            "embedding_score": embedding_scores[i] if i < len(embedding_scores) else distance_to_score(dists[i]) if i < len(dists) else None,
            "bm25_score": bm25_scores[i] if i < len(bm25_scores) else None,
            "hybrid_score": hybrid_scores[i] if i < len(hybrid_scores) else None,
            "paper_score": paper_scores[i] if i < len(paper_scores) else None,
            "cross_encoder_score": cross_encoder_scores[i] if i < len(cross_encoder_scores) else None,
            "mmr_score": mmr_scores[i] if i < len(mmr_scores) else None,
            "paperId": meta.get("paperId"),
            "title": meta.get("title"),
            "url": meta.get("url"),
            "year": meta.get("year"),
            "text_source": meta.get("text_source"),
            "section": meta.get("section"),
            "supporting_chunks": meta.get("supporting_chunks"),
            "text": doc,
        })

    if not out["results"]:
        out["note"] = "No papers passed the hybrid relevance filter. Try lowering --min-score or rebuilding the index."

    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_suggest(n: int, metadata_file: str, filtered_file: str):
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
        suggestions.append({
            "paperId": pid,
            "title": (rec.get("title") or "").strip(),
            "year": rec.get("year") or 0,
            "url": get_best_url(rec),
        })

    suggestions = list({item["paperId"]: item for item in suggestions}.values())
    suggestions.sort(key=lambda item: int(item.get("year") or 0), reverse=True)

    out = {
        "metadata_file": metadata_file,
        "filtered_file": filtered_file,
        "suggestions": suggestions[:n],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="HybReDe hybrid retrieval (local)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build local hybrid index from filtered + metadata")
    p_index.add_argument("--metadata", default=DEFAULT_METADATA_FILE, help="Path to metadata JSON")
    p_index.add_argument("--filtered", default=DEFAULT_FILTERED_FILE, help="Path to filtered JSON")

    p_query = sub.add_parser("query", help="Retrieve top-k relevant papers for a user query")
    p_query.add_argument("text", type=str, help="User query text")
    p_query.add_argument("--k", type=int, default=5, help="Number of results (top-k)")
    p_query.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum embedding relevance score. Use <= 0 to disable the semantic filter.",
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
