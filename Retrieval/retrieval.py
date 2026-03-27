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
import sys
import unicodedata
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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from console_utils import dump_json_console
from project_paths import (
    FILTERED_PAPERS_PATH,
    FULLTEXT_DIR,
    METADATA_PATH,
    RAG_STORE_DIR,
)
from run_manifest import RunManifest

DEFAULT_FILTERED_FILE = FILTERED_PAPERS_PATH
DEFAULT_METADATA_FILE = METADATA_PATH
CHROMA_DIR = RAG_STORE_DIR
COLLECTION_NAME = "hybrede"
LEXICAL_INDEX_FILE = os.path.join(CHROMA_DIR, "lexical_index.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_WORDS = 400
CHUNK_OVERLAP = 80
ABSTRACT_WORD_LIMIT = 220

DEFAULT_MIN_SCORE = 0.45
QUERY_OVERSAMPLE_FACTOR = 3
HYBRID_CANDIDATE_POOL = 40
LEXICAL_CANDIDATE_POOL = 25
CROSS_ENCODER_TOP_N = 20
MMR_CANDIDATE_POOL = 15
MMR_LAMBDA = 0.78
BM25_K1 = 1.5
BM25_B = 0.75
FIELD_SCORE_CAPS = {"title": 0.45, "abstract": 0.28, "body": 0.18}
FIELD_HIT_CAPS = {"title": 3, "abstract": 3, "body": 2}
QUERY_TYPE_BOOST_CAP = 0.06
QUERY_TYPE_BOOST_PER_TERM = 0.015
STRICT_LEXICAL_MIN_EMBEDDING = 0.18
STRICT_LEXICAL_SOFT_RATIO = 0.70
STRICT_LEXICAL_MIN_TITLE_HITS = 2
STRICT_LEXICAL_MIN_SUPPORT_HITS = 1
PAPER_SUPPORT_MAX_CHUNKS = 3
PAPER_SUPPORT_MIN_RELATIVE_SCORE = 0.85
PAPER_SUPPORT_PER_CHUNK = 0.025
PAPER_SUPPORT_MAX_BONUS = 0.075
PAPER_SECTION_DIVERSITY_BONUS = 0.01
PAPER_SECTION_DIVERSITY_MAX = 0.02
CROSS_ENCODER_BLEND_WEIGHT = 0.16

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "of", "on", "or", "that", "the", "to", "what",
    "which", "with",
}

INSTRUMENT_DETECTION_SPECIFIC_TERMS = [
    "instrument", "questionnaire", "scale", "measure", "measurement",
    "survey", "psychometric", "validation", "inventory", "checklist",
]
INSTRUMENT_DETECTION_BROAD_TERMS = ["tool", "tools", "assessment", "assess"]
QUALITATIVE_DETECTION_TERMS = [
    "qualitative", "interview", "focus group", "perception", "experience",
    "acceptability", "feasibility", "barrier", "barriers", "challenge",
    "challenges", "implementation", "adoption",
]
EVIDENCE_DETECTION_TERMS = [
    "evidence", "effectiveness", "efficacy", "outcome", "trial", "randomized",
]

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
            "barrier", "barriers", "challenge", "challenges",
            "implementation", "adoption", "facilitator", "facilitators",
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
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", normalized)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^\w\-\s]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_lexical(text: str) -> List[str]:
    normalized = normalize_text_for_match(text).replace("-", " ")
    return [tok for tok in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE) if tok]


def singularize_term(term: str) -> str:
    if not term:
        return term

    normalized = normalize_text_for_match(term)
    if " " in normalized or "-" in normalized:
        return normalized

    irregular_or_invariant = {
        "analysis", "analyses", "basis", "crisis", "diagnosis", "diabetes",
        "evidence", "hypothesis", "mathematics", "news", "physics", "thesis",
    }
    if normalized in irregular_or_invariant:
        return normalized
    if normalized.endswith(("ss", "us", "is", "ics", "ness", "osis", "asis")):
        return normalized
    if normalized.endswith("ies") and len(normalized) > 4:
        return f"{normalized[:-3]}y"
    if normalized.endswith("ses") and len(normalized) > 4 and not normalized.endswith(("sses", "uses", "ises")):
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 4:
        if normalized.endswith(("aes", "ees", "oes")):
            return normalized
        if normalized.endswith("tes") and normalized in {"diabetes"}:
            return normalized
        return normalized[:-1]
    return normalized


def canonicalize_term(term: str) -> str:
    normalized = normalize_text_for_match(term).replace("-", " ")
    tokens = tokenize_lexical(normalized)
    if not tokens:
        return ""
    return " ".join(singularize_term(token) for token in tokens if token)


def build_term_variant_groups(terms: List[str]) -> List[List[str]]:
    grouped: Dict[str, List[str]] = {}
    ordered_keys: List[str] = []

    for term in terms:
        normalized = normalize_text_for_match(term)
        if not normalized:
            continue
        key = canonicalize_term(normalized)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)

        variants = grouped[key]
        for variant in {normalized, normalized.replace("-", " "), key, key.replace("-", " ")}:
            cleaned_variant = normalize_text_for_match(variant)
            if cleaned_variant and cleaned_variant not in variants:
                variants.append(cleaned_variant)

    return [grouped[key] for key in ordered_keys]


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


def build_retrieval_score_payload(
    cross_encoder_score: Optional[float] = None,
    final_score: Optional[float] = None,
) -> dict:
    return {
        "cross_encoder_score": cross_encoder_score,
        "final_score": final_score,
    }


def enrich_metadata_with_scores(metadata: Optional[dict] = None, **score_values) -> dict:
    enriched = dict(metadata or {})
    retrieval_scores = {
        key: value for key, value in build_retrieval_score_payload(**score_values).items()
        if value is not None
    }
    if retrieval_scores:
        enriched.update(retrieval_scores)
        enriched["retrieval_scores"] = retrieval_scores
    return enriched


def semantic_distance_from_embeddings(query_embedding: List[float], candidate_embedding: List[float]) -> Optional[float]:
    if query_embedding is None or candidate_embedding is None:
        return None
    if len(query_embedding) == 0 or len(candidate_embedding) == 0:
        return None
    cos = max(-1.0, min(1.0, cosine_similarity(query_embedding, candidate_embedding)))
    return math.sqrt(max(0.0, 2.0 - (2.0 * cos)))


def semantic_score_from_embeddings(query_embedding: List[float], candidate_embedding: List[float]) -> Optional[float]:
    distance = semantic_distance_from_embeddings(query_embedding, candidate_embedding)
    return distance_to_score(distance)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    if len(vec_a) == 0 or len(vec_b) == 0:
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


def calibrate_cross_encoder_scores(scores: List[float]) -> List[float]:
    calibrated = []
    for score in scores:
        clipped = max(-12.0, min(12.0, float(score) / 3.0))
        calibrated.append(1.0 / (1.0 + math.exp(-clipped)))
    return calibrated


def extract_query_terms(query_text: str) -> List[str]:
    normalized_query = normalize_text_for_match(query_text)
    raw_terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", normalized_query, flags=re.UNICODE)

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
    pattern = rf"(?<![\w]){re.escape(term)}(?![\w])"
    return re.search(pattern, text, flags=re.UNICODE) is not None


def count_term_group_hits(text: str, term_groups: List[List[str]]) -> int:
    if not text or not term_groups:
        return 0
    hits = 0
    for variants in term_groups:
        if any(contains_exact_term(text, variant) for variant in variants):
            hits += 1
    return hits


def match_term_groups(text: str, term_groups: List[List[str]]) -> List[List[str]]:
    if not text or not term_groups:
        return []
    return [
        variants for variants in term_groups
        if any(contains_exact_term(text, variant) for variant in variants)
    ]


def query_matches_any_groups(normalized_query: str, raw_terms: List[str]) -> bool:
    return bool(match_term_groups(normalized_query, build_term_variant_groups(raw_terms)))


def detect_query_type(normalized_query: str) -> str:
    instrument_specific_hits = count_term_group_hits(
        normalized_query,
        build_term_variant_groups(INSTRUMENT_DETECTION_SPECIFIC_TERMS),
    )
    instrument_broad_hits = count_term_group_hits(
        normalized_query,
        build_term_variant_groups(INSTRUMENT_DETECTION_BROAD_TERMS),
    )
    if instrument_specific_hits >= 1 or instrument_broad_hits >= 2:
        return "instrument"
    if query_matches_any_groups(normalized_query, QUALITATIVE_DETECTION_TERMS):
        return "qualitative"
    if query_matches_any_groups(normalized_query, EVIDENCE_DETECTION_TERMS):
        return "evidence"
    return "general"


def contains_query_phrase(query_analysis: dict, text: str) -> bool:
    if not text:
        return False
    normalized_query = query_analysis.get("normalized_query", "")
    if not normalized_query or " " not in normalized_query:
        return False
    normalized_query_alt = normalized_query.replace("-", " ")
    if normalized_query in text:
        return True
    return normalized_query_alt != normalized_query and normalized_query_alt in text


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


def get_text_for_paper(
    meta_record: dict,
    fulltext_dir: str = FULLTEXT_DIR,
    preloaded_full_text: Optional[str] = None,
) -> Tuple[str, str, str, int, str]:
    title = (meta_record.get("title") or "").strip()
    paper_id = (meta_record.get("paperId") or "").strip()
    year = meta_record.get("year") or 0
    abstract_text = get_abstract_text(meta_record)

    full_text = preloaded_full_text if preloaded_full_text is not None else load_full_text(paper_id, fulltext_dir=fulltext_dir)
    if full_text:
        return title, abstract_text, full_text, year, "fulltext"

    return title, abstract_text, abstract_text or "No abstract provided", year, "abstract"


def build_chunk_records(
    meta_record: dict,
    fulltext_dir: str = FULLTEXT_DIR,
    preloaded_full_text: Optional[str] = None,
) -> List[dict]:
    title, abstract_text, text_for_index, year, text_source = get_text_for_paper(
        meta_record,
        fulltext_dir=fulltext_dir,
        preloaded_full_text=preloaded_full_text,
    )
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
    lexical_terms = list(base_terms)

    for phrase, variants in DOMAIN_EXPANSIONS.items():
        normalized_phrase = normalize_text_for_match(phrase)
        if normalized_phrase in normalized_query or any(term == normalized_phrase for term in base_terms):
            lexical_terms.extend(extract_query_terms(" ".join([phrase] + variants)))

    detected_query_type = detect_query_type(normalized_query)

    intent_terms = list(QUERY_TYPE_CONFIG[detected_query_type]["intent_terms"])  # copy — never mutate the global config
    lexical_terms = dedupe_preserve_order(lexical_terms)
    expanded_terms = dedupe_preserve_order(lexical_terms + intent_terms)
    base_term_groups = build_term_variant_groups(base_terms)
    lexical_term_groups = build_term_variant_groups(lexical_terms)

    return {
        "raw_query": query_text,
        "normalized_query": normalized_query,
        "query_type": detected_query_type,
        "base_terms": base_terms,
        "base_term_groups": base_term_groups,
        "lexical_terms": lexical_terms,
        "lexical_term_groups": lexical_term_groups,
        "expanded_terms": expanded_terms,
        "hybrid_weights": dict(QUERY_TYPE_CONFIG[detected_query_type]["hybrid_weights"]),  # copy
        "section_boosts": dict(QUERY_TYPE_CONFIG[detected_query_type]["section_boosts"]),  # copy
        "intent_terms": intent_terms,
    }


def compute_field_match_score(query_analysis: dict, text: str, field_name: str) -> float:
    normalized_text = normalize_text_for_match(text)
    if not normalized_text:
        return 0.0

    term_groups = query_analysis.get("lexical_term_groups") or query_analysis.get("base_term_groups") or []
    if not term_groups:
        return 0.0

    matched_term_groups = match_term_groups(normalized_text, term_groups)
    if not matched_term_groups:
        return 0.0

    field_weights = {"title": 0.34, "abstract": 0.22, "body": 0.14}
    phrase_weight = {"title": 0.20, "abstract": 0.14, "body": 0.08}
    hit_weight = {"title": 0.05, "abstract": 0.035, "body": 0.025}

    coverage = len(matched_term_groups) / max(1, len(term_groups))
    score = field_weights.get(field_name, 0.14) * coverage
    score += hit_weight.get(field_name, 0.025) * min(FIELD_HIT_CAPS.get(field_name, 2), len(matched_term_groups))

    normalized_query = query_analysis["normalized_query"]
    normalized_query_alt = normalized_query.replace("-", " ")
    if normalized_query and " " in normalized_query:
        if normalized_query in normalized_text:
            score += phrase_weight.get(field_name, 0.08)
        if normalized_query_alt != normalized_query and normalized_query_alt in normalized_text:
            score += phrase_weight.get(field_name, 0.08) * 0.7

    return min(FIELD_SCORE_CAPS.get(field_name, 0.18), score)


def compute_query_type_boost(query_analysis: dict, record: dict) -> float:
    intent_terms = query_analysis["intent_terms"]
    if not intent_terms:
        return 0.0

    searchable = " ".join([
        record.get("title", ""),
        record.get("abstract_text", ""),
    ])
    normalized_searchable = normalize_text_for_match(searchable)
    matches = {
        term for term in intent_terms
        if contains_exact_term(normalized_searchable, term)
    }
    if not matches:
        return 0.0
    return min(QUERY_TYPE_BOOST_CAP, len(matches) * QUERY_TYPE_BOOST_PER_TERM)


def compute_section_boost(query_analysis: dict, section: str) -> float:
    normalized_section = normalize_section_name(section)
    return SECTION_BASE_BOOSTS.get(normalized_section, 0.03) + query_analysis["section_boosts"].get(normalized_section, 0.0)


def compute_source_boost(text_source: str) -> float:
    if text_source == "fulltext":
        return 0.06
    return 0.01


def passes_strict_lexical_gate(query_analysis: dict, candidate: dict, min_score: Optional[float]) -> bool:
    term_groups = query_analysis.get("base_term_groups") or query_analysis.get("lexical_term_groups") or []
    if not term_groups:
        return False

    normalized_title = normalize_text_for_match(candidate.get("title", ""))
    normalized_abstract = normalize_text_for_match(candidate.get("abstract_text", ""))
    normalized_body = normalize_text_for_match(candidate.get("text", ""))

    title_hits = count_term_group_hits(normalized_title, term_groups)
    abstract_hits = count_term_group_hits(normalized_abstract, term_groups)
    body_hits = count_term_group_hits(normalized_body, term_groups)
    support_hits = abstract_hits + body_hits

    embedding_score = candidate.get("embedding_score", 0.0) or 0.0
    semantic_floor = min_score if min_score is not None else 0.25
    soft_semantic_floor = max(STRICT_LEXICAL_MIN_EMBEDDING, semantic_floor * STRICT_LEXICAL_SOFT_RATIO)
    if embedding_score < soft_semantic_floor:
        return False

    if contains_query_phrase(query_analysis, normalized_title):
        return True
    if title_hits >= STRICT_LEXICAL_MIN_TITLE_HITS and support_hits >= STRICT_LEXICAL_MIN_SUPPORT_HITS:
        return True
    if title_hits >= 1 and (
        contains_query_phrase(query_analysis, normalized_abstract) or
        contains_query_phrase(query_analysis, normalized_body)
    ):
        return True
    return False


def score_exact_match_rerank(query_analysis: dict, doc: str, meta: dict, embedding_score: Optional[float]) -> float:
    semantic_score = embedding_score or 0.0
    term_groups = query_analysis.get("base_term_groups") or []
    if not term_groups:
        return semantic_score

    normalized_query = query_analysis.get("normalized_query", "")
    normalized_query_alt = normalized_query.replace("-", " ")
    normalized_title = normalize_text_for_match((meta or {}).get("title", ""))
    normalized_doc = normalize_text_for_match(doc)

    title_matches = match_term_groups(normalized_title, term_groups)
    doc_matches = match_term_groups(normalized_doc, term_groups)
    combined_matches = []
    seen_keys = set()
    for variants in title_matches + doc_matches:
        key = tuple(variants)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        combined_matches.append(variants)

    title_hits = len(title_matches)
    doc_hits = len(doc_matches)

    coverage_bonus = len(combined_matches) / len(term_groups) if term_groups else 0.0

    phrase_bonus = 0.0
    if normalized_query and " " in normalized_query:
        if normalized_query in normalized_title:
            phrase_bonus += 0.30
        if normalized_query in normalized_doc:
            phrase_bonus += 0.18
        if normalized_query_alt != normalized_query:
            if normalized_query_alt in normalized_title:
                phrase_bonus += 0.18
            if normalized_query_alt in normalized_doc:
                phrase_bonus += 0.10

    title_bonus = title_hits * 0.12
    doc_bonus = doc_hits * 0.05

    return semantic_score + (coverage_bonus * 0.35) + phrase_bonus + title_bonus + doc_bonus


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
            body_tokens = tokenize_lexical(record.get("text", ""))
            search_tokens = title_tokens + body_tokens

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

        candidates = []

        for record in self.records:
            scored = self.score_record(query_analysis, record)
            if scored["lexical_score"] <= 0:
                continue

            candidates.append({
                "chunk_id": record["chunk_id"],
                **scored,
            })

        candidates.sort(key=lambda item: (item["lexical_score"], item["bm25_score"]), reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["bm25_rank"] = rank
        return candidates[:limit]

    def score_record(self, query_analysis: dict, record: dict) -> dict:
        query_terms = tokenize_lexical(" ".join(query_analysis.get("lexical_terms") or query_analysis["base_terms"]))
        query_terms = dedupe_preserve_order(query_terms)
        if not query_terms:
            return {
                "bm25_score": 0.0,
                "lexical_score": 0.0,
                "title_field_score": 0.0,
                "abstract_field_score": 0.0,
                "body_field_score": 0.0,
            }

        chunk_id = record["chunk_id"]
        tf = self.search_tf.get(chunk_id, Counter())
        if not tf:
            return {
                "bm25_score": 0.0,
                "lexical_score": 0.0,
                "title_field_score": 0.0,
                "abstract_field_score": 0.0,
                "body_field_score": 0.0,
            }

        doc_count = len(self.records)
        avg_doc_length = self.avg_doc_length or 1.0
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

        return {
            "bm25_score": bm25_score,
            "lexical_score": bm25_score + title_score + abstract_score + body_score,
            "title_field_score": title_score,
            "abstract_field_score": abstract_score,
            "body_field_score": body_score,
        }


class HybridCollectionProxy:
    def __init__(self, collection, lexical_index: Optional[LexicalIndex], min_score: Optional[float], oversample_factor: int = QUERY_OVERSAMPLE_FACTOR):
        self._collection = collection
        self._lexical_index = lexical_index
        self._min_score = min_score if min_score is not None and min_score > 0 else None
        self._oversample_factor = max(1, int(oversample_factor))

    def __getattr__(self, name):
        return getattr(self._collection, name)

    def _fallback_vector_query(self, query_embeddings: List[List[float]], n_results: int, include: List[str], query_texts: List[str]):
        candidate_pool = max(n_results, n_results * self._oversample_factor)
        requested_include = list(dict.fromkeys(list(include) + ["embeddings"]))
        try:
            raw_result = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=candidate_pool,
                include=requested_include,
            )
        except Exception:
            raw_result = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=candidate_pool,
                include=include,
            )

        ids_by_query = raw_result.get("ids", [])
        documents_by_query = raw_result.get("documents", [])
        metadatas_by_query = raw_result.get("metadatas", [])
        distances_by_query = raw_result.get("distances", [])
        embeddings_by_query = raw_result.get("embeddings", [])

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
            row_embeddings = embeddings_by_query[q_idx] if q_idx < len(embeddings_by_query) else []
            analysis = build_query_analysis(query_text)
            retrieval_notes = ["vector-only fallback"]
            fallback_embeddings = fetch_embeddings_by_id(
                self._collection,
                [
                    chunk_id for idx, chunk_id in enumerate(row_ids)
                    if not (idx < len(row_embeddings) and row_embeddings[idx] is not None)
                ],
            )

            keep = []
            for i, _chunk_id in enumerate(row_ids):
                candidate_embedding = row_embeddings[i] if i < len(row_embeddings) else None
                if candidate_embedding is None:
                    candidate_embedding = fallback_embeddings.get(row_ids[i])
                embedding_score = semantic_score_from_embeddings(query_embeddings[q_idx], candidate_embedding)
                if self._min_score is None or embedding_score is None or embedding_score >= self._min_score:
                    keep.append(i)

            row_ids = [row_ids[i] for i in keep]
            row_docs = [row_docs[i] for i in keep if i < len(row_docs)]
            row_metas = [row_metas[i] for i in keep if i < len(row_metas)]
            row_embeddings = [row_embeddings[i] if i < len(row_embeddings) else None for i in keep]

            row_missing_embeddings = [
                chunk_id for idx, chunk_id in enumerate(row_ids)
                if not (idx < len(row_embeddings) and row_embeddings[idx] is not None)
            ]
            if row_missing_embeddings:
                fallback_embeddings.update(fetch_embeddings_by_id(self._collection, row_missing_embeddings))
            row_embedding_scores = [
                semantic_score_from_embeddings(
                    query_embeddings[q_idx],
                    row_embeddings[i] if i < len(row_embeddings) and row_embeddings[i] is not None else fallback_embeddings.get(row_ids[i]),
                )
                for i in range(len(row_ids))
            ]
            row_distances = [
                semantic_distance_from_embeddings(
                    query_embeddings[q_idx],
                    row_embeddings[i] if i < len(row_embeddings) and row_embeddings[i] is not None else fallback_embeddings.get(row_ids[i]),
                )
                for i in range(len(row_ids))
            ]
            row_final_scores = [
                score_exact_match_rerank(
                    query_analysis=analysis,
                    doc=row_docs[i] if i < len(row_docs) else "",
                    meta=row_metas[i] if i < len(row_metas) and isinstance(row_metas[i], dict) else {},
                    embedding_score=row_embedding_scores[i] if i < len(row_embedding_scores) else None,
                )
                for i in range(len(row_ids))
            ]

            ranked_indices = list(range(len(row_ids)))
            ranked_indices.sort(
                key=lambda i: (
                    row_final_scores[i],
                    row_embedding_scores[i] if row_embedding_scores[i] is not None else 0.0,
                ),
                reverse=True,
            )

            filtered_indices = []
            seen_papers = set()
            for i in ranked_indices:
                meta = row_metas[i] if i < len(row_metas) and isinstance(row_metas[i], dict) else {}
                paper_id = meta.get("paperId") or row_ids[i]
                if paper_id in seen_papers:
                    continue
                seen_papers.add(paper_id)
                filtered_indices.append(i)

            ranked_indices = filtered_indices[:n_results]
            if ranked_indices and analysis.get("base_terms"):
                retrieval_notes.extend(["exact-term rerank", "paper dedupe"])

            out["ids"].append([row_ids[i] for i in ranked_indices])
            out["documents"].append([row_docs[i] for i in ranked_indices if i < len(row_docs)])
            out["distances"].append([row_distances[i] for i in ranked_indices if i < len(row_distances)])
            embedding_scores = [
                row_embedding_scores[i] if i < len(row_embedding_scores) and row_embedding_scores[i] is not None else 0.0
                for i in ranked_indices
            ]
            final_scores = [
                row_final_scores[i] if i < len(row_final_scores) else 0.0
                for i in ranked_indices
            ]
            metadata_rows = []
            for list_idx, row_idx in enumerate(ranked_indices):
                metadata_rows.append(
                    enrich_metadata_with_scores(
                        row_metas[row_idx] if row_idx < len(row_metas) and isinstance(row_metas[row_idx], dict) else {},
                        cross_encoder_score=None,
                        final_score=final_scores[list_idx] if list_idx < len(final_scores) else 0.0,
                    )
                )
            out["metadatas"].append(metadata_rows)
            out["embedding_scores"].append(embedding_scores)
            out["bm25_scores"].append([0.0 for _ in ranked_indices])
            out["hybrid_scores"].append(list(final_scores))
            out["paper_scores"].append(list(final_scores))
            out["cross_encoder_scores"].append([None for _ in ranked_indices])
            out["mmr_scores"].append(list(final_scores))
            out["final_scores"].append(list(final_scores))
            out["query_analysis"].append(analysis)
            out["retrieval_notes"].append(retrieval_notes)

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
        try:
            raw_vector_result = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=candidate_pool,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception:
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
        "embedding": None,
        "distance": None,
        "vector_rank": None,
        "bm25_rank": None,
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


def fetch_embeddings_by_id(collection, candidate_ids: List[str]) -> Dict[str, List[float]]:
    if collection is None or not candidate_ids or not hasattr(collection, "get"):
        return {}

    try:
        payload = collection.get(ids=candidate_ids, include=["embeddings"])
    except Exception:
        return {}

    ids = payload.get("ids", []) if isinstance(payload, dict) else []
    embeddings = payload.get("embeddings", []) if isinstance(payload, dict) else []
    return {
        chunk_id: embeddings[idx]
        for idx, chunk_id in enumerate(ids)
        if idx < len(embeddings) and embeddings[idx] is not None
    }


def enrich_candidates_from_collection(collection, lexical_index: LexicalIndex, candidate_ids: List[str]) -> Dict[str, dict]:
    if collection is None or not candidate_ids or not hasattr(collection, "get"):
        return {}

    try:
        payload = collection.get(ids=candidate_ids, include=["documents", "metadatas", "embeddings"])
    except Exception:
        try:
            payload = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
        except Exception:
            return {}

    ids = payload.get("ids", []) if isinstance(payload, dict) else []
    documents = payload.get("documents", []) if isinstance(payload, dict) else []
    metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
    embeddings = payload.get("embeddings", []) if isinstance(payload, dict) else []

    enriched: Dict[str, dict] = {}
    for idx, chunk_id in enumerate(ids):
        base_record = lexical_index.record_by_id.get(chunk_id)
        if not base_record:
            continue

        meta = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        doc = documents[idx] if idx < len(documents) else base_record.get("text", "")
        embedding = embeddings[idx] if idx < len(embeddings) else None

        enriched[chunk_id] = {
            "text": doc or base_record.get("text", ""),
            "embedding": embedding,
            "paperId": meta.get("paperId") or base_record.get("paperId"),
            "title": meta.get("title") or base_record.get("title", ""),
            "url": meta.get("url") or base_record.get("url", ""),
            "year": meta.get("year") if meta.get("year") is not None else base_record.get("year"),
            "text_source": meta.get("text_source") or base_record.get("text_source", "unknown"),
            "section": meta.get("section") or base_record.get("section", "body"),
        }

    return enriched


def aggregate_papers(candidates: List[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for candidate in candidates:
        paper_id = candidate.get("paperId") or candidate["chunk_id"]
        grouped[paper_id].append(candidate)

    paper_entries = []
    for paper_id, paper_candidates in grouped.items():
        paper_candidates.sort(key=lambda item: item["chunk_score"], reverse=True)
        best_candidate = paper_candidates[0]
        best_score = max(best_candidate["chunk_score"], 1e-9)
        support_candidates = []
        for candidate in paper_candidates[1:]:
            if len(support_candidates) >= PAPER_SUPPORT_MAX_CHUNKS:
                break
            if candidate["chunk_score"] < best_score * PAPER_SUPPORT_MIN_RELATIVE_SCORE:
                continue
            support_candidates.append(candidate)

        support_bonus = min(
            PAPER_SUPPORT_MAX_BONUS,
            sum(
                min(PAPER_SUPPORT_PER_CHUNK, (candidate["chunk_score"] / best_score) * PAPER_SUPPORT_PER_CHUNK)
                for candidate in support_candidates
            ),
        )
        unique_sections = {
            normalize_section_name(best_candidate.get("section", "body"))
        }
        unique_sections.update(
            normalize_section_name(candidate.get("section", "body"))
            for candidate in support_candidates
        )
        section_diversity_bonus = min(
            PAPER_SECTION_DIVERSITY_MAX,
            max(0, len(unique_sections) - 1) * PAPER_SECTION_DIVERSITY_BONUS,
        )
        support_bonus += section_diversity_bonus
        paper_score = best_candidate["chunk_score"] + support_bonus

        representative = dict(best_candidate)
        representative["paper_score"] = paper_score
        representative["supporting_chunks"] = len(paper_candidates)
        representative["support_bonus"] = support_bonus
        representative["supporting_sections"] = sorted(unique_sections)
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
    calibrated_scores = calibrate_cross_encoder_scores([float(score) for score in raw_scores])

    for idx, calibrated_score in enumerate(calibrated_scores):
        papers[idx]["cross_encoder_score"] = calibrated_score
        papers[idx]["paper_score"] += CROSS_ENCODER_BLEND_WEIGHT * calibrated_score

    papers.sort(key=lambda item: (item["paper_score"], item["chunk_score"]), reverse=True)
    return "cross-encoder applied"


def apply_mmr_selection(papers: List[dict], k: int) -> List[dict]:
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
    metadatas = []
    final_scores = []
    for paper in selected_papers:
        final_score = paper.get("mmr_score") if paper.get("mmr_score") is not None else paper.get("paper_score", 0.0)
        final_scores.append(final_score)
        metadatas.append(
            enrich_metadata_with_scores(
                {
                    "paperId": paper.get("paperId"),
                    "title": paper.get("title"),
                    "url": paper.get("url"),
                    "year": paper.get("year"),
                    "text_source": paper.get("text_source"),
                    "section": paper.get("section"),
                    "supporting_chunks": paper.get("supporting_chunks", 1),
                },
                cross_encoder_score=paper.get("cross_encoder_score"),
                final_score=final_score,
            )
        )

    return {
        "ids": [paper["chunk_id"] for paper in selected_papers],
        "documents": [paper["text"] for paper in selected_papers],
        "metadatas": metadatas,
        "distances": [paper.get("distance") for paper in selected_papers],
        "embedding_scores": [paper.get("embedding_score", 0.0) for paper in selected_papers],
        "bm25_scores": [paper.get("bm25_score", 0.0) for paper in selected_papers],
        "hybrid_scores": [paper.get("hybrid_score", 0.0) for paper in selected_papers],
        "paper_scores": [paper.get("paper_score", 0.0) for paper in selected_papers],
        "cross_encoder_scores": [paper.get("cross_encoder_score") for paper in selected_papers],
        "mmr_scores": [paper.get("mmr_score") for paper in selected_papers],
        "final_scores": final_scores,
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
    embeddings_by_query = raw_vector_result.get("embeddings", [])

    for q_idx, query_text in enumerate(query_texts):
        query_analysis = build_query_analysis(query_text)
        retrieval_notes = [
            "hybrid retrieval",
            f"query_type={query_analysis['query_type']}",
            "hybrid candidate generation",
        ]

        row_ids = ids_by_query[q_idx] if q_idx < len(ids_by_query) else []
        row_docs = documents_by_query[q_idx] if q_idx < len(documents_by_query) else []
        row_metas = metadatas_by_query[q_idx] if q_idx < len(metadatas_by_query) else []
        row_embeddings = embeddings_by_query[q_idx] if q_idx < len(embeddings_by_query) else []
        fallback_embeddings = fetch_embeddings_by_id(
            collection,
            [
                chunk_id for idx, chunk_id in enumerate(row_ids)
                if not (idx < len(row_embeddings) and row_embeddings[idx] is not None)
            ],
        )

        candidate_by_id: Dict[str, dict] = {}
        lexical_only_ids: List[str] = []

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
            candidate_embedding = row_embeddings[vector_rank - 1] if vector_rank - 1 < len(row_embeddings) else None
            if candidate_embedding is None:
                candidate_embedding = fallback_embeddings.get(chunk_id)
            candidate["embedding"] = candidate_embedding
            candidate["distance"] = semantic_distance_from_embeddings(query_embeddings[q_idx], candidate_embedding)
            candidate["embedding_score"] = semantic_score_from_embeddings(query_embeddings[q_idx], candidate_embedding) or 0.0
            candidate["vector_rank"] = vector_rank
            lexical_scores = lexical_index.score_record(query_analysis, base_record)
            candidate["bm25_score"] = lexical_scores.get("bm25_score", 0.0)
            candidate["lexical_score"] = lexical_scores.get("lexical_score", 0.0)
            candidate["title_field_score"] = lexical_scores.get("title_field_score", 0.0)
            candidate["abstract_field_score"] = lexical_scores.get("abstract_field_score", 0.0)
            candidate["body_field_score"] = lexical_scores.get("body_field_score", 0.0)
            candidate_by_id[chunk_id] = candidate

        lexical_limit = max(k * QUERY_OVERSAMPLE_FACTOR, LEXICAL_CANDIDATE_POOL)
        lexical_candidates = lexical_index.search(query_analysis, limit=lexical_limit)
        for lexical_candidate in lexical_candidates:
            chunk_id = lexical_candidate["chunk_id"]
            base_record = lexical_index.record_by_id.get(chunk_id)
            if not base_record:
                continue

            candidate = candidate_by_id.get(chunk_id)
            if candidate is None:
                candidate = build_candidate_record(base_record)
                candidate_by_id[chunk_id] = candidate
                lexical_only_ids.append(chunk_id)

            candidate["bm25_score"] = lexical_candidate.get("bm25_score", 0.0)
            candidate["lexical_score"] = lexical_candidate.get("lexical_score", 0.0)
            candidate["title_field_score"] = lexical_candidate.get("title_field_score", 0.0)
            candidate["abstract_field_score"] = lexical_candidate.get("abstract_field_score", 0.0)
            candidate["body_field_score"] = lexical_candidate.get("body_field_score", 0.0)
            candidate["bm25_rank"] = lexical_candidate.get("bm25_rank")

        if lexical_only_ids:
            enriched = enrich_candidates_from_collection(collection, lexical_index, lexical_only_ids)
            for chunk_id in lexical_only_ids:
                candidate = candidate_by_id.get(chunk_id)
                details = enriched.get(chunk_id)
                if not candidate or not details:
                    continue
                candidate["text"] = details.get("text", candidate.get("text", ""))
                candidate["embedding"] = details.get("embedding")
                candidate["paperId"] = details.get("paperId", candidate.get("paperId"))
                candidate["title"] = details.get("title", candidate.get("title", ""))
                candidate["url"] = details.get("url", candidate.get("url", ""))
                candidate["year"] = details.get("year", candidate.get("year"))
                candidate["text_source"] = details.get("text_source", candidate.get("text_source", "unknown"))
                candidate["section"] = details.get("section", candidate.get("section", "body"))

        if not candidate_by_id:
            empty_row = build_result_row([], query_analysis, retrieval_notes + ["no candidates"])
            for key, value in empty_row.items():
                out[key].append(value)
            continue

        candidates = list(candidate_by_id.values())
        for candidate in candidates:
            candidate["distance"] = semantic_distance_from_embeddings(query_embeddings[q_idx], candidate.get("embedding"))
            candidate["embedding_score"] = semantic_score_from_embeddings(query_embeddings[q_idx], candidate.get("embedding")) or 0.0
        normalized_embedding_scores = normalize_scores([candidate["embedding_score"] for candidate in candidates])
        normalized_bm25_scores = normalize_scores([candidate["bm25_score"] for candidate in candidates])

        embedding_weight = query_analysis["hybrid_weights"]["embedding"]
        bm25_weight = query_analysis["hybrid_weights"]["bm25"]

        scored_candidates = []
        rescued_candidates = 0
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
            if min_score is None or candidate["embedding_score"] >= min_score:
                scored_candidates.append(candidate)
                continue
            if passes_strict_lexical_gate(query_analysis, candidate, min_score=min_score):
                scored_candidates.append(candidate)
                rescued_candidates += 1

        if not scored_candidates:
            empty_row = build_result_row([], query_analysis, retrieval_notes + ["all candidates filtered by min_score"])
            for key, value in empty_row.items():
                out[key].append(value)
            continue

        retrieval_notes.append(
            f"candidate union: vector={len(row_ids)} lexical={len(lexical_candidates)} merged={len(candidates)}"
        )
        if rescued_candidates:
            retrieval_notes.append(f"strict lexical rescue={rescued_candidates}")

        scored_candidates.sort(key=lambda item: (item["chunk_score"], item["embedding_score"], item["bm25_score"]), reverse=True)
        paper_entries = aggregate_papers(scored_candidates)
        retrieval_notes.append("paper-level aggregation")

        cross_encoder_note = apply_cross_encoder(query_analysis, paper_entries)
        if cross_encoder_note:
            retrieval_notes.append(cross_encoder_note)

        selected_papers = apply_mmr_selection(paper_entries, k)
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
    manifest = RunManifest("retrieval_index")
    lexical_index_existed = os.path.exists(LEXICAL_INDEX_FILE)
    chroma_dir_existed = os.path.exists(CHROMA_DIR)

    filtered = load_json(filtered_file)
    metadata = load_json(metadata_file)
    meta_index = build_metadata_index(metadata)
    allowed_set = {p.get("paperId") for p in filtered if p.get("paperId")}
    allowed_ids = sorted(allowed_set)

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

        full_text = load_full_text(pid, fulltext_dir=FULLTEXT_DIR)
        text_source = "fulltext" if full_text else "abstract"
        if text_source == "fulltext":
            fulltext_papers += 1
        else:
            abstract_fallback_papers += 1

        records = build_chunk_records(
            meta_rec,
            fulltext_dir=FULLTEXT_DIR,
            preloaded_full_text=full_text,
        )
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

    manifest.add_event(
        "updated_index" if chroma_dir_existed else "created_index",
        CHROMA_DIR,
        {
            "collection": COLLECTION_NAME,
            "chunk_count": total_chunks,
        },
    )
    manifest.add_event(
        "updated" if lexical_index_existed else "created",
        LEXICAL_INDEX_FILE,
        {"record_count": len(lexical_records)},
    )

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

    manifest.set_summary(
        metadata_path=os.path.relpath(metadata_file, BASE_DIR),
        filtered_path=os.path.relpath(filtered_file, BASE_DIR),
        allowed_papers=len(allowed_set),
        missing_in_metadata=missing_in_metadata,
        fulltext_papers=fulltext_papers,
        abstract_fallback_papers=abstract_fallback_papers,
        total_chunks=total_chunks,
        lexical_records=len(lexical_records),
        collection_size=collection_count(collection),
    )
    manifest_path = manifest.write()
    print(f"Run manifest written to: {manifest_path}")


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
        final_score = final_scores[i] if i < len(final_scores) else None
        out["results"].append({
            "chunk_id": cid,
            "score": final_score,
            "final_score": final_score,
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

    dump_json_console(out, ensure_ascii=False, indent=2)


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
    dump_json_console(out, ensure_ascii=False, indent=2)


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
