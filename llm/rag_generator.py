import argparse
import json
import os
import re
import sys

# Add project root to path so sibling packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from console_utils import dump_json_console
from Retrieval.retrieval import (
    CHROMA_DIR,
    get_chroma_collection,
    get_embedding_model,
    require_retrieval_dependencies,
)
from llm.interface import get_llm_provider

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence in the retrieved corpus to answer this question."
PLACEHOLDER_CITATION_RE = re.compile(r"([\[(])\s*Papers?\s+([0-9,\sand&]+)\s*([\])])", re.IGNORECASE)
RETRIEVAL_SCORE_FIELDS = (
    "embedding_score",
    "cross_encoder_score",
)


def _get_score_from_result_or_metadata(score_lists: dict, score_name: str, index: int, metadata: dict):
    values = score_lists.get(score_name) or []
    if index < len(values) and values[index] is not None:
        return values[index]

    retrieval_scores = metadata.get("retrieval_scores")
    if isinstance(retrieval_scores, dict) and retrieval_scores.get(score_name) is not None:
        return retrieval_scores.get(score_name)

    return metadata.get(score_name)


def _build_retrieval_scores(score_lists: dict, index: int, metadata: dict) -> dict:
    return {
        score_name: score_value
        for score_name in RETRIEVAL_SCORE_FIELDS
        for score_value in [_get_score_from_result_or_metadata(score_lists, score_name, index, metadata)]
        if score_value is not None
    }


def _build_context_and_sources(retrieval_result: dict):
    documents = retrieval_result.get("documents", [[]])[0]
    metadatas = retrieval_result.get("metadatas", [[]])[0]
    score_lists = {
        score_name: retrieval_result.get(f"{score_name}s", [[]])[0]
        for score_name in RETRIEVAL_SCORE_FIELDS
    }

    sources = []
    context_blocks = []

    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        meta = meta or {}
        paper_id = (meta.get("paperId") or "").strip()
        title = meta.get("title") or "Unknown"
        url = meta.get("url") or "N/A"
        year = meta.get("year") or "N/A"
        text_source = meta.get("text_source") or "unknown"
        section = meta.get("section") or "unknown"
        supporting_chunks = meta.get("supporting_chunks") or 1
        retrieval_scores = _build_retrieval_scores(score_lists, idx - 1, meta)

        source = {
            "rank": idx,
            "title": title,
            "url": url,
            "year": year,
            "paperId": paper_id,
            "text_source": text_source,
            "text": doc,
            "section": section,
            "supporting_chunks": supporting_chunks,
            "semantic_similarity": retrieval_scores.get("embedding_score"),
            "cross_encoder_score": retrieval_scores.get("cross_encoder_score"),
            "keyword_overlap": meta.get("keyword_overlap"),
        }
        sources.append(source)

        context_lines = [
            f"[Source {idx}]",
            f"paperId: {paper_id or 'N/A'}",
            f"title: {title}",
            f"year: {year}",
            f"url: {url}",
            f"text_source: {text_source}",
            f"section: {section}",
            f"supporting_chunks: {supporting_chunks}",
        ]
        context_lines.append(f"excerpt:\n{doc}")
        context_blocks.append("\n".join(context_lines))

    return "\n\n".join(context_blocks), sources


def _build_source_index(sources: list) -> str:
    lines = ["References:"]
    for source in sources:
        rank = source.get("rank", "?")
        title = source.get("title") or "Unknown"
        year = source.get("year") or "N/A"
        lines.append(f"[{rank}] {title} ({year})")
    return "\n".join(lines)


def _replace_placeholder_citations(answer: str, sources: list) -> str:
    def repl(match):
        paper_numbers = [int(num) for num in re.findall(r"\d+", match.group(2))]
        valid_numbers = [n for n in paper_numbers if 1 <= n <= len(sources)]
        if not valid_numbers:
            return match.group(0)
        return "[" + ", ".join(str(n) for n in valid_numbers) + "]"

    return PLACEHOLDER_CITATION_RE.sub(repl, answer)


def _answer_has_allowed_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer))


REF_SECTION_RE = re.compile(
    r"((?:^|\n)[ \t]*(?:\d+[\.\)]\s*)?References[ \t]*\n)(.*?)$",
    re.IGNORECASE | re.DOTALL,
)


def _rebuild_references_section(answer: str, sources: list) -> str:
    """Replace the bare-numbered References section with titled entries."""
    match = REF_SECTION_RE.search(answer)
    if not match:
        return answer

    body = answer[: match.start()]
    cited = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", body)))

    header = match.group(1).rstrip()
    entries = []
    for n in cited:
        idx = n - 1
        if 0 <= idx < len(sources):
            title = sources[idx].get("title") or "Unknown"
            year = sources[idx].get("year") or "N/A"
            entries.append(f"[{n}] {title} ({year})")

    return body + header + "\n\n" + "\n\n".join(entries)


def _normalize_text_answer(answer: str, sources: list) -> str:
    normalized = (answer or "").strip()
    if not normalized:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    normalized = _replace_placeholder_citations(normalized, sources)
    if sources:
        normalized = _rebuild_references_section(normalized, sources)
    if sources and not _answer_has_allowed_citation(normalized):
        normalized = f"{normalized}\n\n{_build_source_index(sources)}"
    return normalized


def _build_prompt(
    user_query: str,
    context_text: str,
    answer_template: str,
    output_mode: str,
    sources: list,
):
    n_sources = len(sources)

    base_system = f"""
    - Use ONLY the provided retrieved evidence.
    - Do NOT introduce external knowledge.
    - Do NOT provide clinical recommendations.
    - Do NOT extrapolate beyond the evidence.
    - If the retrieved evidence is empty or not directly relevant, respond exactly with: {INSUFFICIENT_EVIDENCE_MESSAGE}
    - Cite every substantive claim using numbered references matching the [Source N] number in the evidence, for example: [1] or [1, 3].
    - Only use reference numbers between 1 and {n_sources}.
    - Never use paperId values, DOIs, author-year references, or any other citation format.

    This system provides AI-assisted literature synthesis only.
    It does not replace professional judgment."""

    if answer_template == "structured":
        base_system += """

Use this structure:
1) Summary of the relevant topic (2-4 sentences)
2) Key findings (3-6 bullet points)
3) Limitations / gaps
4) Practical implications
5) References

Every sentence or bullet with a factual claim in sections 1-4 must include one or more numbered citations like [1] or [1, 2]."""

    if output_mode == "json":
        base_system += """

Return ONLY valid JSON with this schema:
{
  "summary": "string",
  "key_findings": ["string"],
  "limitations": ["string"],
  "practical_implications": ["string"],
  "citations_used": ["[1]", "[2]"]
}"""

    prompt = f"""Question: {user_query}

Retrieved evidence:
{context_text}

Please synthesize these papers into a coherent answer."""

    return base_system, prompt


def generate_rag_answer(
    user_query: str,
    provider: str = "openai",
    k: int = 5,
    answer_template: str = "default",
    output_mode: str = "text",
    **kwargs,
) -> dict:
    """
    Complete RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context
    3. Generate answer with LLM

    Args:
        user_query: User's question
        provider: "openai" or "ollama"
        k: Number of papers to retrieve
        answer_template: "default" or "structured"
        output_mode: "text" or "json"
        **kwargs: Additional args for LLM provider

    Returns:
        {"answer": str, "sources": list, "query": str}
    """
    if not user_query.strip():
        raise ValueError("Query must not be empty.")

    require_retrieval_dependencies()
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError("No index found. Run: python Retrieval/retrieval.py index")

    print(f"Retrieving {k} relevant papers...")
    collection = get_chroma_collection()
    model = get_embedding_model()

    q_emb = model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True).tolist()[0]

    retrieval_result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        query_text=user_query,
        include=["documents", "metadatas", "distances"],
    )

    context_text, sources = _build_context_and_sources(retrieval_result)
    retrieval_notes = retrieval_result.get("retrieval_notes", [[]])[0]

    if not sources:
        return {
            "query": user_query,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "provider": provider,
            "template": answer_template,
            "output_mode": output_mode,
            "retrieval_notes": retrieval_notes,
            "insufficient_evidence": True,
        }

    llm = get_llm_provider(provider, **kwargs)
    system_message, prompt = _build_prompt(
        user_query,
        context_text,
        answer_template,
        output_mode,
        sources,
    )

    print(f"Generating answer with {provider}...")
    raw_answer = llm.generate(prompt, system_message)

    answer = raw_answer
    parse_error = None
    if output_mode == "json":
        try:
            answer = json.loads(raw_answer)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
            answer = {"raw_text": raw_answer}
    else:
        answer = _normalize_text_answer(raw_answer, sources)

    result = {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "provider": provider,
        "template": answer_template,
        "output_mode": output_mode,
        "retrieval_notes": retrieval_notes,
        "insufficient_evidence": False,
    }

    if parse_error:
        result["parse_error"] = parse_error

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG answer with configurable templates")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("provider", type=str, choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved papers")
    parser.add_argument(
        "--template",
        type=str,
        choices=["default", "structured"],
        default="structured",
        help="Answer template style",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model name. For Ollama, examples: qwen3:14b, gemma3:12b.",
    )

    args = parser.parse_args()

    extra_kwargs = {}
    if args.model:
        extra_kwargs["model"] = args.model

    result = generate_rag_answer(
        args.query,
        provider=args.provider,
        k=args.k,
        answer_template=args.template,
        output_mode=args.output,
        **extra_kwargs,
    )
    dump_json_console(result, ensure_ascii=False, indent=2)
