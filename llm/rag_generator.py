import json
import sys
import os
import argparse

# Add project root to path so sibling packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Retrieval.retrieval import (
    CHROMA_DIR,
    EMBED_MODEL_NAME,
    SentenceTransformer,
    get_chroma_collection,
    require_retrieval_dependencies,
)
from llm.interface import get_llm_provider

# Used to build rules for the model and the answer template
def _build_prompt(user_query: str, context_text: str, answer_template: str, output_mode: str):
    base_system = """
    - Use ONLY the provided retrieved evidence.
    - Do NOT introduce external knowledge.
    - Do NOT provide clinical recommendations.
    - Do NOT extrapolate beyond the evidence.
    - Cite paperId after each major claim.
    - If evidence is insufficient, explicitly state that.
    
    This system provides AI-assisted literature synthesis only.
    It does not replace professional judgment."""

    if answer_template == "structured":
        base_system += """

Use this structure:
1) Summary of the relevant topic (2-4 sentences)
2) Key findings (3-6 bullet points)
3) Limitations / gaps
4) Practical implications
5) Citations used"""

    if output_mode == "json":
        base_system += """

Return ONLY valid JSON with this schema:
{
  "summary": "string",
  "key_findings": ["string"],
  "limitations": ["string"],
  "practical_implications": ["string"],
  "citations_used": ["[Paper N]"]
}"""

    prompt = f"""Question: {user_query}

Retrieved papers:
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

    # Match retrieval.py runtime checks and fail with explicit messages.
    require_retrieval_dependencies()
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError("No index found. Run: python Retrieval/retrieval.py index")

    # Get LLM
    llm = get_llm_provider(provider, **kwargs)
    
    # Retrieve relevant papers
    print(f"Retrieving {k} relevant papers...")
    collection = get_chroma_collection()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    
    q_emb = model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True).tolist()[0]
    
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Build context from retrieved chunks
    sources = []
    context_text = ""
    
    for i, (doc, meta) in enumerate(zip(
        res.get("documents", [[]])[0],
        res.get("metadatas", [[]])[0]
    )):
        context_text += f"\n[Paper {i+1}] {meta.get('title', 'Unknown')}\n"
        context_text += f"URL: {meta.get('url', 'N/A')}\n"
        context_text += f"Year: {meta.get('year', 'N/A')}\n"
        context_text += f"Text source: {meta.get('text_source', 'unknown')}\n"
        context_text += f"Excerpt: {doc}\n"
        
        sources.append({
            "title": meta.get("title"),
            "url": meta.get("url"),
            "year": meta.get("year"),
            "paperId": meta.get("paperId"),
            "text_source": meta.get("text_source"),
        })
    
    # Generate answer
    system_message, prompt = _build_prompt(user_query, context_text, answer_template, output_mode)
    
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

    result = {
        "query": user_query,
        "answer": answer,
        "sources": sources,
        "provider": provider,
        "template": answer_template,
        "output_mode": output_mode,
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

    args = parser.parse_args()

    result = generate_rag_answer(
        args.query,
        provider=args.provider,
        k=args.k,
        answer_template=args.template,
        output_mode=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
