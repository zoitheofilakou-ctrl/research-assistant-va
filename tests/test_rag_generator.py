import unittest
from unittest.mock import Mock, patch

from llm import rag_generator


class FakeArray:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


class RagGeneratorTests(unittest.TestCase):
    def test_generate_rag_answer_returns_insufficient_evidence_without_calling_llm(self):
        fake_collection = Mock()
        fake_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "final_scores": [[]],
            "retrieval_notes": [["all candidates filtered by min_score"]],
        }
        fake_model = Mock()
        fake_model.encode.return_value = FakeArray([[0.1, 0.2]])

        with patch("llm.rag_generator.require_retrieval_dependencies"):
            with patch("llm.rag_generator.os.path.exists", return_value=True):
                with patch("llm.rag_generator.get_chroma_collection", return_value=fake_collection):
                    with patch("llm.rag_generator.get_embedding_model", return_value=fake_model):
                        with patch("llm.rag_generator.get_llm_provider") as get_llm_provider:
                            result = rag_generator.generate_rag_answer(
                                "blockchain in finance",
                                provider="ollama",
                            )

        self.assertEqual(rag_generator.INSUFFICIENT_EVIDENCE_MESSAGE, result["answer"])
        self.assertEqual([], result["sources"])
        self.assertTrue(result["insufficient_evidence"])
        get_llm_provider.assert_not_called()

    def test_normalize_text_answer_replaces_placeholder_citations(self):
        sources = [
            {"paperId": "paper-a", "title": "Paper A", "year": 2024},
            {"paperId": "paper-b", "title": "Paper B", "year": 2023},
        ]

        answer = rag_generator._normalize_text_answer(
            "Key finding one [Paper 1]. Comparative point [Papers 1 & 2].",
            sources,
        )

        self.assertIn("(paperId: paper-a)", answer)
        self.assertIn("(paperId: paper-a, paperId: paper-b)", answer)

    def test_normalize_text_answer_appends_real_source_index_when_missing(self):
        sources = [{"paperId": "paper-a", "title": "Paper A", "year": 2024}]

        answer = rag_generator._normalize_text_answer("Ungrounded-looking answer text.", sources)

        self.assertIn("Citations:", answer)
        self.assertIn("paperId: paper-a | Paper A (2024)", answer)

    def test_build_context_and_sources_propagates_retrieval_score_breakdown(self):
        context_text, sources = rag_generator._build_context_and_sources(
            {
                "documents": [["Evidence excerpt."]],
                "metadatas": [[{
                    "paperId": "paper-a",
                    "title": "Paper A",
                    "url": "https://example.org/paper-a",
                    "year": 2024,
                    "text_source": "fulltext",
                    "section": "results",
                    "supporting_chunks": 2,
                }]],
                "embedding_scores": [[0.91]],
                "bm25_scores": [[6.0]],
                "hybrid_scores": [[0.84]],
                "paper_scores": [[0.88]],
                "cross_encoder_scores": [[0.42]],
                "mmr_scores": [[0.93]],
                "final_scores": [[0.93]],
            }
        )

        self.assertIn("final_score: 0.930", context_text)
        self.assertEqual(0.93, sources[0]["final_score"])
        self.assertEqual(0.91, sources[0]["retrieval_scores"]["embedding_score"])
        self.assertEqual(6.0, sources[0]["retrieval_scores"]["bm25_score"])
        self.assertEqual(0.42, sources[0]["retrieval_scores"]["cross_encoder_score"])


if __name__ == "__main__":
    unittest.main()
