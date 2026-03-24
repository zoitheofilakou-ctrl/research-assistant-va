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


if __name__ == "__main__":
    unittest.main()
