import unittest
from unittest.mock import patch

from Retrieval import retrieval


class FakeLexicalIndex:
    def __init__(self, records, scores_by_id):
        self.records = records
        self.record_by_id = {record["chunk_id"]: record for record in records}
        self._scores_by_id = scores_by_id

    def score_record(self, query_analysis, record):
        return dict(self._scores_by_id.get(record["chunk_id"], {
            "bm25_score": 0.0,
            "lexical_score": 0.0,
            "title_field_score": 0.0,
            "abstract_field_score": 0.0,
            "body_field_score": 0.0,
        }))

    def search(self, query_analysis, limit):
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


class FakeCollection:
    def __init__(self, records_by_id=None, embeddings_by_id=None):
        self._records_by_id = records_by_id or {}
        self._embeddings_by_id = embeddings_by_id or {}

    def get(self, ids, include=None):
        return {
            "ids": [chunk_id for chunk_id in ids if chunk_id in self._records_by_id],
            "documents": [self._records_by_id[chunk_id]["text"] for chunk_id in ids if chunk_id in self._records_by_id],
            "metadatas": [
                {
                    "paperId": self._records_by_id[chunk_id]["paperId"],
                    "title": self._records_by_id[chunk_id]["title"],
                    "url": self._records_by_id[chunk_id]["url"],
                    "year": self._records_by_id[chunk_id]["year"],
                    "text_source": self._records_by_id[chunk_id]["text_source"],
                    "section": self._records_by_id[chunk_id]["section"],
                }
                for chunk_id in ids if chunk_id in self._records_by_id
            ],
            "embeddings": [self._embeddings_by_id.get(chunk_id) for chunk_id in ids if chunk_id in self._records_by_id],
        }


class RetrievalTests(unittest.TestCase):
    def test_normalize_text_for_match_preserves_unicode_letters(self):
        normalized = retrieval.normalize_text_for_match("Café naïve β-blocker / implementation")

        self.assertIn("café", normalized)
        self.assertIn("naïve", normalized)
        self.assertIn("β-blocker", normalized)

    def test_singularize_term_keeps_irregular_words(self):
        self.assertEqual("analysis", retrieval.singularize_term("analysis"))
        self.assertEqual("diabetes", retrieval.singularize_term("diabetes"))
        self.assertEqual("basis", retrieval.singularize_term("basis"))

    def test_extract_query_terms_normalizes_expected_variants(self):
        terms = retrieval.extract_query_terms("evidence-based practices instruments")

        self.assertIn("evidence-based", terms)
        self.assertIn("evidence based", terms)
        self.assertIn("practice", terms)
        self.assertIn("instrument", terms)

    def test_term_group_hits_do_not_double_count_variants(self):
        groups = retrieval.build_term_variant_groups([
            "barrier", "barriers", "person-centered", "person centered"
        ])

        hits = retrieval.count_term_group_hits("person centered barrier in care", groups)

        self.assertEqual(2, hits)

    def test_build_query_analysis_separates_lexical_and_intent_terms(self):
        analysis = retrieval.build_query_analysis(
            "implementation person-centered care rehabilitation workforce barriers facilitators"
        )

        self.assertEqual("qualitative", analysis["query_type"])
        self.assertIn("implementation", analysis["lexical_terms"])
        self.assertIn("barriers", analysis["lexical_terms"])
        self.assertNotIn("qualitative", analysis["lexical_terms"])
        self.assertIn("qualitative", analysis["expanded_terms"])
        self.assertIn("interview", analysis["expanded_terms"])

    def test_detect_query_type_avoids_simple_substring_false_positive(self):
        analysis = retrieval.build_query_analysis("implementational planning for service delivery")

        self.assertEqual("general", analysis["query_type"])

    def test_compute_field_match_score_is_capped(self):
        analysis = retrieval.build_query_analysis(
            "implementation person-centered care rehabilitation workforce barriers facilitators"
        )
        text = (
            "implementation barriers facilitators person-centered care rehabilitation workforce "
            "implementation barriers facilitators person-centered care rehabilitation workforce"
        )

        score = retrieval.compute_field_match_score(analysis, text, "abstract")

        self.assertLessEqual(score, retrieval.FIELD_SCORE_CAPS["abstract"])

    def test_compute_query_type_boost_zero_when_no_matches(self):
        analysis = retrieval.build_query_analysis("implementation barriers facilitators")
        record = {"title": "Randomized rehabilitation outcomes", "abstract_text": ""}

        self.assertEqual(0.0, retrieval.compute_query_type_boost(analysis, record))

    def test_compute_query_type_boost_capped_on_title_and_abstract_only(self):
        analysis = retrieval.build_query_analysis("implementation barriers facilitators")
        record = {
            "title": "Implementation barriers facilitators in care",
            "abstract_text": "Qualitative study on adoption and feasibility",
            "text": "interview experience perception theme challenge adoption",
        }

        score = retrieval.compute_query_type_boost(analysis, record)

        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, retrieval.QUERY_TYPE_BOOST_CAP)

    def test_aggregate_papers_limits_support_bonus_for_many_medium_chunks(self):
        candidates = [
            {
                "chunk_id": "p1:0000",
                "paperId": "p1",
                "chunk_score": 0.84,
                "title": "One",
                "abstract_text": "",
                "url": "",
                "year": 2024,
                "text_source": "fulltext",
                "section": "results",
                "text": "A",
            },
            {
                "chunk_id": "p2:0000",
                "paperId": "p2",
                "chunk_score": 0.95,
                "title": "Two",
                "abstract_text": "",
                "url": "",
                "year": 2023,
                "text_source": "fulltext",
                "section": "body",
                "text": "C",
            },
        ]
        for idx in range(1, 9):
            candidates.append({
                "chunk_id": f"p1:{idx:04d}",
                "paperId": "p1",
                "chunk_score": 0.70,
                "title": "One",
                "abstract_text": "",
                "url": "",
                "year": 2024,
                "text_source": "fulltext",
                "section": "body",
                "text": f"Support {idx}",
            })

        papers = retrieval.aggregate_papers(candidates)

        self.assertEqual("p2", papers[0]["paperId"])
        self.assertEqual(9, papers[1]["supporting_chunks"])
        self.assertLessEqual(papers[1]["support_bonus"], retrieval.PAPER_SUPPORT_MAX_BONUS + retrieval.PAPER_SECTION_DIVERSITY_MAX)

    def test_lexical_index_score_record_uses_title_and_body_for_bm25(self):
        record = {
            "chunk_id": "p1:0000",
            "paperId": "p1",
            "title": "Rehabilitation implementation",
            "abstract_text": "facilitators uniqueabstractterm",
            "url": "",
            "year": 2024,
            "text_source": "abstract",
            "section": "abstract",
            "text": "barriers in rehabilitation",
        }
        lexical_index = retrieval.LexicalIndex([record])
        analysis = retrieval.build_query_analysis("uniqueabstractterm")

        scored = lexical_index.score_record(analysis, record)

        self.assertEqual(0.0, scored["bm25_score"])
        self.assertGreater(scored["abstract_field_score"], 0.0)

    def test_hybrid_query_result_excludes_weak_lexical_only_candidate(self):
        vector_record = {
            "chunk_id": "vec:0000",
            "paperId": "vec",
            "title": "Implementation barriers in rehabilitation",
            "abstract_text": "Staff perceptions",
            "url": "",
            "year": 2024,
            "text_source": "fulltext",
            "section": "body",
            "text": "implementation barriers facilitators rehabilitation workforce",
        }
        lexical_only_record = {
            "chunk_id": "lex:0000",
            "paperId": "lex",
            "title": "Lexical only paper",
            "abstract_text": "implementation barriers facilitators",
            "url": "",
            "year": 2024,
            "text_source": "abstract",
            "section": "abstract",
            "text": "implementation barriers facilitators",
        }
        lexical_index = FakeLexicalIndex(
            [vector_record, lexical_only_record],
            {
                "vec:0000": {
                    "bm25_score": 6.0,
                    "lexical_score": 6.4,
                    "title_field_score": 0.2,
                    "abstract_field_score": 0.1,
                    "body_field_score": 0.1,
                },
                "lex:0000": {
                    "bm25_score": 100.0,
                    "lexical_score": 101.0,
                    "title_field_score": 0.5,
                    "abstract_field_score": 0.5,
                    "body_field_score": 0.0,
                },
            },
        )
        collection = FakeCollection(
            records_by_id={"vec:0000": vector_record, "lex:0000": lexical_only_record},
            embeddings_by_id={"vec:0000": [1.0, 0.0], "lex:0000": [0.0, 1.0]},
        )
        raw_vector_result = {
            "ids": [["vec:0000"]],
            "documents": [[vector_record["text"]]],
            "metadatas": [[{
                "paperId": vector_record["paperId"],
                "title": vector_record["title"],
                "url": vector_record["url"],
                "year": vector_record["year"],
                "text_source": vector_record["text_source"],
                "section": vector_record["section"],
            }]],
            "distances": [[0.8]],
            "embeddings": [[[1.0, 0.0]]],
        }

        with patch("Retrieval.retrieval.apply_cross_encoder", return_value="cross-encoder skipped"):
            with patch("Retrieval.retrieval.apply_mmr_selection", side_effect=lambda papers, k: papers[:k]):
                result = retrieval.hybrid_query_result(
                    collection=collection,
                    lexical_index=lexical_index,
                    raw_vector_result=raw_vector_result,
                    query_embeddings=[[1.0, 0.0]],
                    query_texts=["implementation barriers facilitators rehabilitation"],
                    k=5,
                    min_score=0.55,
                )

        self.assertEqual(["vec:0000"], result["ids"][0])
        self.assertNotIn("lex:0000", result["ids"][0])

    def test_hybrid_query_result_can_rescue_strong_lexical_only_candidate(self):
        lexical_only_record = {
            "chunk_id": "lex:0000",
            "paperId": "lex",
            "title": "Evidence-based practice instrument for rehabilitation",
            "abstract_text": "Validation study of an instrument for rehabilitation practice",
            "url": "",
            "year": 2024,
            "text_source": "abstract",
            "section": "abstract",
            "text": "instrument validation rehabilitation evidence-based practice",
        }
        lexical_index = FakeLexicalIndex(
            [lexical_only_record],
            {
                "lex:0000": {
                    "bm25_score": 8.0,
                    "lexical_score": 8.6,
                    "title_field_score": 0.3,
                    "abstract_field_score": 0.18,
                    "body_field_score": 0.1,
                },
            },
        )
        collection = FakeCollection(
            records_by_id={"lex:0000": lexical_only_record},
            embeddings_by_id={"lex:0000": [0.5, 0.8660254]},
        )
        raw_vector_result = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "embeddings": [[]],
        }

        with patch("Retrieval.retrieval.apply_cross_encoder", return_value="cross-encoder skipped"):
            with patch("Retrieval.retrieval.apply_mmr_selection", side_effect=lambda papers, k: papers[:k]):
                result = retrieval.hybrid_query_result(
                    collection=collection,
                    lexical_index=lexical_index,
                    raw_vector_result=raw_vector_result,
                    query_embeddings=[[1.0, 0.0]],
                    query_texts=["evidence-based practice instrument rehabilitation"],
                    k=5,
                    min_score=0.55,
                )

        self.assertEqual(["lex:0000"], result["ids"][0])
        self.assertTrue(any("strict lexical rescue=1" in note for note in result["retrieval_notes"][0]))

    def test_hybrid_query_result_uses_embedding_score_not_raw_chroma_distance(self):
        vector_record = {
            "chunk_id": "vec:0000",
            "paperId": "vec",
            "title": "Implementation barriers in rehabilitation",
            "abstract_text": "Staff perceptions",
            "url": "",
            "year": 2024,
            "text_source": "fulltext",
            "section": "body",
            "text": "implementation barriers facilitators rehabilitation workforce",
        }
        lexical_index = FakeLexicalIndex(
            [vector_record],
            {
                "vec:0000": {
                    "bm25_score": 6.0,
                    "lexical_score": 6.4,
                    "title_field_score": 0.2,
                    "abstract_field_score": 0.1,
                    "body_field_score": 0.1,
                },
            },
        )
        raw_vector_result = {
            "ids": [["vec:0000"]],
            "documents": [[vector_record["text"]]],
            "metadatas": [[{
                "paperId": vector_record["paperId"],
                "title": vector_record["title"],
                "url": vector_record["url"],
                "year": vector_record["year"],
                "text_source": vector_record["text_source"],
                "section": vector_record["section"],
            }]],
            "distances": [[99.0]],
            "embeddings": [[[1.0, 0.0]]],
        }

        with patch("Retrieval.retrieval.apply_cross_encoder", return_value="cross-encoder skipped"):
            with patch("Retrieval.retrieval.apply_mmr_selection", side_effect=lambda papers, k: papers[:k]):
                result = retrieval.hybrid_query_result(
                    collection=FakeCollection(records_by_id={"vec:0000": vector_record}, embeddings_by_id={"vec:0000": [1.0, 0.0]}),
                    lexical_index=lexical_index,
                    raw_vector_result=raw_vector_result,
                    query_embeddings=[[1.0, 0.0]],
                    query_texts=["implementation barriers facilitators rehabilitation"],
                    k=5,
                    min_score=0.55,
                )

        self.assertGreater(result["embedding_scores"][0][0], 0.99)

    def test_build_result_row_embeds_score_breakdown_in_metadata(self):
        row = retrieval.build_result_row(
            [
                {
                    "chunk_id": "p1:0000",
                    "paperId": "p1",
                    "title": "Paper One",
                    "url": "https://example.org/p1",
                    "year": 2024,
                    "text_source": "fulltext",
                    "section": "results",
                    "supporting_chunks": 2,
                    "text": "Findings.",
                    "distance": 0.1,
                    "embedding_score": 0.91,
                    "bm25_score": 6.0,
                    "hybrid_score": 0.84,
                    "paper_score": 0.88,
                    "cross_encoder_score": 0.42,
                    "mmr_score": 0.93,
                }
            ],
            {"query_type": "general"},
            ["hybrid retrieval"],
        )

        metadata = row["metadatas"][0]

        self.assertEqual(0.93, row["final_scores"][0])
        self.assertEqual(0.93, metadata["final_score"])
        self.assertEqual(0.42, metadata["retrieval_scores"]["cross_encoder_score"])
        self.assertEqual({"cross_encoder_score", "final_score"}, set(metadata["retrieval_scores"].keys()))

    def test_calibrate_cross_encoder_scores_is_not_relative_minmax(self):
        calibrated = retrieval.calibrate_cross_encoder_scores([0.1, 0.2])

        self.assertLess(max(calibrated), 0.6)
        self.assertGreater(calibrated[1], calibrated[0])


if __name__ == "__main__":
    unittest.main()
