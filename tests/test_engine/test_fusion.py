"""Tests for Reciprocal Rank Fusion."""
from lossless_agent.engine.fusion import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = reciprocal_rank_fusion(
            [("a", 0.9), ("b", 0.8), ("c", 0.7)],
            k=60,
        )
        ids = [doc_id for doc_id, _ in results]
        assert ids == ["a", "b", "c"]

    def test_two_lists_merge(self):
        fts = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        vec = [("b", 0.95), ("d", 0.85), ("a", 0.75)]
        results = reciprocal_rank_fusion(fts, vec, k=60)
        ids = [doc_id for doc_id, _ in results]
        # b appears rank 0 in vec + rank 1 in fts → highest RRF
        # a appears rank 0 in fts + rank 2 in vec → second
        assert ids[0] == "b"
        assert ids[1] == "a"
        # c only in fts, d only in vec
        assert set(ids) == {"a", "b", "c", "d"}

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [], k=60)
        assert results == []

    def test_no_overlap(self):
        fts = [("a", 0.9)]
        vec = [("b", 0.9)]
        results = reciprocal_rank_fusion(fts, vec, k=60)
        ids = [doc_id for doc_id, _ in results]
        assert set(ids) == {"a", "b"}
        # Same rank in their respective lists, so same RRF score
        assert results[0][1] == results[1][1]

    def test_k_parameter_affects_scores(self):
        data = [("a", 1.0), ("b", 0.5)]
        r1 = reciprocal_rank_fusion(data, k=1)
        r2 = reciprocal_rank_fusion(data, k=100)
        # With k=1: scores are 1/2 and 1/3
        # With k=100: scores are 1/101 and 1/102
        # Score gap should be larger with smaller k
        gap_small_k = r1[0][1] - r1[1][1]
        gap_large_k = r2[0][1] - r2[1][1]
        assert gap_small_k > gap_large_k
