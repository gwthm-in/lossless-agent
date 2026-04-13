"""Reciprocal Rank Fusion (RRF) for merging multiple ranked result lists.

Used to combine results from FTS5 keyword search and vector similarity
search into a single unified ranking.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    *result_lists: List[Tuple[str, float]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result list is a sequence of (doc_id, score) tuples, ordered by
    relevance (best first). The RRF score for each document is:

        sum(1 / (k + rank + 1)) across all lists where it appears

    The parameter k controls how much weight is given to lower-ranked
    results. k=60 is the standard value from the original RRF paper
    (Cormack et al., 2009).

    Args:
        *result_lists: One or more ranked lists of (doc_id, score) tuples.
        k: Smoothing constant (default: 60).

    Returns:
        Merged list of (doc_id, rrf_score) tuples, sorted by RRF score desc.
    """
    scores: Dict[str, float] = defaultdict(float)

    for results in result_lists:
        for rank, (doc_id, _original_score) in enumerate(results):
            scores[doc_id] += 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: -x[1])
