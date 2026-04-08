"""Dynamic system prompt injection based on compaction state."""
from __future__ import annotations

from typing import Dict, List, Optional

from lossless_agent.store.models import Summary

_COMPACTION_PROMPT = """\
IMPORTANT: This conversation has been heavily compacted. Some details may be in compressed summaries.
Before answering questions about past context:
1. Use lcm_grep to search for relevant terms
2. Use lcm_describe to inspect summary metadata
3. Use lcm_expand_query for detailed retrieval
4. Cite summary IDs when referencing compacted context
If uncertain about details from compacted history, say so explicitly."""


class CompactionAwarePrompt:
    """Generates system prompt additions based on compaction depth."""

    def generate(
        self,
        summaries: List[Summary],
        depth_threshold: int = 2,
        condensed_threshold: int = 2,
    ) -> Optional[str]:
        if not summaries:
            return None

        stats = self.get_compaction_stats(summaries)

        if stats["max_depth"] >= depth_threshold:
            return _COMPACTION_PROMPT
        if stats["condensed_count"] >= condensed_threshold:
            return _COMPACTION_PROMPT

        return None

    def get_compaction_stats(self, summaries: List[Summary]) -> Dict[str, int]:
        if not summaries:
            return {
                "max_depth": 0,
                "leaf_count": 0,
                "condensed_count": 0,
                "total_summaries": 0,
            }

        return {
            "max_depth": max(s.depth for s in summaries),
            "leaf_count": sum(1 for s in summaries if s.kind == "leaf"),
            "condensed_count": sum(1 for s in summaries if s.kind == "condensed"),
            "total_summaries": len(summaries),
        }
