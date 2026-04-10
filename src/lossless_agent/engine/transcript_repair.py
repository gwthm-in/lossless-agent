"""Transcript repair: fix tool-call / tool-result ordering issues."""
from __future__ import annotations

from copy import deepcopy
from typing import List

from lossless_agent.store.models import Message


def generate_fallback_tool_call_id(message_id: int, seq: int) -> str:
    """Generate a fallback tool_call_id for messages missing one."""
    return f"toolu_lcm_{message_id}_{seq}"


class TranscriptRepairer:
    """Repair common tool-call/tool-result ordering and pairing problems.

    The ``repair`` method applies four fixes in order:

    1. **Reorder** – move ``tool`` (toolResult) messages to be right after
       their matching ``assistant`` tool-call message (matched by
       ``tool_call_id``).
    2. **Synthesise** – insert a synthetic error ``tool`` message for any
       tool call that has no matching result.
    3. **De-duplicate** – drop duplicate ``tool`` messages with the same
       ``tool_call_id`` (keep the first).
    4. **Drop orphans** – remove ``tool`` messages whose ``tool_call_id``
       does not match any tool call in the transcript.
    """

    # Sentinel content used for synthesised missing results
    MISSING_RESULT_CONTENT = (
        "[Tool result missing - may have been lost during context management]"
    )

    def repair(self, messages: List[Message]) -> List[Message]:
        """Return a repaired copy of *messages*."""
        msgs = [deepcopy(m) for m in messages]

        # Step 0: Assign fallback tool_call_ids to assistant messages that have
        # a tool_name but no tool_call_id (prevents API crashes).
        for m in msgs:
            if m.role == "assistant" and m.tool_name and not m.tool_call_id:
                m.tool_call_id = generate_fallback_tool_call_id(m.id, m.seq)

        # Identify tool-call messages (assistant messages with a tool_call_id)
        # and tool-result messages (role='tool' with a tool_call_id).
        tool_call_ids = self._extract_tool_call_ids(msgs)
        self._extract_tool_result_ids(msgs)

        # Step 1 + 3: Separate out tool results, de-dup, then re-insert
        # after their matching tool call.
        result_map: dict[str, Message] = {}
        non_result_msgs: List[Message] = []
        for m in msgs:
            if m.role == "tool" and m.tool_call_id:
                if m.tool_call_id not in result_map:
                    result_map[m.tool_call_id] = m
                # else: duplicate – drop it
            else:
                non_result_msgs.append(m)

        # Step 4: Drop orphan results (no matching call)
        result_map = {
            tcid: m for tcid, m in result_map.items() if tcid in tool_call_ids
        }

        # Re-insert results right after their call
        ordered: List[Message] = []
        used_results: set = set()
        for m in non_result_msgs:
            ordered.append(m)
            if m.tool_call_id and m.tool_call_id in tool_call_ids and m.role == "assistant":
                tcid = m.tool_call_id
                if tcid in result_map:
                    ordered.append(result_map[tcid])
                    used_results.add(tcid)

        # Step 2: Insert synthetic results for calls with no result
        final: List[Message] = []
        for m in ordered:
            final.append(m)
            if (
                m.role == "assistant"
                and m.tool_call_id
                and m.tool_call_id in tool_call_ids
                and m.tool_call_id not in used_results
                and m.tool_call_id not in result_map
            ):
                synthetic = Message(
                    id=-1,
                    conversation_id=m.conversation_id,
                    seq=-1,
                    role="tool",
                    content=self.MISSING_RESULT_CONTENT,
                    token_count=0,
                    tool_call_id=m.tool_call_id,
                    tool_name=m.tool_name,
                    created_at=m.created_at,
                )
                final.append(synthetic)

        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_call_ids(msgs: List[Message]) -> set:
        """Return tool_call_ids from assistant messages (tool calls)."""
        return {
            m.tool_call_id
            for m in msgs
            if m.role == "assistant" and m.tool_call_id
        }

    @staticmethod
    def _extract_tool_result_ids(msgs: List[Message]) -> set:
        """Return tool_call_ids from tool-result messages."""
        return {
            m.tool_call_id
            for m in msgs
            if m.role == "tool" and m.tool_call_id
        }
