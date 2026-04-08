"""Structured prompt templates for the summarisation engine."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a context-compaction summarization engine. "
    "Follow user instructions exactly and return plain text summary content only."
)


def build_leaf_prompt(
    messages_text: str,
    target_tokens: int,
    custom_instructions: str = "",
    previous_summary: str = "",
    aggressive: bool = False,
) -> str:
    """Build the user prompt for leaf (message) summarisation."""
    parts: list[str] = []

    parts.append(f"Target token count: {target_tokens}")

    instr = custom_instructions.strip() if custom_instructions else ""
    parts.append(f"Operator instructions: {instr or '(none)'}")

    prev = previous_summary.strip() if previous_summary else ""
    if prev:
        parts.append(f"<previous_context>\n{prev}\n</previous_context>")
    else:
        parts.append("Previous context: (none)")

    parts.append(f"<messages>\n{messages_text}\n</messages>")

    if aggressive:
        parts.append("AGGRESSIVE: Compress much harder.")

    return "\n\n".join(parts)


def build_condensed_prompt(
    summaries_text: str,
    target_tokens: int,
    depth: int,
    custom_instructions: str = "",
    aggressive: bool = False,
) -> str:
    """Build the user prompt for condensed (summary-of-summaries) summarisation."""
    parts: list[str] = []

    parts.append(f"Target token count: {target_tokens}")

    if depth == 1:
        parts.append("Guidance: focus on what is new, changed, or resolved")
    elif depth == 2:
        parts.append("Guidance: preserve key decisions and outcomes")
    else:
        parts.append("Guidance: retain only the most critical facts")

    instr = custom_instructions.strip() if custom_instructions else ""
    parts.append(f"Operator instructions: {instr or '(none)'}")

    parts.append(f"<summaries>\n{summaries_text}\n</summaries>")

    if aggressive:
        parts.append("AGGRESSIVE: Compress much harder.")

    return "\n\n".join(parts)
