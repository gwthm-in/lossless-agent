"""Structured prompt templates for the summarisation engine.

The imperative to *summarise* lives in the user prompt (not a system prompt): summarizer backends
include a Claude Code OAuth token path that can reject a custom system prompt, and command/OpenAI
backends don't take one either. Keeping the instruction in the user message works for all of them.
"""
from __future__ import annotations

# Kept for backwards-compatibility / callers that can safely set a system prompt. The user prompt
# is self-contained, so this is optional.
SYSTEM_PROMPT = (
    "You are a context-compaction summarization engine. "
    "Follow user instructions exactly and return plain text summary content only."
)

_LEAF_INSTRUCTION = (
    "You are compacting a long AI-agent working session into durable long-term memory. "
    "Write a dense, factual SUMMARY of the conversation chunk in <messages>, at most {target} tokens.\n"
    "Rules:\n"
    "- This is a SUMMARY, not a reply. Do NOT continue, answer, or role-play the conversation, and "
    "never address a reader (no \"what would you like next\", no greetings, no first person).\n"
    "- Third person, past tense (\"The user asked… The assistant fixed…\").\n"
    "- Preserve the concrete, recall-critical details: decisions and WHY, file/function paths, "
    "commands, identifiers (PRs, commits, DB names), error messages, numbers, and any open threads "
    "or TODOs. Aggregate them across the whole chunk — do not just echo one message.\n"
    "- Drop pleasantries, acknowledgements, and transient status chatter."
)

_CONDENSED_INSTRUCTION = (
    "You are merging several lower-level summaries of an AI-agent session into ONE higher-level "
    "summary for durable long-term memory. Write a single dense, factual SUMMARY of at most {target} "
    "tokens.\n"
    "Rules:\n"
    "- This is a SUMMARY, not a reply. Do NOT continue or role-play the conversation, and never "
    "address a reader. Third person, past tense.\n"
    "- Consolidate decisions, outcomes, file paths, identifiers, and open threads across the inputs; "
    "remove redundancy and transient chatter; keep specifics over generalities."
)


def build_leaf_prompt(
    messages_text: str,
    target_tokens: int,
    custom_instructions: str = "",
    previous_summary: str = "",
    aggressive: bool = False,
) -> str:
    """Build the user prompt for leaf (message) summarisation."""
    parts: list[str] = [_LEAF_INSTRUCTION.format(target=target_tokens)]

    instr = custom_instructions.strip() if custom_instructions else ""
    if instr:
        parts.append(f"Additional operator instructions: {instr}")

    prev = previous_summary.strip() if previous_summary else ""
    if prev:
        parts.append(
            "Summary of earlier context (do not repeat it; continuity only):\n"
            f"<previous_context>\n{prev}\n</previous_context>"
        )

    parts.append(f"<messages>\n{messages_text}\n</messages>")

    if aggressive:
        parts.append("Compress harder than the target — keep only the most important facts.")

    parts.append("Output only the summary text, nothing else.")
    return "\n\n".join(parts)


def build_condensed_prompt(
    summaries_text: str,
    target_tokens: int,
    depth: int,
    custom_instructions: str = "",
    aggressive: bool = False,
) -> str:
    """Build the user prompt for condensed (summary-of-summaries) summarisation."""
    parts: list[str] = [_CONDENSED_INSTRUCTION.format(target=target_tokens)]

    if depth <= 1:
        parts.append("Emphasis: what was done, decided, changed, or resolved.")
    elif depth == 2:
        parts.append("Emphasis: key decisions and outcomes; drop step-by-step detail.")
    else:
        parts.append("Emphasis: retain only the most critical durable facts.")

    instr = custom_instructions.strip() if custom_instructions else ""
    if instr:
        parts.append(f"Additional operator instructions: {instr}")

    parts.append(f"<summaries>\n{summaries_text}\n</summaries>")

    if aggressive:
        parts.append("Compress harder — keep only the most important facts.")

    parts.append("Output only the summary text, nothing else.")
    return "\n\n".join(parts)
