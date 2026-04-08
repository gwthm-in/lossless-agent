#!/usr/bin/env python3
"""Framework integration example using GenericAdapter lifecycle.

Shows the full on_turn_start / on_turn_end pattern, tool handling,
system prompt injection, and session end.

Run: python examples/framework_integration.py
"""
import asyncio
import json
from lossless_agent import GenericAdapter, LCMConfig


async def mock_summarize(text: str) -> str:
    """Replace with your LLM summarization call."""
    return f"Summary: {text[:100]}..."


async def main() -> None:
    config = LCMConfig(db_path=":memory:", max_context_tokens=8000)
    adapter = GenericAdapter(config, mock_summarize)

    session = "user-123-session-1"

    # --- System prompt injection ---
    system_prompt = adapter.get_system_prompt_block()
    print("System prompt block (first 120 chars):")
    print(f"  {system_prompt[:120]}...")
    print()

    # --- Get tool definitions for your LLM ---
    tools = adapter.get_tools()
    print(f"Available tools: {[t['function']['name'] for t in tools]}")
    print()

    # --- Turn 1: Start ---
    context = await adapter.on_turn_start(session, "What did we discuss yesterday?")
    print(f"Turn 1 context: {context}")  # None on first turn (no history yet)

    # --- Turn 1: End (persist messages) ---
    await adapter.on_turn_end(session, [
        {"role": "user", "content": "What did we discuss yesterday?"},
        {"role": "assistant", "content": "I don't have any previous context yet."},
    ])

    # --- Ingest more history to demonstrate tool use ---
    for i in range(15):
        await adapter.on_turn_end(session, [
            {"role": "user", "content": f"Let's discuss deployment strategy {i}"},
            {"role": "assistant", "content": f"Here's my analysis of strategy {i}: we should use blue-green deployments with canary releases for safety."},
        ])

    # --- Turn 2: Context is now available ---
    context = await adapter.on_turn_start(session, "Remind me about deployment")
    if context:
        print(f"Turn 2 context ({len(context)} chars): {context[:100]}...")
    print()

    # --- Simulate tool call from LLM ---
    tool_result = await adapter.handle_tool_call("lcm_grep", {"query": "deployment"})
    parsed = json.loads(tool_result)
    print(f"lcm_grep('deployment') returned {len(parsed)} results")

    if parsed:
        first = parsed[0]
        print(f"  First result snippet: {first.get('snippet', '')[:80]}...")
    print()

    # --- Session end: final compaction ---
    await adapter.on_session_end(session)
    print("Session ended, final compaction complete.")

    # --- Stats (GenericAdapter convenience method) ---
    stats = await adapter.get_stats(session)
    print(f"Session stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
