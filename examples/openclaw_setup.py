#!/usr/bin/env python3
"""OpenClaw + Lossless Agent setup.

Uses the OpenClawAdapter for native integration with OpenClaw agents.
The adapter provides tool schemas with openclaw_metadata and the
Lossless Recall Policy system prompt block.

Install:
    pip install lossless-agent

Run:
    python examples/openclaw_setup.py
"""
import asyncio
import json
from lossless_agent import OpenClawAdapter, LCMConfig


async def mock_summarize(text: str) -> str:
    """Replace with your LLM summarization call."""
    return f"Summary: {text[:100]}..."


async def main() -> None:
    # 1. Configure
    config = LCMConfig(db_path=":memory:", max_context_tokens=8000)
    adapter = OpenClawAdapter(config, mock_summarize)

    session = "openclaw-session-1"

    # 2. Get the system prompt block (inject into your agent's system prompt)
    system_block = adapter.get_system_prompt_block()
    print("=== System Prompt Block ===")
    print(system_block)
    print()

    # 3. Get tool schemas (register with your OpenClaw agent)
    tools = adapter.get_tools()
    print(f"Tools: {[t['function']['name'] for t in tools]}")
    print(f"Each tool includes openclaw_metadata: {tools[0].get('openclaw_metadata')}")
    print()

    # 4. Lifecycle: on_turn_start -> agent turn -> on_turn_end
    #    Start turn — returns assembled context if history exists
    context = await adapter.on_turn_start(session, "Hello!")
    print(f"First turn context: {context}")  # None on first turn

    # End turn — persist the messages
    await adapter.on_turn_end(session, [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
    ])

    # Ingest a bunch of messages to demonstrate compaction
    for i in range(20):
        await adapter.on_turn_end(session, [
            {"role": "user", "content": f"Tell me about deployment strategy {i}"},
            {
                "role": "assistant",
                "content": (
                    f"Strategy {i}: Use blue-green deployments with canary "
                    f"releases. Monitor error rates for 15 minutes before "
                    f"promoting. Rollback automatically on >1% error rate."
                ),
            },
        ])

    # 5. Start another turn — now context is assembled from summaries + raw
    context = await adapter.on_turn_start(session, "Summarize our deployment discussion")
    if context:
        print(f"Assembled context ({len(context)} chars): {context[:120]}...")
    print()

    # 6. Handle tool calls from the agent
    result = await adapter.handle_tool_call("lcm_grep", {"query": "deployment"})
    parsed = json.loads(result)
    print(f"lcm_grep('deployment') -> {len(parsed)} results")
    if parsed:
        print(f"  First: {parsed[0].get('snippet', '')[:80]}...")
    print()

    # 7. Session end — final compaction
    await adapter.on_session_end(session)
    print("Session ended. Final compaction complete.")

    # 8. Stats
    stats = await adapter.get_stats(session)
    print(f"Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
