#!/usr/bin/env python3
"""Basic LCM integration in ~30 lines.

Run: python examples/basic_integration.py
"""
import asyncio
from lossless_agent import SimpleAdapter, LCMConfig


async def mock_summarize(text: str) -> str:
    """Cheap mock summarizer — replace with your LLM call."""
    return f"Summary: {text[:80]}..."


async def main() -> None:
    config = LCMConfig(db_path=":memory:", max_context_tokens=4000)
    adapter = SimpleAdapter(":memory:", mock_summarize, config)

    # 1. Ingest messages
    messages = [
        {"role": "user", "content": f"Message {i}: discuss topic {i}"} for i in range(20)
    ]
    await adapter.ingest("session-1", messages)
    print(f"Ingested {len(messages)} messages")

    # 2. Compact
    created = await adapter.compact("session-1")
    print(f"Compaction created {created} summaries")

    # 3. Retrieve context within budget
    context = await adapter.retrieve("session-1", budget_tokens=2000)
    print(f"Context length: {len(context or '')} chars")

    # 4. Search
    results = await adapter.search("topic 5")
    print(f"Search returned {len(results)} results")

    # 5. Expand a summary (if any exist)
    if results:
        for r in results:
            if r.get("summary_id"):
                detail = await adapter.expand(r["summary_id"])
                print(f"Expanded summary: {list(detail.keys())}")
                break

    adapter.close()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
