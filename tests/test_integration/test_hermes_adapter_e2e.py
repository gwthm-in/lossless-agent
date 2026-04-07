"""End-to-end integration tests for the HermesAdapter.

Tests cover the full turn lifecycle, tool integration, session end
compaction, system prompt content, and tool schema validation.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from lossless_agent.config import LCMConfig
from lossless_agent.adapters.hermes import HermesAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summarize_fn() -> AsyncMock:
    """Return an AsyncMock summarizer that generates plausible summaries."""

    async def _summarize(text: str) -> str:
        words = text.split()
        snippet = " ".join(words[:15]) if len(words) > 15 else " ".join(words)
        return (
            f"Summary: Discussion about Python debugging and deployment. "
            f"Key topics: error handling, database optimization, caching. "
            f"Snippet: {snippet}"
        )

    mock = AsyncMock(side_effect=_summarize)
    return mock


def _make_config(tmp_path, db_name: str = "hermes_test.db") -> LCMConfig:
    """Build an LCMConfig with aggressive settings for fast testing."""
    return LCMConfig(
        db_path=str(tmp_path / db_name),
        fresh_tail_count=4,
        leaf_chunk_tokens=500,
        leaf_min_fanout=4,
        condensed_min_fanout=3,
        context_threshold=0.3,
        max_context_tokens=5000,
        summary_budget_ratio=0.5,
    )


def _make_turn_messages(turn_num: int):
    """Generate realistic user + assistant message dicts for a turn."""
    user_msgs = [
        "I'm getting a segfault in my C extension module when calling PyObject_GetAttr. The backtrace shows it crashes in the tp_getattro slot. What could cause this?",
        "I added Py_INCREF but now I'm getting a memory leak. Valgrind shows 2MB of unreachable blocks after processing 1000 requests.",
        "Switched to using PyMem_Malloc instead of raw malloc. The leak is gone but performance dropped by 15%. Any optimization ideas?",
        "I profiled with perf and the hotspot is in the string comparison loop. Should I use Python's intern mechanism for the attribute names?",
        "Interning helped - 40% speedup on the hot path. Now I need to handle the GIL properly for the background worker thread.",
        "Released the GIL during the I/O operation using Py_BEGIN_ALLOW_THREADS. But now I'm getting occasional data corruption.",
        "Found it - the shared buffer wasn't protected by a mutex. Added pthread_mutex_lock around the critical section. Corruption is fixed.",
        "Final benchmarks: 3x faster than the pure Python version, no memory leaks, thread-safe. Ready to publish to PyPI.",
        "Setting up the CI pipeline with cibuildwheel for binary wheels on Linux, macOS, and Windows.",
        "All wheels built successfully. Published to PyPI with twine. The package has 500 downloads in the first week.",
    ]

    assistant_msgs = [
        "A segfault in tp_getattro usually means you're accessing a freed object. Check that the object's reference count is correct. Use Py_XINCREF/Py_XDECREF for nullable pointers.",
        "Memory leaks with Py_INCREF typically mean you're incrementing without a matching decrement. Use a ref-count debugger or compile Python with --with-pydebug to catch imbalances.",
        "PyMem_Malloc routes through Python's memory allocator which has overhead for small blocks. Use PyMem_Malloc only for Python-managed memory. For hot paths, consider a memory pool or arena allocator.",
        "Yes, string interning via PyUnicode_InternInPlace is exactly what you want for repeated attribute lookups. CPython already interns identifiers, but explicit interning ensures it for your custom strings.",
        "For GIL management with background threads: release the GIL only when you don't touch Python objects. Pattern: Py_BEGIN_ALLOW_THREADS, do C work, Py_END_ALLOW_THREADS. Never access PyObject* between those macros.",
        "Data corruption with the GIL released means your C code has a race condition on shared state. Even though you released the GIL, your C data structures still need protection via mutex or atomic operations.",
        "Good catch. For the mutex, make sure to use the same pattern: lock, modify buffer, unlock. Consider using a read-write lock if reads vastly outnumber writes for better concurrency.",
        "Excellent results! For PyPI publishing, make sure to include: proper README, LICENSE, type stubs (.pyi files), and a MANIFEST.in. Use twine upload with --sign for GPG signatures.",
        "cibuildwheel is the gold standard. Set CIBW_TEST_COMMAND to run your test suite during the build. This catches platform-specific issues before release. Don't forget to set the minimum macOS deployment target.",
        "500 downloads in a week is a great start! Consider adding: badges to README, changelog, contribution guidelines, and GitHub Actions for automated releases on tag push.",
    ]

    idx = turn_num % len(user_msgs)
    return [
        {"role": "user", "content": user_msgs[idx], "token_count": 50},
        {"role": "assistant", "content": assistant_msgs[idx], "token_count": 60},
    ]


# ===================================================================
# Test 1: Adapter turn lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_adapter_turn_lifecycle(tmp_path):
    """Walk through the full turn lifecycle: start -> end -> start again.
    Eventually compaction triggers and context is returned."""
    config = _make_config(tmp_path)
    summarize_fn = _make_summarize_fn()
    adapter = HermesAdapter(config, summarize_fn)
    session_key = "lifecycle-session"

    # First turn start on empty conversation => None
    context = await adapter.on_turn_start(session_key, "Hello")
    assert context is None

    # Add messages across multiple turns
    for turn_num in range(15):
        messages = _make_turn_messages(turn_num)
        await adapter.on_turn_end(session_key, messages)

        # Check if context is returned on next turn start
        ctx = await adapter.on_turn_start(session_key, f"Turn {turn_num + 1} question")
        if ctx is not None:
            pass

    # After enough turns with aggressive settings, we should get context back
    # (tail messages exist so context assembler returns something)
    final_ctx = await adapter.on_turn_start(session_key, "Final question")
    assert final_ctx is not None, (
        "Expected context after multiple turns with messages"
    )


# ===================================================================
# Test 2: Adapter tool integration
# ===================================================================


@pytest.mark.asyncio
async def test_adapter_tool_integration(tmp_path):
    """Run turns, force compaction, then use tools via handle_tool_call."""
    config = _make_config(tmp_path)
    summarize_fn = _make_summarize_fn()
    adapter = HermesAdapter(config, summarize_fn)
    session_key = "tool-integration-session"

    # Add several turns worth of messages
    for turn_num in range(10):
        messages = _make_turn_messages(turn_num)
        await adapter.on_turn_end(session_key, messages)

    # Force compaction via on_session_end (runs exhaustive compaction)
    await adapter.on_session_end(session_key)

    # lcm_grep: search for content
    grep_result_json = await adapter.handle_tool_call(
        "lcm_grep", {"query": "segfault"}
    )
    grep_results = json.loads(grep_result_json)
    assert isinstance(grep_results, list)
    assert len(grep_results) > 0
    # Each result should have expected fields
    for r in grep_results:
        assert "type" in r
        assert "id" in r
        assert "content_snippet" in r

    # lcm_describe: get metadata for a summary
    # First find a summary via grep
    summary_grep = await adapter.handle_tool_call(
        "lcm_grep", {"query": "debugging", "scope": "summaries"}
    )
    summary_results = json.loads(summary_grep)
    if len(summary_results) > 0:
        summary_id = summary_results[0]["id"]

        describe_json = await adapter.handle_tool_call(
            "lcm_describe", {"summary_id": summary_id}
        )
        describe_result = json.loads(describe_json)
        assert "summary_id" in describe_result
        assert "kind" in describe_result
        assert "depth" in describe_result

        # lcm_expand: drill into the summary
        expand_json = await adapter.handle_tool_call(
            "lcm_expand", {"summary_id": summary_id}
        )
        expand_result = json.loads(expand_json)
        assert "summary_id" in expand_result
        assert "kind" in expand_result
        assert "children" in expand_result
        assert len(expand_result["children"]) > 0

    # Unknown tool returns error
    error_json = await adapter.handle_tool_call("lcm_unknown", {})
    error_result = json.loads(error_json)
    assert "error" in error_result


# ===================================================================
# Test 3: Adapter session end
# ===================================================================


@pytest.mark.asyncio
async def test_adapter_session_end(tmp_path):
    """Verify on_session_end runs final compaction."""
    config = _make_config(tmp_path)
    summarize_fn = _make_summarize_fn()
    adapter = HermesAdapter(config, summarize_fn)
    session_key = "session-end-test"

    # Add enough messages to have material for compaction
    for turn_num in range(8):
        messages = _make_turn_messages(turn_num)
        await adapter.on_turn_end(session_key, messages)

    # Run session end
    await adapter.on_session_end(session_key)

    # Verify summarize_fn was called (compaction happened)
    assert summarize_fn.call_count > 0

    # Verify summaries exist in the database
    conv = adapter._conv_store.get_or_create(session_key)
    summaries = adapter._sum_store.get_by_conversation(conv.id)
    assert len(summaries) > 0, "Expected summaries after session end compaction"


# ===================================================================
# Test 4: Adapter system prompt
# ===================================================================


def test_adapter_system_prompt(tmp_path):
    """get_system_prompt_block returns text mentioning all recall tools."""
    config = _make_config(tmp_path)
    summarize_fn = _make_summarize_fn()
    adapter = HermesAdapter(config, summarize_fn)

    prompt = adapter.get_system_prompt_block()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "lcm_grep" in prompt
    assert "lcm_describe" in prompt
    assert "lcm_expand" in prompt


# ===================================================================
# Test 5: Adapter tool schemas
# ===================================================================


def test_adapter_tool_schemas(tmp_path):
    """get_tools returns valid OpenAI function-calling schemas."""
    config = _make_config(tmp_path)
    summarize_fn = _make_summarize_fn()
    adapter = HermesAdapter(config, summarize_fn)

    tools = adapter.get_tools()
    assert isinstance(tools, list)
    assert len(tools) == 3  # lcm_grep, lcm_describe, lcm_expand

    tool_names = set()
    for tool in tools:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool

        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert isinstance(func["description"], str)
        assert len(func["description"]) > 0

        params = func["parameters"]
        assert "type" in params
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert isinstance(params["required"], list)

        tool_names.add(func["name"])

    assert tool_names == {"lcm_grep", "lcm_describe", "lcm_expand"}
