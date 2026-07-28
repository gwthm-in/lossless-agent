"""Summarizer factory — shared by the MCP server and the Claude Code capture hook.

A *summarizer* is an ``async (prompt: str) -> str`` callable the compaction engine invokes
to compress a chunk of transcript into a summary node. Provider selection is config-driven so
callers never hard-code a backend:

    build_summarize_fn(config, command)   # command > anthropic > openai > truncation

This module deliberately has no dependency on the MCP server (or the ``mcp`` package), so
lightweight integrations (hooks, scripts) can import it without pulling in a server stack.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from lossless_agent.engine.compaction import SummarizeFn


def make_openai_summarizer(model: str, base_url: str = "") -> SummarizeFn:
    """Summarizer that calls any OpenAI-compatible endpoint (OpenAI, LiteLLM, Azure, Groq…).

    Requires OPENAI_API_KEY and the ``openai`` package (``pip install openai``).
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "LCM_SUMMARY_PROVIDER=openai requires the openai package. "
            "Install with: pip install openai"
        )
    _model = model or "gpt-4o-mini"
    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)

    async def summarize(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return summarize


def make_anthropic_summarizer(model: str) -> SummarizeFn:
    """Summarizer that calls the Anthropic Messages API directly.

    Requires ANTHROPIC_API_KEY in the environment and the ``anthropic`` package
    (``pip install anthropic``). This is the low-latency path — a single HTTPS call, no CLI
    cold-start — and is the recommended backend for the Claude Code capture hook.
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        raise ImportError(
            "LCM_SUMMARY_PROVIDER=anthropic requires the anthropic package. "
            "Install with: pip install anthropic"
        )
    _model = model or "claude-haiku-4-5-20251001"
    client = _anthropic.AsyncAnthropic()

    async def summarize(prompt: str) -> str:
        message = await client.messages.create(
            model=_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text  # type: ignore[union-attr]

    return summarize


def make_command_summarizer(command: str) -> SummarizeFn:
    """Summarizer that pipes the prompt to an external command via stdin.

    The command reads the prompt from stdin and writes the summary to stdout.
    Example: ``LCM_SUMMARIZE_COMMAND='python my_summarizer.py'``.
    """
    async def summarize(prompt: str) -> str:
        # LCM_SUMMARIZING guards against self-capture: if the command spawns its own agent
        # session (e.g. `claude -p`), the capture hook it inherits sees this and bails.
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "LCM_SUMMARIZING": "1"},
        )
        stdout, stderr = await proc.communicate(prompt.encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"Summarize command failed (exit {proc.returncode}): {stderr.decode()}"
            )
        return stdout.decode("utf-8").strip()

    return summarize


def make_truncation_summarizer() -> SummarizeFn:
    """Built-in deterministic summarizer: truncates to target length.

    A fallback — it preserves the DAG structure so original messages are still recoverable via
    ``lcm_expand``, even though the summary text is truncated rather than intelligently
    compressed. Configure a real provider (anthropic/openai/command) for quality summaries.
    """
    async def summarize(prompt: str) -> str:
        target_tokens = 1200  # default leaf target
        char_limit = target_tokens * 4
        if len(prompt) <= char_limit:
            return prompt
        return prompt[:char_limit] + "\n[Summary truncated — configure --summarize-command for LLM-quality summaries]"

    return summarize


def build_summarize_fn(config, command: Optional[str] = None) -> SummarizeFn:
    """Return the configured summarizer (first match wins):
    1. explicit ``command`` (LCM_SUMMARIZE_COMMAND / --summarize-command)
    2. LCM_SUMMARY_PROVIDER=anthropic
    3. LCM_SUMMARY_PROVIDER=openai  (also LiteLLM, Azure, Groq, …)
    4. deterministic truncation fallback
    """
    if command:
        return make_command_summarizer(command)
    if config and config.summary_provider == "anthropic":
        return make_anthropic_summarizer(config.summary_model)
    if config and config.summary_provider == "openai":
        return make_openai_summarizer(config.summary_model, config.summary_base_url)
    return make_truncation_summarizer()


def build_expansion_fn(config, command: Optional[str] = None) -> SummarizeFn:
    """Like :func:`build_summarize_fn` but for ``lcm_expand_query`` synthesis — honours
    ``LCM_EXPANSION_MODEL`` (config.expansion_model) when set."""
    if command:
        return make_command_summarizer(command)
    if config and config.summary_provider == "anthropic":
        return make_anthropic_summarizer(config.expansion_model or config.summary_model)
    if config and config.summary_provider == "openai":
        return make_openai_summarizer(
            config.expansion_model or config.summary_model, config.summary_base_url
        )
    return make_truncation_summarizer()
