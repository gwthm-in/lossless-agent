"""Embedding function factory for semantic retrieval.

Supports any OpenAI-compatible HTTP endpoint (OpenAI, LiteLLM proxy,
Ollama, etc.). No hard dependency on openai SDK — uses urllib.request
in a thread-pool executor so it doesn't block the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import urllib.error
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional

if TYPE_CHECKING:
    from lossless_agent.config import LCMConfig

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[List[float]]]


def _http_embed(
    text: str,
    base_url: str,
    model: str,
    api_key: str,
) -> List[float]:
    """Synchronous HTTP call to an OpenAI-compatible /embeddings endpoint."""
    url = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"input": text, "model": model}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"][0]["embedding"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Embedding HTTP {exc.code}: {body[:200]}"
        ) from exc


def make_embedder(config: "LCMConfig") -> Optional[EmbedFn]:
    """Return an async embed function configured from *config*, or None.

    Returns None when cross-session retrieval is not configured
    (i.e. ``cross_session_enabled`` is False or ``embedding_base_url``
    is empty), so callers can guard with ``if embed_fn is not None``.
    """
    if not config.cross_session_enabled:
        return None
    if not config.embedding_base_url:
        logger.warning(
            "cross_session_enabled=True but embedding_base_url is not set — "
            "cross-session retrieval disabled"
        )
        return None

    base_url = config.embedding_base_url
    model = config.embedding_model
    api_key = config.embedding_api_key

    async def _embed(text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _http_embed,
            text,
            base_url,
            model,
            api_key,
        )

    return _embed
