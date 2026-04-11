"""Embedding function factory for semantic retrieval.

Supports any OpenAI-compatible HTTP endpoint (OpenAI, LiteLLM proxy,
Ollama, etc.). No hard dependency on openai SDK — uses urllib.request
in a thread-pool executor so it doesn't block the event loop.

Also supports local embeddings via fastembed (BAAI/bge-small-en-v1.5)
for zero-cost, zero-latency embedding at ingestion time.
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
BatchEmbedFn = Callable[[List[str]], Awaitable[List[List[float]]]]


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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _http_embed,
            text,
            base_url,
            model,
            api_key,
        )

    return _embed


# ------------------------------------------------------------------
# Local embeddings via fastembed (no API key, no network)
# ------------------------------------------------------------------

_local_model = None          # lazy singleton
_local_model_name: str = ""  # tracks which model is loaded


def _get_local_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Lazy-load the fastembed model (singleton to avoid re-downloading).

    Re-loads if *model_name* differs from the currently loaded model so
    that config changes at runtime produce the correct embeddings.
    """
    global _local_model, _local_model_name
    if _local_model is None or _local_model_name != model_name:
        try:
            from fastembed import TextEmbedding
            logger.info("Loading local embedding model: %s", model_name)
            _local_model = TextEmbedding(model_name)
            _local_model_name = model_name
        except ImportError:
            raise ImportError(
                "fastembed is required for local embeddings. "
                "Install with: pip install fastembed"
            )
    return _local_model


def _local_embed_sync(text: str, model_name: str) -> List[float]:
    """Synchronous local embedding for a single text."""
    model = _get_local_model(model_name)
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def _local_embed_batch_sync(texts: List[str], model_name: str) -> List[List[float]]:
    """Synchronous local batch embedding."""
    if not texts:
        return []
    model = _get_local_model(model_name)
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


def make_local_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> EmbedFn:
    """Return an async embed function using local fastembed model.

    No API key needed. No network calls. Uses BAAI/bge-small-en-v1.5
    (384 dimensions) by default.
    """
    async def _embed(text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _local_embed_sync, text, model_name,
        )

    return _embed


def make_local_batch_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> BatchEmbedFn:
    """Return an async batch embed function using local fastembed model.

    Efficient for ingestion — embeds multiple texts in one call.
    """
    async def _embed_batch(texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _local_embed_batch_sync, texts, model_name,
        )

    return _embed_batch


def make_raw_vector_embedder(config: "LCMConfig") -> Optional[EmbedFn]:
    """Return an embed function for raw vector retrieval, or None if disabled.

    Uses local fastembed by default (raw_vector_use_local=True).
    Falls back to API embedder if configured otherwise.
    """
    if not config.raw_vector_enabled:
        return None

    if config.raw_vector_use_local:
        return make_local_embedder(config.raw_vector_model)

    # Fall back to API embedder if local is disabled
    if not config.embedding_base_url:
        logger.warning(
            "raw_vector_enabled=True, raw_vector_use_local=False, "
            "but embedding_base_url is not set — raw vector retrieval disabled"
        )
        return None

    return make_embedder(config)


def make_raw_vector_batch_embedder(config: "LCMConfig") -> Optional[BatchEmbedFn]:
    """Return a batch embed function for raw vector ingestion, or None if disabled.

    Local mode (raw_vector_use_local=True) uses fastembed's native batch
    embed for efficiency.  API mode falls back to a sequential loop over
    the single-embed function so it is never a silent no-op.
    """
    if not config.raw_vector_enabled:
        return None

    if config.raw_vector_use_local:
        return make_local_batch_embedder(config.raw_vector_model)

    # API mode: no dedicated batch endpoint, wrap single-embed in a loop
    single_embed = make_raw_vector_embedder(config)
    if single_embed is None:
        return None

    async def _api_batch(texts: List[str]) -> List[List[float]]:
        import asyncio
        return list(await asyncio.gather(*[single_embed(t) for t in texts]))

    return _api_batch
