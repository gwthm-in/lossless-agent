# PR #3: Raw Vector Embeddings at Ingestion Time

## Problem

PR #2 added pgvector-based semantic search, but it embeds **compacted summaries** — meaning compaction must complete before vectors are available. On LongMemEval, this creates ~50 LLM calls per question just for compaction, pushing benchmark runtime to ~20 hours.

Meanwhile, the core retrieval problem is simple: FTS5 keyword search misses semantic matches (e.g., query says "yoga classes" but evidence says "Serenity Yoga"). We need semantic retrieval, but we don't need compaction to get it.

## Solution

Embed raw message chunks at ingestion time using local embeddings (no LLM calls). Keep the DAG compaction for its original purpose (context window management), but decouple retrieval from compaction entirely.

## Architecture

```
CURRENT (PR #2):
  Messages → Compaction (50 LLM calls) → Summaries → Embed → VectorStore → Search
                                                                              ↑
                                                                        BLOCKED until
                                                                        compaction done

PROPOSED (PR #3):
  Messages → Embed locally → VectorStore → Search    (instant, 0 LLM calls)
       ↓
  Compaction (async, background) → DAG summaries → Context assembly
```

## Components

### 1. Message Embedder (modify `engine/embedder.py`)

**Current:** `make_embedder()` returns an async function that calls an external OpenAI-compatible API.

**Change:** Add a `make_local_embedder()` that uses fastembed (BAAI/bge-small-en-v1.5, dim=384) for zero-cost local embeddings. Falls back to API embedder if configured.

```python
def make_local_embedder() -> EmbedFn:
    """Local embeddings via fastembed. No API key needed."""
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    
    async def embed(text: str) -> List[float]:
        # fastembed is sync, run in executor
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()
    
    return embed

def make_batch_local_embedder() -> BatchEmbedFn:
    """Batch embed for ingestion efficiency."""
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    
    async def embed_batch(texts: List[str]) -> List[List[float]]:
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]
    
    return embed_batch
```

### 2. Message Vector Store (new table in `store/vector_store.py`)

**Current:** `summary_embeddings` table stores embeddings keyed by `summary_id`.

**Change:** Add `message_embeddings` table storing embeddings keyed by `message_id`.

```sql
CREATE TABLE IF NOT EXISTS message_embeddings (
    message_id    TEXT PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    embedding     vector(384),     -- bge-small-en-v1.5 dim
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_msg_embed_ann
    ON message_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_msg_embed_conv
    ON message_embeddings (conversation_id);
```

New methods on VectorStore:
```python
def store_message(self, message_id: str, conversation_id: int, embedding: List[float])
def store_messages_batch(self, items: List[Tuple[str, int, List[float]]])
def search_messages(self, query_embedding: List[float], top_k: int = 20,
                    exclude_conversation_id: int = None,
                    conversation_ids: List[int] = None,
                    min_score: float = 0.35) -> List[Tuple[str, float]]
```

### 3. Ingestion Hook (modify `mcp_server.py` → `lcm_ingest`)

**Current:** `lcm_ingest` appends messages → runs compaction if threshold met.

**Change:** After appending messages, immediately embed them (batch) and store vectors. Compaction still runs independently.

```python
async def lcm_ingest(session_key, messages, ...):
    # 1. Append messages (existing)
    msg_ids = store.append_messages(...)
    
    # 2. NEW: Embed and store vectors (local, fast, no LLM)
    if vector_store and embed_fn:
        texts = [m.content for m in messages if m.content]
        embeddings = await embed_batch_fn(texts)
        vector_store.store_messages_batch([
            (msg_id, conv_id, emb) for msg_id, emb in zip(msg_ids, embeddings)
        ])
    
    # 3. Compaction (existing, unchanged)
    await maybe_compact(...)
```

### 4. Hybrid Retrieval (modify `tools/recall.py` → `lcm_grep`)

**Current:** FTS5 keyword search only.

**Change:** Add vector search as a parallel retrieval path, merge results.

```python
async def lcm_grep(query, ...):
    results = []
    
    # Path 1: FTS5 keyword search (existing)
    fts_results = fts_search(query, ...)
    
    # Path 2: Vector similarity search (NEW)
    if vector_store and embed_fn:
        query_embedding = await embed_fn(query)
        vec_results = vector_store.search_messages(query_embedding, top_k=20)
        # Fetch actual message content for vector hits
        vec_messages = message_store.get_by_ids([r.id for r in vec_results])
    
    # Path 3: Merge with Reciprocal Rank Fusion
    merged = reciprocal_rank_fusion(fts_results, vec_results, k=60)
    
    return merged
```

### 5. Reciprocal Rank Fusion (new `engine/fusion.py`)

Simple RRF implementation for merging ranked lists:

```python
def reciprocal_rank_fusion(
    *result_lists: List[Tuple[str, float]],  # [(id, score), ...]
    k: int = 60
) -> List[Tuple[str, float]]:
    """Merge multiple ranked result lists using RRF."""
    scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 6. Config Changes (modify `config.py`)

New config fields:
```python
# Raw vector retrieval
raw_vector_enabled: bool = True           # LCM_RAW_VECTOR_ENABLED
raw_vector_model: str = "BAAI/bge-small-en-v1.5"  # LCM_RAW_VECTOR_MODEL
raw_vector_dim: int = 384                 # LCM_RAW_VECTOR_DIM
raw_vector_top_k: int = 20               # LCM_RAW_VECTOR_TOP_K
raw_vector_min_score: float = 0.35        # LCM_RAW_VECTOR_MIN_SCORE
raw_vector_use_local: bool = True         # LCM_RAW_VECTOR_USE_LOCAL (fastembed vs API)
```

## What This Does NOT Change

- **DAG compaction**: Still runs as before for context window assembly
- **PR #2 summary embeddings**: Still available for cross-session semantic search over summaries
- **FTS5 search**: Still the primary keyword path, now augmented with vectors
- **Context assembly**: ContextAssembler still uses DAG summaries for building the LLM context window

## Dependencies

- `fastembed` (already installed from PR #2 benchmark work)
- `pgvector` (already installed from PR #2)
- PostgreSQL with vector extension (already set up)

## Migration

- New table `message_embeddings` created automatically on first use
- Existing messages can be backfilled with a migration script
- Zero-downtime: if vectors aren't available, falls back to FTS5 only

## Expected Impact on LongMemEval

- Ingestion: ~50 LLM calls/question → 0 LLM calls (local embeddings only)
- Retrieval: FTS5 keyword misses → FTS5 + vector hybrid catches semantic matches
- Runtime: ~20 hours → ~1-2 hours (back to original benchmark speed)
- Estimated score improvement: 41.4% → 60-75% (based on CortiLoop's +15% from multi-path retrieval)

## Future PRs (not in scope)

- PR #4: Query decomposition (+7% expected)
- PR #5: LLM reranking (+5% expected)  
- PR #6: Knowledge graph / entity extraction
- PR #7: Contradiction resolution for knowledge-update questions
