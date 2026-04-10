# Semantic Cross-Session Retrieval

lossless-agent can optionally retrieve relevant context from **other sessions**
using vector similarity search. When enabled, summaries are embedded at
compaction time and the assembler pulls in cross-session context alongside
the current session's DAG.

## Prerequisites

1. **Postgres with pgvector** — the vector store uses `pgvector` for
   embedding storage and similarity search.

2. **Install Postgres dependencies:**

```bash
pip install lossless-agent[postgres]
```

This pulls in `psycopg2-binary`.

3. **Enable the pgvector extension** in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## How it works

```
Compaction creates a summary
        │
        ▼
  _maybe_embed()  ←── fires after leaf and condensed creation
        │
        ▼
  Embedder.embed(text)  ←── calls any OpenAI-compatible endpoint
        │
        ▼
  VectorStore.upsert(summary_id, embedding)
        │
        ▼
  Stored in pgvector with HNSW index (IVFFlat fallback)
```

At assembly time:

```
ContextAssembler.assemble(conv_id)
        │
        ├── normal DAG summaries + tail messages (same as before)
        │
        └── cross_session_context(conv_id)
                │
                ├── embed the current session's latest messages
                ├── VectorStore.query(embedding, top_k)
                └── return summaries from OTHER sessions
                    (filtered by token budget)
```

## Configuration

Enable cross-session retrieval and point at your embedding provider:

```python
from lossless_agent.config import LCMConfig

config = LCMConfig(
    cross_session_enabled=True,
    embedding_base_url="https://api.openai.com/v1",
    embedding_model="text-embedding-3-small",
    embedding_dim=1536,
    embedding_api_key="sk-...",       # falls back to OPENAI_API_KEY
    cross_session_top_k=5,
    cross_session_token_budget=2000,
)
```

Or via environment variables:

```bash
export LCM_CROSS_SESSION_ENABLED=true
export LCM_EMBEDDING_BASE_URL=https://api.openai.com/v1
export LCM_EMBEDDING_MODEL=text-embedding-3-small
export LCM_EMBEDDING_DIM=1536
export LCM_EMBEDDING_API_KEY=sk-...
export LCM_CROSS_SESSION_TOP_K=5
export LCM_CROSS_SESSION_TOKEN_BUDGET=2000
```

See [Configuration](configuration.md) for the full option reference.

## Key classes

### VectorStore (`store/vector_store.py`)

pgvector-backed store with HNSW indexing and IVFFlat fallback.

```python
class VectorStore:
    def __init__(self, dsn: str, embedding_dim: int = 1536) -> None: ...
    def ensure_table(self) -> None: ...
    def upsert(self, summary_id: str, embedding: List[float],
               conversation_id: int, content: str) -> None: ...
    def query(self, embedding: List[float], top_k: int = 5,
              exclude_conversation_id: Optional[int] = None) -> List[dict]: ...
```

### Embedder (`engine/embedder.py`)

Factory-pattern embedder that works with any OpenAI-compatible endpoint.
Uses `urllib` — no extra HTTP dependencies.

```python
class Embedder:
    def __init__(self, base_url: str, model: str, api_key: str,
                 dim: int = 1536) -> None: ...
    def embed(self, text: str) -> List[float]: ...

def create_embedder(config: LCMConfig) -> Optional[Embedder]: ...
```

### Assembler cross-session method

`ContextAssembler.cross_session_context(conv_id)` embeds the current
session's recent messages, queries the vector store for similar summaries
from other sessions, and returns them within the configured token budget.

## Using a local embedding server

Any OpenAI-compatible embedding API works. For example, with
[Ollama](https://ollama.ai):

```bash
export LCM_EMBEDDING_BASE_URL=http://localhost:11434/v1
export LCM_EMBEDDING_MODEL=nomic-embed-text
export LCM_EMBEDDING_DIM=768
```

## Disabling cross-session retrieval

Cross-session retrieval is **off by default**. If you've enabled it and
want to turn it off:

```bash
export LCM_CROSS_SESSION_ENABLED=false
```

The existing summaries and embeddings remain in Postgres but won't be
queried during assembly.
