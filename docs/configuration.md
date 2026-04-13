# Configuration

All configuration lives in `LCMConfig`. You can set values directly,
load from environment variables, or merge multiple sources.

## Creating a config

```python
from lossless_agent.config import LCMConfig

# Defaults
config = LCMConfig()

# Explicit values
config = LCMConfig(
    db_path="memory.db",
    max_context_tokens=64_000,
    fresh_tail_count=10,
)

# From environment variables
config = LCMConfig.from_env()

# From a dictionary (e.g. parsed TOML)
config = LCMConfig.from_dict({"db_path": "memory.db"})

# Merge overrides onto a base config
config = LCMConfig.merge(base_config, {"max_context_tokens": 64_000})
```

## All options

### Core

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `enabled` | `LCM_ENABLED` | `True` | Master switch to enable/disable lossless memory |
| `db_path` | `LCM_DATABASE_PATH` | `~/.lossless-agent/lcm.db` | Path to the SQLite database file |
| `database_dsn` | `LCM_DATABASE_DSN` | `""` | Postgres DSN (e.g. `postgresql://user:pass@host/db`). When set, uses Postgres instead of SQLite |
| `max_context_tokens` | `LCM_MAX_CONTEXT_TOKENS` | `128000` | Maximum context window size in tokens |
| `ignore_session_patterns` | `LCM_IGNORE_SESSION_PATTERNS` | `[]` | Comma-separated glob patterns for sessions to skip entirely |

### Compaction

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `fresh_tail_count` | `LCM_FRESH_TAIL_COUNT` | `64` | Number of recent messages to keep uncompacted |
| `leaf_chunk_tokens` | `LCM_LEAF_CHUNK_TOKENS` | `20000` | Max tokens per leaf compaction chunk |
| `leaf_min_fanout` | `LCM_LEAF_MIN_FANOUT` | `8` | Minimum messages needed to trigger leaf compaction |
| `condensed_min_fanout` | `LCM_CONDENSED_MIN_FANOUT` | `4` | Minimum orphan summaries needed for condensed compaction |
| `context_threshold` | `LCM_CONTEXT_THRESHOLD` | `0.75` | Fraction of context limit that triggers compaction (0.0–1.0) |
| `leaf_target_tokens` | `LCM_LEAF_TARGET_TOKENS` | `2400` | Target token count for leaf summaries |
| `condensed_target_tokens` | `LCM_CONDENSED_TARGET_TOKENS` | `2000` | Target token count for condensed summaries |
| `incremental_max_depth` | `LCM_INCREMENTAL_MAX_DEPTH` | `1` | Max condensed compaction depth per incremental cycle |
| `summary_timeout_ms` | `LCM_SUMMARY_TIMEOUT_MS` | `60000` | Timeout in ms for summarisation LLM calls |
| `circuit_breaker_threshold` | `LCM_CIRCUIT_BREAKER_THRESHOLD` | `5` | Consecutive summarisation failures before circuit opens |
| `circuit_breaker_cooldown_ms` | `LCM_CIRCUIT_BREAKER_COOLDOWN_MS` | `1800000` | Circuit breaker cooldown period in ms |

### Summarisation

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `summary_provider` | `LCM_SUMMARY_PROVIDER` | `""` | LLM provider: `anthropic`, `openai`, or `""` (truncation fallback) |
| `summary_model` | `LCM_SUMMARY_MODEL` | `""` | Model name for summarisation (provider-specific default if empty) |
| `summary_base_url` | `LCM_SUMMARY_BASE_URL` | `""` | Base URL for OpenAI-compatible summarisation endpoint (e.g. LiteLLM proxy) |
| `expansion_model` | `LCM_EXPANSION_MODEL` | `""` | Model for `lcm_expand_query`; falls back to `summary_model` |
| `summary_budget_ratio` | `LCM_SUMMARY_BUDGET_RATIO` | `0.4` | Fraction of remaining context budget allocated to summaries (0.0–1.0) |

**Provider defaults when model is not set:**
- `anthropic` → `claude-haiku-4-5-20251001`
- `openai` → `gpt-4o-mini`

**LiteLLM / OpenAI-compatible proxy:** Set `summary_provider=openai` and point `summary_base_url`
at your proxy. Any model string accepted by LiteLLM works (e.g. `anthropic/claude-haiku-4-5-20251001`,
`bedrock/anthropic.claude-3-haiku`, `ollama/llama3`).

```bash
export LCM_SUMMARY_PROVIDER=openai
export LCM_SUMMARY_BASE_URL=http://localhost:4000
export LCM_SUMMARY_MODEL=anthropic/claude-haiku-4-5-20251001
```

### Context assembly

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `max_assembly_token_budget` | `LCM_MAX_ASSEMBLY_TOKEN_BUDGET` | `None` | Hard cap on assembled context tokens (overrides `max_context_tokens`) |
| `custom_instructions` | `LCM_CUSTOM_INSTRUCTIONS` | `""` | Extra instructions injected into every summarisation prompt |
| `timezone` | `LCM_TIMEZONE` | `""` | Timezone for timestamp formatting in assembled context (e.g. `Asia/Kolkata`) |

### Cross-session semantic retrieval

Requires Postgres + pgvector. See [Semantic Retrieval](semantic-retrieval.md) for full setup.

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `cross_session_enabled` | `LCM_CROSS_SESSION_ENABLED` | `False` | Enable cross-session semantic retrieval via pgvector |
| `embedding_base_url` | `LCM_EMBEDDING_BASE_URL` | `""` | Base URL for OpenAI-compatible embedding API |
| `embedding_model` | `LCM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `embedding_dim` | `LCM_EMBEDDING_DIM` | `1536` | Dimension of embedding vectors |
| `embedding_api_key` | `LCM_EMBEDDING_API_KEY` | `""` | API key for embedding endpoint (falls back to `OPENAI_API_KEY`) |
| `cross_session_top_k` | `LCM_CROSS_SESSION_TOP_K` | `5` | Number of cross-session results to retrieve |
| `cross_session_token_budget` | `LCM_CROSS_SESSION_TOKEN_BUDGET` | `2000` | Max tokens for cross-session context |
| `cross_session_min_score` | `LCM_CROSS_SESSION_MIN_SCORE` | `0.70` | Minimum cosine similarity to include a cross-session hit |

### Raw vector retrieval (local embeddings)

Embeds messages at ingestion time using a local model (no API key needed). Requires pgvector.

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `raw_vector_enabled` | `LCM_RAW_VECTOR_ENABLED` | `False` | Enable raw vector retrieval (opt-in) |
| `raw_vector_model` | `LCM_RAW_VECTOR_MODEL` | `mixedbread-ai/mxbai-embed-large-v1` | Local embedding model (via fastembed) |
| `raw_vector_dim` | `LCM_RAW_VECTOR_DIM` | `1024` | Dimension of raw vectors |
| `raw_vector_top_k` | `LCM_RAW_VECTOR_TOP_K` | `20` | Number of raw vector results to retrieve |
| `raw_vector_min_score` | `LCM_RAW_VECTOR_MIN_SCORE` | `0.35` | Minimum cosine similarity to include a raw vector hit |
| `raw_vector_use_local` | `LCM_RAW_VECTOR_USE_LOCAL` | `True` | Use fastembed (local, no API key); set `False` to use an API endpoint |

Install local embeddings support:

```bash
pip install 'lossless-agent[local-embeddings]'
```

## Validation

Call `config.validate()` to check for invalid values. It returns a list
of error strings (empty if valid):

```python
config = LCMConfig(context_threshold=1.5)
errors = config.validate()
# ["context_threshold must be between 0.0 and 1.0"]
```

## Derived configs

LCMConfig produces specialised configs for the engine layer:

```python
compaction_cfg = config.to_compaction_config()  # CompactionConfig
assembler_cfg = config.to_assembler_config()    # AssemblerConfig
```

These are also available as properties for backward compatibility:

```python
config.compaction   # same as to_compaction_config()
config.assembler    # same as to_assembler_config()
```
