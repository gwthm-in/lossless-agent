# API Reference

## Adapters

### SimpleAdapter

```python
class SimpleAdapter:
    def __init__(self, db_path: str, summarize_fn: SummarizeFn,
                 config: Optional[LCMConfig] = None) -> None: ...

    async def ingest(self, session_key: str, messages: List[dict]) -> None: ...
    async def retrieve(self, session_key: str, budget_tokens: int) -> Optional[str]: ...
    async def compact(self, session_key: str) -> int: ...
    async def search(self, query: str, session_key: Optional[str] = None) -> List[dict]: ...
    async def expand(self, summary_id: str) -> dict: ...
    def close(self) -> None: ...
```

**Parameters:**

- `db_path` — path to the SQLite database file
- `summarize_fn` — async callable `(str) -> str` that summarises text
- `config` — optional LCMConfig; defaults are used if omitted

### GenericAdapter

```python
class GenericAdapter(AgentAdapter):
    def __init__(self, config: LCMConfig, summarize_fn: SummarizeFn) -> None: ...

    # AgentAdapter lifecycle
    async def on_turn_start(self, session_key: str, user_message: str) -> Optional[str]: ...
    async def on_turn_end(self, session_key: str, messages: List[dict]) -> None: ...
    async def on_session_end(self, session_key: str) -> None: ...

    # AgentAdapter tools
    def get_tools(self) -> List[dict]: ...
    async def handle_tool_call(self, name: str, arguments: dict) -> str: ...
    def get_system_prompt_block(self) -> str: ...

    # Convenience methods
    async def store_message(self, session_key: str, role: str, content: str,
                            token_count: int = 0, tool_call_id: Optional[str] = None,
                            tool_name: Optional[str] = None) -> None: ...
    async def get_context(self, session_key: str, max_tokens: int) -> Optional[str]: ...
    async def force_compact(self, session_key: str) -> None: ...
    async def get_stats(self, session_key: str) -> dict: ...
```

### AgentAdapter (ABC)

```python
class AgentAdapter(ABC):
    async def on_turn_start(self, session_key: str, user_message: str) -> Optional[str]: ...
    async def on_turn_end(self, session_key: str, messages: List[dict]) -> None: ...
    async def on_session_end(self, session_key: str) -> None: ...
    def get_tools(self) -> List[dict]: ...
    async def handle_tool_call(self, name: str, arguments: dict) -> str: ...
    def get_system_prompt_block(self) -> str: ...
```

## Engine

### CompactionEngine

```python
class CompactionEngine:
    def __init__(self, msg_store: AbstractMessageStore,
                 sum_store: AbstractSummaryStore,
                 summarize_fn: SummarizeFn,
                 config: CompactionConfig | None = None) -> None: ...

    def select_chunk(self, conv_id: int) -> List[Message]: ...
    def needs_compaction(self, conv_id: int, context_limit: int) -> bool: ...
    async def compact_leaf(self, conv_id: int) -> Optional[Summary]: ...
    async def compact_condensed(self, conv_id: int, depth: int = 0) -> Optional[Summary]: ...
    async def run_incremental(self, conv_id: int, context_limit: int) -> List[Summary]: ...
```

**CompactionConfig fields:**

| Field | Type | Default |
|-------|------|---------|
| `fresh_tail_count` | int | 8 |
| `leaf_chunk_tokens` | int | 20000 |
| `leaf_min_fanout` | int | 4 |
| `condensed_min_fanout` | int | 3 |
| `context_threshold` | float | 0.75 |
| `leaf_target_tokens` | int | 1200 |
| `condensed_target_tokens` | int | 2000 |

### ContextAssembler

```python
class ContextAssembler:
    def __init__(self, msg_store: AbstractMessageStore,
                 sum_store: AbstractSummaryStore,
                 config: AssemblerConfig) -> None: ...

    def assemble(self, conv_id: int) -> AssembledContext: ...
    def format_context(self, assembled: AssembledContext) -> str: ...
```

**AssemblerConfig fields:**

| Field | Type | Default |
|-------|------|---------|
| `max_context_tokens` | int | *(required)* |
| `summary_budget_ratio` | float | 0.4 |
| `fresh_tail_count` | int | 8 |

**AssembledContext:**

```python
@dataclass
class AssembledContext:
    summaries: List[Summary]
    messages: List[Message]
    total_tokens: int
```

## Recall tools

### lcm_grep

```python
def lcm_grep(
    db: Database,
    query: str,
    scope: str = "all",           # "all", "messages", or "summaries"
    conversation_id: Optional[int] = None,
    limit: int = 20,
) -> List[GrepResult]: ...
```

**GrepResult:**

```python
@dataclass
class GrepResult:
    type: str                # "message" or "summary"
    id: Union[str, int]
    content_snippet: str     # truncated to 200 chars
    conversation_id: int
    metadata: Dict[str, Any]
```

### lcm_describe

```python
def lcm_describe(db: Database, summary_id: str) -> Optional[DescribeResult]: ...
```

**DescribeResult:**

```python
@dataclass
class DescribeResult:
    summary_id: str
    kind: str                   # "leaf" or "condensed"
    depth: int
    content: str
    token_count: int
    source_token_count: int
    earliest_at: str
    latest_at: str
    child_ids: List[str]
    source_message_count: int
```

### lcm_expand

```python
def lcm_expand(db: Database, summary_id: str) -> Optional[ExpandResult]: ...
```

**ExpandResult:**

```python
@dataclass
class ExpandResult:
    summary_id: str
    kind: str
    children: List[Union[Message, Summary]]
```

## Vector Store

### VectorStore

pgvector-backed embedding store for cross-session retrieval.

```python
class VectorStore:
    def __init__(self, dsn: str, embedding_dim: int = 1536) -> None: ...

    def ensure_table(self) -> None: ...
    def upsert(self, summary_id: str, embedding: List[float],
               conversation_id: int, content: str) -> None: ...
    def query(self, embedding: List[float], top_k: int = 5,
              exclude_conversation_id: Optional[int] = None) -> List[dict]: ...
```

**Parameters:**

- `dsn` — Postgres connection string
- `embedding_dim` — vector dimension (must match your embedding model)

**query() returns** a list of dicts with keys: `summary_id`, `conversation_id`, `content`, `distance`.

### Embedder

```python
class Embedder:
    def __init__(self, base_url: str, model: str, api_key: str,
                 dim: int = 1536) -> None: ...

    def embed(self, text: str) -> List[float]: ...

def create_embedder(config: LCMConfig) -> Optional[Embedder]: ...
```

**Parameters:**

- `base_url` — OpenAI-compatible API base URL
- `model` — embedding model name
- `api_key` — API key (falls back to `OPENAI_API_KEY` env var)
- `dim` — embedding dimension

`create_embedder()` returns `None` if cross-session retrieval is disabled.

## Configuration

### LCMConfig

```python
@dataclass
class LCMConfig:
    # Core
    enabled: bool = True
    db_path: str = "~/.lossless-agent/lcm.db"
    database_dsn: str = ""            # Postgres DSN; when set, uses Postgres
    max_context_tokens: int = 128_000
    ignore_session_patterns: List[str] = field(default_factory=list)

    # Compaction
    fresh_tail_count: int = 64
    leaf_chunk_tokens: int = 20_000
    leaf_min_fanout: int = 8
    condensed_min_fanout: int = 4
    context_threshold: float = 0.75
    leaf_target_tokens: int = 2400
    condensed_target_tokens: int = 2000
    incremental_max_depth: int = 1
    summary_timeout_ms: int = 60_000
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_ms: int = 1_800_000

    # Summarisation
    summary_provider: str = ""        # "anthropic", "openai", or "" (truncation)
    summary_model: str = ""           # model name; provider default used when empty
    summary_base_url: str = ""        # OpenAI-compatible base URL (e.g. LiteLLM proxy)
    expansion_model: str = ""         # for lcm_expand_query; falls back to summary_model
    summary_budget_ratio: float = 0.4

    # Context assembly
    max_assembly_token_budget: Optional[int] = None
    custom_instructions: str = ""
    timezone: str = ""

    # Cross-session semantic retrieval (requires pgvector)
    cross_session_enabled: bool = False
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""       # falls back to OPENAI_API_KEY
    cross_session_top_k: int = 5
    cross_session_token_budget: int = 2000
    cross_session_min_score: float = 0.70

    # Raw vector retrieval (local embeddings, requires pgvector)
    raw_vector_enabled: bool = False
    raw_vector_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    raw_vector_dim: int = 1024
    raw_vector_top_k: int = 20
    raw_vector_min_score: float = 0.35
    raw_vector_use_local: bool = True  # use fastembed; False = API endpoint

    @classmethod
    def from_env(cls) -> LCMConfig: ...
    @classmethod
    def from_dict(cls, d: dict) -> LCMConfig: ...
    @staticmethod
    def merge(base: LCMConfig, overrides: dict) -> LCMConfig: ...
    def validate(self) -> List[str]: ...
    def to_compaction_config(self) -> CompactionConfig: ...
    def to_assembler_config(self) -> AssemblerConfig: ...
    @property
    def resolved_db_path(self) -> str: ...
    @property
    def effective_bootstrap_max_tokens(self) -> int: ...
```
