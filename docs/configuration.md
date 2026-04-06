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

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `enabled` | `LCM_ENABLED` | `True` | Master switch to enable/disable lossless memory |
| `db_path` | `LCM_DATABASE_PATH` | `~/.lossless-agent/lcm.db` | Path to the SQLite database file |
| `fresh_tail_count` | `LCM_FRESH_TAIL_COUNT` | `8` | Number of recent messages to keep uncompacted |
| `leaf_chunk_tokens` | `LCM_LEAF_CHUNK_TOKENS` | `20000` | Max tokens per leaf compaction chunk |
| `leaf_min_fanout` | `LCM_LEAF_MIN_FANOUT` | `4` | Minimum messages needed to trigger leaf compaction |
| `condensed_min_fanout` | `LCM_CONDENSED_MIN_FANOUT` | `3` | Minimum orphan summaries needed for condensed compaction |
| `context_threshold` | `LCM_CONTEXT_THRESHOLD` | `0.75` | Fraction of context limit that triggers compaction (0.0–1.0) |
| `leaf_target_tokens` | `LCM_LEAF_TARGET_TOKENS` | `1200` | Target token count for leaf summaries |
| `condensed_target_tokens` | `LCM_CONDENSED_TARGET_TOKENS` | `2000` | Target token count for condensed summaries |
| `max_context_tokens` | `LCM_MAX_CONTEXT_TOKENS` | `128000` | Maximum context window size in tokens |
| `summary_budget_ratio` | `LCM_SUMMARY_BUDGET_RATIO` | `0.4` | Fraction of remaining budget allocated to summaries (0.0–1.0) |
| `summary_model` | `LCM_SUMMARY_MODEL` | `""` | Model name to use for summarisation |
| `summary_provider` | `LCM_SUMMARY_PROVIDER` | `""` | Provider for the summarisation model |
| `expansion_model` | `LCM_EXPANSION_MODEL` | `""` | Model name for query expansion |
| `ignore_session_patterns` | `LCM_IGNORE_SESSION_PATTERNS` | `[]` | Comma-separated glob patterns for sessions to skip |
| `incremental_max_depth` | `LCM_INCREMENTAL_MAX_DEPTH` | `1` | Max condensed compaction depth per incremental cycle |
| `summary_timeout_ms` | `LCM_SUMMARY_TIMEOUT_MS` | `60000` | Timeout in ms for summarisation calls |

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
