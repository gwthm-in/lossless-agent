"""Canonical configuration for the LCM system.

LCMConfig consolidates all tunables with a clear precedence hierarchy:
    env vars  >  explicit config  >  defaults
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List, Optional

from lossless_agent.engine.compaction import CompactionConfig
from lossless_agent.engine.assembler import AssemblerConfig


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.strip().lower() in ("true", "1", "yes")


def _parse_optional_int(value: str) -> Optional[int]:
    """Parse an optional int: empty string -> None, otherwise int."""
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


@dataclass
class LCMConfig:
    """Combined configuration for the LCM system.

    Every field has a corresponding ``LCM_*`` environment variable that
    takes precedence over values supplied at construction time.
    """

    enabled: bool = True
    db_path: str = "~/.lossless-agent/lcm.db"
    fresh_tail_count: int = 64
    leaf_chunk_tokens: int = 20_000
    leaf_min_fanout: int = 8
    condensed_min_fanout: int = 4
    context_threshold: float = 0.75
    leaf_target_tokens: int = 2400
    condensed_target_tokens: int = 2000
    max_context_tokens: int = 128_000
    summary_budget_ratio: float = 0.4
    summary_model: str = ""
    summary_provider: str = ""
    expansion_model: str = ""
    ignore_session_patterns: List[str] = field(default_factory=list)
    incremental_max_depth: int = 1
    summary_timeout_ms: int = 60_000
    summary_max_overage_factor: float = 3.0
    condensed_min_fanout_hard: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_ms: int = 1_800_000

    # New config options
    stateless_session_patterns: List[str] = field(default_factory=list)
    skip_stateless_sessions: bool = True
    new_session_retain_depth: int = 2
    bootstrap_max_tokens: int = 6000
    large_file_summary_provider: str = ""
    large_file_summary_model: str = ""
    delegation_timeout_ms: int = 120_000
    prune_heartbeat_ok: bool = False
    max_assembly_token_budget: Optional[int] = None
    custom_instructions: str = ""
    timezone: str = ""
    database_dsn: str = ""  # Postgres DSN, e.g. postgresql://user:pass@host/db

    # Semantic / cross-session retrieval (requires pgvector + database_dsn)
    cross_session_enabled: bool = False
    embedding_base_url: str = ""   # OpenAI-compatible /embeddings base URL
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""    # API key; falls back to OPENAI_API_KEY env var
    cross_session_top_k: int = 5
    cross_session_token_budget: int = 2000
    cross_session_min_score: float = 0.70  # discard hits below this cosine similarity

    # Raw vector retrieval (embeds messages at ingestion time, no LLM calls)
    raw_vector_enabled: bool = False  # opt-in; requires pgvector + database_dsn
    raw_vector_model: str = "BAAI/bge-small-en-v1.5"
    raw_vector_dim: int = 384
    raw_vector_top_k: int = 20
    raw_vector_min_score: float = 0.35
    raw_vector_use_local: bool = True  # use fastembed (no API key needed)

    # ------------------------------------------------------------------
    # Env-var mapping
    # ------------------------------------------------------------------

    _COMMA_LIST_FIELDS = {"ignore_session_patterns", "stateless_session_patterns"}

    _ENV_MAP = {
        "enabled": ("LCM_ENABLED", _parse_bool),
        "db_path": ("LCM_DATABASE_PATH", str),
        "fresh_tail_count": ("LCM_FRESH_TAIL_COUNT", int),
        "leaf_chunk_tokens": ("LCM_LEAF_CHUNK_TOKENS", int),
        "leaf_min_fanout": ("LCM_LEAF_MIN_FANOUT", int),
        "condensed_min_fanout": ("LCM_CONDENSED_MIN_FANOUT", int),
        "context_threshold": ("LCM_CONTEXT_THRESHOLD", float),
        "leaf_target_tokens": ("LCM_LEAF_TARGET_TOKENS", int),
        "condensed_target_tokens": ("LCM_CONDENSED_TARGET_TOKENS", int),
        "max_context_tokens": ("LCM_MAX_CONTEXT_TOKENS", int),
        "summary_budget_ratio": ("LCM_SUMMARY_BUDGET_RATIO", float),
        "summary_model": ("LCM_SUMMARY_MODEL", str),
        "summary_provider": ("LCM_SUMMARY_PROVIDER", str),
        "expansion_model": ("LCM_EXPANSION_MODEL", str),
        "ignore_session_patterns": ("LCM_IGNORE_SESSION_PATTERNS", None),
        "incremental_max_depth": ("LCM_INCREMENTAL_MAX_DEPTH", int),
        "summary_timeout_ms": ("LCM_SUMMARY_TIMEOUT_MS", int),
        "summary_max_overage_factor": ("LCM_SUMMARY_MAX_OVERAGE_FACTOR", float),
        "condensed_min_fanout_hard": ("LCM_CONDENSED_MIN_FANOUT_HARD", int),
        "circuit_breaker_threshold": ("LCM_CIRCUIT_BREAKER_THRESHOLD", int),
        "circuit_breaker_cooldown_ms": ("LCM_CIRCUIT_BREAKER_COOLDOWN_MS", int),
        "stateless_session_patterns": ("LCM_STATELESS_SESSION_PATTERNS", None),
        "skip_stateless_sessions": ("LCM_SKIP_STATELESS_SESSIONS", _parse_bool),
        "new_session_retain_depth": ("LCM_NEW_SESSION_RETAIN_DEPTH", int),
        "bootstrap_max_tokens": ("LCM_BOOTSTRAP_MAX_TOKENS", int),
        "large_file_summary_provider": ("LCM_LARGE_FILE_SUMMARY_PROVIDER", str),
        "large_file_summary_model": ("LCM_LARGE_FILE_SUMMARY_MODEL", str),
        "delegation_timeout_ms": ("LCM_DELEGATION_TIMEOUT_MS", int),
        "prune_heartbeat_ok": ("LCM_PRUNE_HEARTBEAT_OK", _parse_bool),
        "max_assembly_token_budget": ("LCM_MAX_ASSEMBLY_TOKEN_BUDGET", _parse_optional_int),
        "custom_instructions": ("LCM_CUSTOM_INSTRUCTIONS", str),
        "timezone": ("LCM_TIMEZONE", str),
        "database_dsn": ("LCM_DATABASE_DSN", str),
        "cross_session_enabled": ("LCM_CROSS_SESSION_ENABLED", _parse_bool),
        "embedding_base_url": ("LCM_EMBEDDING_BASE_URL", str),
        "embedding_model": ("LCM_EMBEDDING_MODEL", str),
        "embedding_dim": ("LCM_EMBEDDING_DIM", int),
        "embedding_api_key": ("LCM_EMBEDDING_API_KEY", str),
        "cross_session_top_k": ("LCM_CROSS_SESSION_TOP_K", int),
        "cross_session_token_budget": ("LCM_CROSS_SESSION_TOKEN_BUDGET", int),
        "cross_session_min_score": ("LCM_CROSS_SESSION_MIN_SCORE", float),
        "raw_vector_enabled": ("LCM_RAW_VECTOR_ENABLED", _parse_bool),
        "raw_vector_model": ("LCM_RAW_VECTOR_MODEL", str),
        "raw_vector_dim": ("LCM_RAW_VECTOR_DIM", int),
        "raw_vector_top_k": ("LCM_RAW_VECTOR_TOP_K", int),
        "raw_vector_min_score": ("LCM_RAW_VECTOR_MIN_SCORE", float),
        "raw_vector_use_local": ("LCM_RAW_VECTOR_USE_LOCAL", _parse_bool),
    }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "LCMConfig":
        """Create a config by reading ``LCM_*`` environment variables.

        Falls back to field defaults for any variable that is not set.
        """
        kwargs: dict = {}
        for field_name, (env_var, converter) in cls._ENV_MAP.items():
            raw = os.environ.get(env_var)
            if raw is None:
                continue
            if field_name in cls._COMMA_LIST_FIELDS:
                kwargs[field_name] = [
                    p.strip() for p in raw.split(",") if p.strip()
                ]
            else:
                kwargs[field_name] = converter(raw)  # type: ignore[operator]
        # embedding_api_key falls back to OPENAI_API_KEY when not explicitly set
        if "embedding_api_key" not in kwargs:
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                kwargs["embedding_api_key"] = openai_key
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "LCMConfig":
        """Create a config from a plain dictionary (e.g. parsed TOML/JSON)."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @staticmethod
    def merge(base: "LCMConfig", overrides: dict) -> "LCMConfig":
        """Return a new config with *overrides* applied on top of *base*."""
        merged = asdict(base)
        valid_fields = {f.name for f in fields(LCMConfig)}
        for k, v in overrides.items():
            if k in valid_fields:
                merged[k] = v
        return LCMConfig(**merged)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_compaction_config(self) -> CompactionConfig:
        """Map flat LCMConfig fields to the engine's CompactionConfig."""
        return CompactionConfig(
            fresh_tail_count=self.fresh_tail_count,
            leaf_chunk_tokens=self.leaf_chunk_tokens,
            leaf_min_fanout=self.leaf_min_fanout,
            condensed_min_fanout=self.condensed_min_fanout,
            condensed_min_fanout_hard=self.condensed_min_fanout_hard,
            context_threshold=self.context_threshold,
            leaf_target_tokens=self.leaf_target_tokens,
            condensed_target_tokens=self.condensed_target_tokens,
            summary_max_overage_factor=self.summary_max_overage_factor,
            summary_timeout_ms=self.summary_timeout_ms,
            custom_instructions=self.custom_instructions,
        )

    def to_expand_query_config(self):
        """Map flat LCMConfig fields to ExpandQueryConfig."""
        from lossless_agent.tools.expand_query import ExpandQueryConfig
        return ExpandQueryConfig(
            timeout_ms=self.delegation_timeout_ms,
            expansion_model=self.expansion_model,
        )

    def to_large_file_config(self):
        """Map flat LCMConfig fields to LargeFileConfig."""
        from lossless_agent.engine.large_files import LargeFileConfig
        return LargeFileConfig(
            summary_provider=self.large_file_summary_provider,
            summary_model=self.large_file_summary_model,
        )

    def to_assembler_config(self) -> AssemblerConfig:
        """Map flat LCMConfig fields to the engine's AssemblerConfig."""
        return AssemblerConfig(
            max_context_tokens=self.max_context_tokens,
            summary_budget_ratio=self.summary_budget_ratio,
            fresh_tail_count=self.fresh_tail_count,
            max_assembly_token_budget=self.max_assembly_token_budget,
        )

    # ------------------------------------------------------------------
    # Backward-compatible properties used by adapters
    # ------------------------------------------------------------------

    @property
    def compaction(self) -> CompactionConfig:
        """Backward-compatible access for adapters."""
        return self.to_compaction_config()

    @property
    def assembler(self) -> AssemblerConfig:
        """Backward-compatible access for adapters."""
        return self.to_assembler_config()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of validation errors (empty if the config is valid)."""
        errors: List[str] = []
        if not (0.0 <= self.context_threshold <= 1.0):
            errors.append(
                "context_threshold must be between 0.0 and 1.0"
            )
        if not (0.0 <= self.summary_budget_ratio <= 1.0):
            errors.append(
                "summary_budget_ratio must be between 0.0 and 1.0"
            )
        if self.fresh_tail_count < 1:
            errors.append("fresh_tail_count must be >= 1")
        if self.leaf_min_fanout < 2:
            errors.append("leaf_min_fanout must be >= 2")
        if self.condensed_min_fanout < 2:
            errors.append("condensed_min_fanout must be >= 2")
        if self.leaf_chunk_tokens <= 0:
            errors.append("leaf_chunk_tokens must be > 0")
        if self.max_context_tokens <= 0:
            errors.append("max_context_tokens must be > 0")
        if self.delegation_timeout_ms < 0:
            errors.append("delegation_timeout_ms must be >= 0")
        if self.new_session_retain_depth < 0:
            errors.append("new_session_retain_depth must be >= 0")
        return errors

    @property
    def effective_bootstrap_max_tokens(self) -> int:
        """Return bootstrap_max_tokens floored at max(6000, leaf_chunk_tokens * 0.3)."""
        return max(self.bootstrap_max_tokens, int(self.leaf_chunk_tokens * 0.3))

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def resolved_db_path(self) -> str:
        """Return *db_path* with ``~`` expanded to the user home directory."""
        return str(Path(self.db_path).expanduser())
