"""Canonical configuration for the LCM system.

LCMConfig consolidates all tunables with a clear precedence hierarchy:
    env vars  >  explicit config  >  defaults
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import List

from lossless_agent.engine.compaction import CompactionConfig
from lossless_agent.engine.assembler import AssemblerConfig


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.strip().lower() in ("true", "1", "yes")


@dataclass
class LCMConfig:
    """Combined configuration for the LCM system.

    Every field has a corresponding ``LCM_*`` environment variable that
    takes precedence over values supplied at construction time.
    """

    enabled: bool = True
    db_path: str = "~/.lossless-agent/lcm.db"
    fresh_tail_count: int = 8
    leaf_chunk_tokens: int = 20_000
    leaf_min_fanout: int = 4
    condensed_min_fanout: int = 3
    context_threshold: float = 0.75
    leaf_target_tokens: int = 1200
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
    circuit_breaker_cooldown_ms: int = 30_000

    # ------------------------------------------------------------------
    # Env-var mapping
    # ------------------------------------------------------------------

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
            if field_name == "ignore_session_patterns":
                kwargs[field_name] = [
                    p.strip() for p in raw.split(",") if p.strip()
                ]
            else:
                kwargs[field_name] = converter(raw)  # type: ignore[operator]
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
        )

    def to_assembler_config(self) -> AssemblerConfig:
        """Map flat LCMConfig fields to the engine's AssemblerConfig."""
        return AssemblerConfig(
            max_context_tokens=self.max_context_tokens,
            summary_budget_ratio=self.summary_budget_ratio,
            fresh_tail_count=self.fresh_tail_count,
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
        return errors

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def resolved_db_path(self) -> str:
        """Return *db_path* with ``~`` expanded to the user home directory."""
        return str(Path(self.db_path).expanduser())
