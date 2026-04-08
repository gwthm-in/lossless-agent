"""Startup banner: one-shot log messages emitted at plugin load time."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from lossless_agent.config import LCMConfig

logger = logging.getLogger(__name__)


class StartupBanner:
    """Emit one-shot log messages keyed by a unique banner key."""

    _emitted: Set[str] = set()

    @classmethod
    def emit_once(cls, key: str, message: str) -> bool:
        """Log *message* if *key* has not been emitted yet. Return True if emitted."""
        if key in cls._emitted:
            return False
        cls._emitted.add(key)
        logger.info(message)
        return True

    @classmethod
    def reset(cls) -> None:
        """Clear emitted set (for tests)."""
        cls._emitted.clear()

    @classmethod
    def log_plugin_loaded(cls, config: "LCMConfig") -> None:
        """Log plugin loaded banner."""
        cls.emit_once(
            "plugin_loaded",
            f"[lcm] Plugin loaded (enabled={config.enabled}, "
            f"db={config.db_path}, threshold={config.context_threshold})",
        )

    @classmethod
    def log_compaction_model(cls, config: "LCMConfig") -> None:
        """Log compaction model info."""
        model = config.summary_model or "(default)"
        provider = config.summary_provider or "(default)"
        cls.emit_once(
            "compaction_model",
            f"[lcm] Compaction model: {model} via {provider}",
        )

    @classmethod
    def log_session_patterns(cls, config: "LCMConfig") -> None:
        """Log ignore + stateless patterns if any."""
        parts = []
        if config.ignore_session_patterns:
            parts.append(f"ignore={config.ignore_session_patterns}")
        if config.stateless_session_patterns:
            parts.append(f"stateless={config.stateless_session_patterns}")
        if parts:
            cls.emit_once(
                "session_patterns",
                f"[lcm] Session patterns: {', '.join(parts)}",
            )
