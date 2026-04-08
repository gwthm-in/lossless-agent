"""Context engine for lossless agent."""

from .assembler import AssemblerConfig, AssembledContext, ContextAssembler
from .compaction import (
    CompactionConfig,
    CompactionEngine,
    LcmProviderAuthError,
    SummarizerTimeoutError,
    summarize_with_escalation,
)
from .circuit_breaker import CircuitBreaker
from .transcript_repair import TranscriptRepairer

__all__ = [
    "AssemblerConfig",
    "AssembledContext",
    "ContextAssembler",
    "CompactionConfig",
    "CompactionEngine",
    "CircuitBreaker",
    "LcmProviderAuthError",
    "SummarizerTimeoutError",
    "TranscriptRepairer",
    "summarize_with_escalation",
]
