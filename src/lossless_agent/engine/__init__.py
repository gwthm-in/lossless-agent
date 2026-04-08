"""Context engine for lossless agent."""

from .assembler import AssemblerConfig, AssembledContext, ContextAssembler
from .bootstrap import SessionBootstrap, BootstrapResult
from .circuit_breaker import CircuitBreaker
from .compaction import (
    CompactionConfig,
    CompactionEngine,
    LcmProviderAuthError,
    SummarizerTimeoutError,
    summarize_with_escalation,
)
from .expansion_auth import (
    ExpansionAuthManager,
    Grant,
    InvalidGrantError,
    ExhaustedBudgetError,
)
from .expansion_policy import ExpansionPolicy, PolicyAction, PolicyDecision
from .delegation_guard import DelegationGuard
from .fts_safety import FTSSafety
from .heartbeat import HeartbeatPruner
from .integrity import IntegrityChecker, CheckResult
from .large_files import LargeFileInterceptor, LargeFileConfig
from .media import MediaAnnotator
from .session_patterns import SessionPatternMatcher
from .startup_banner import StartupBanner
from .structured_text import StructuredTextExtractor
from .system_prompt import CompactionAwarePrompt
from .transcript_repair import TranscriptRepairer

__all__ = [
    "AssemblerConfig",
    "AssembledContext",
    "BootstrapResult",
    "CheckResult",
    "CircuitBreaker",
    "CompactionAwarePrompt",
    "CompactionConfig",
    "CompactionEngine",
    "ContextAssembler",
    "DelegationGuard",
    "ExhaustedBudgetError",
    "ExpansionAuthManager",
    "ExpansionPolicy",
    "FTSSafety",
    "Grant",
    "HeartbeatPruner",
    "IntegrityChecker",
    "InvalidGrantError",
    "LargeFileConfig",
    "LargeFileInterceptor",
    "LcmProviderAuthError",
    "MediaAnnotator",
    "PolicyAction",
    "PolicyDecision",
    "SessionBootstrap",
    "SessionPatternMatcher",
    "StartupBanner",
    "StructuredTextExtractor",
    "SummarizerTimeoutError",
    "TranscriptRepairer",
    "summarize_with_escalation",
]
