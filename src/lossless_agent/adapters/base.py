"""Abstract base class for agent adapters and shared configuration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, List, Optional

from lossless_agent.engine.compaction import CompactionConfig
from lossless_agent.engine.assembler import AssemblerConfig


@dataclass
class LCMConfig:
    """Combined configuration for the LCM system."""

    db_path: str = ":memory:"
    summary_model: str = "default"
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    assembler: AssemblerConfig = field(
        default_factory=lambda: AssemblerConfig(max_context_tokens=128_000)
    )


class AgentAdapter(ABC):
    """Interface that agents use to interact with lossless memory.

    Adapters wrap stores, engines, and tools behind a turn-oriented API
    so agents never touch internals directly.
    """

    @abstractmethod
    async def on_turn_start(
        self, session_key: str, user_message: str
    ) -> Optional[str]:
        """Called before the LLM call.

        Returns context to inject (retrieved summaries), or None.
        """

    @abstractmethod
    async def on_turn_end(
        self, session_key: str, messages: List[dict]
    ) -> None:
        """Called after the LLM response.

        Persists new messages and triggers compaction when needed.
        """

    @abstractmethod
    async def on_session_end(self, session_key: str) -> None:
        """Final compaction pass when a session ends."""

    @abstractmethod
    def get_tools(self) -> List[dict]:
        """Return tool definitions in OpenAI function-calling schema format.

        Covers lcm_grep, lcm_describe, and lcm_expand.
        """

    @abstractmethod
    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        """Execute a recall tool call and return the JSON result."""

    @abstractmethod
    def get_system_prompt_block(self) -> str:
        """Return the Lossless Recall Policy text to inject into the system prompt."""
