"""Agent adapters for lossless memory integration."""

from .base import AgentAdapter, LCMConfig
from .hermes import HermesAdapter
from .openclaw import OpenClawAdapter
from .generic import GenericAdapter
from .simple import SimpleAdapter
from .factory import create_adapter

__all__ = [
    "AgentAdapter",
    "LCMConfig",
    "HermesAdapter",
    "OpenClawAdapter",
    "GenericAdapter",
    "SimpleAdapter",
    "create_adapter",
]
