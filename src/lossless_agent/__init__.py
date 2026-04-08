"""Lossless context management for AI agents."""

__version__ = "0.1.0"

# --- Adapters ---
from lossless_agent.adapters.simple import SimpleAdapter
from lossless_agent.adapters.generic import GenericAdapter
from lossless_agent.adapters.hermes import HermesAdapter
from lossless_agent.adapters.openclaw import OpenClawAdapter
from lossless_agent.adapters.base_impl import BaseAdapter
from lossless_agent.adapters.factory import create_adapter

# --- Configuration ---
from lossless_agent.config import LCMConfig

# --- Storage ---
from lossless_agent.store import Database

# --- Tool result types ---
from lossless_agent.tools import GrepResult, DescribeResult, ExpandResult

__all__ = [
    # Adapters
    "SimpleAdapter",
    "GenericAdapter",
    "HermesAdapter",
    "OpenClawAdapter",
    "BaseAdapter",
    "create_adapter",
    # Config
    "LCMConfig",
    # Store
    "Database",
    # Tool result types
    "GrepResult",
    "DescribeResult",
    "ExpandResult",
]
