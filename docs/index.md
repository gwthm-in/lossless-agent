# lossless-agent

**Lossless context management for AI agents.**

lossless-agent gives your AI agent perfect long-term memory. Instead of
truncating old messages, it compacts them into a DAG of summaries that
preserves every detail and lets the agent drill back in on demand.

## Why lossless-agent?

- **No information loss** — messages are summarised, never deleted. The
  original text is always reachable via `lcm_expand`.
- **Budget-aware assembly** — the context assembler fits summaries and
  recent messages into whatever token budget you have.
- **Agent-agnostic** — works with any LLM framework through adapters
  (SimpleAdapter, GenericAdapter, HermesAdapter) or bring your own.
- **Recall tools** — `lcm_grep`, `lcm_describe`, and `lcm_expand` let
  the agent search and navigate its own memory.
- **Cross-session retrieval** — optionally embed summaries with pgvector
  and pull relevant context from other sessions via semantic search.

## Quick links

- [Quick Start](quickstart.md) — get running in five lines of code
- [Architecture](architecture.md) — how the DAG, engine, and stores fit together
- [Configuration](configuration.md) — every tunable with its env var and default
- [Adapters](adapters.md) — choose the right adapter for your use case
- [API Reference](api.md) — class and function signatures
- [Semantic Retrieval](semantic-retrieval.md) — cross-session context via pgvector

## Installation

```bash
pip install lossless-agent

# With Postgres/pgvector support for cross-session retrieval
pip install lossless-agent[postgres]
```

For development:

```bash
git clone https://github.com/gwthm-in/lossless-agent.git
cd lossless-agent
pip install -e '.[dev]'
```

## License

AGPL-3.0-or-later
