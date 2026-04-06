# Contributing to Lossless Agent

Thank you for your interest in contributing! This guide will help you get started.

## CLA Requirement

Before your PR can be merged, you must sign the Contributor License Agreement (CLA).
See [CLA.md](CLA.md) for details. This enables dual licensing (AGPL v3 + commercial).

## Development Setup

```bash
git clone https://github.com/gwthm-in/lossless-agent.git
cd lossless-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
PYTHONPATH=src python3 -m pytest tests/ --cov=lossless_agent --cov-report=term-missing
```

All new code must have tests. We follow strict TDD: write tests first, then implementation.
Aim for >90% coverage on new code.

## Commit Convention

We follow [Chris Beams' 7 rules](https://cbea.ms/git-commit/) plus conventional commit prefixes:

- `feat:` New feature
- `fix:` Bug fix
- `test:` Adding or updating tests
- `docs:` Documentation changes
- `refactor:` Code restructuring (no behavior change)
- `chore:` Build, CI, tooling changes

Rules:
1. Separate subject from body with a blank line
2. Limit subject to 50 characters
3. Capitalize the subject line
4. Do not end the subject with a period
5. Use imperative mood ("Add feature" not "Added feature")
6. Wrap body at 72 characters
7. Use body to explain what and why, not how

## Branch Naming

- `feature/short-description`
- `fix/short-description`
- `docs/short-description`

## Pull Request Process

1. Fork the repo
2. Create a branch from `main`
3. Write tests first (TDD)
4. Implement the feature/fix
5. Ensure all tests pass
6. Submit PR against `main`
7. Sign the CLA if you haven't already

## Code Style

- **Linter:** ruff (88 char line length)
- **Types:** mypy for type checking
- **Formatter:** ruff format

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Architecture Overview

```
src/lossless_agent/
  store/          # Persistence layer (ABCs + SQLite implementation)
  engine/         # Compaction, assembly, large file handling
  tools/          # Recall tools (grep, describe, expand, expand_query)
  adapters/       # Agent integrations (Hermes, OpenClaw, Generic, Simple)
  config.py       # Configuration with env var overrides
```

### Adding a New Adapter

Inherit from `AgentAdapter` and implement the abstract methods, or use `GenericAdapter` directly.

### Adding a New Store Backend

Implement `AbstractConversationStore`, `AbstractMessageStore`, and `AbstractSummaryStore` from `store/abc.py`.

## Reporting Bugs

Please include:
- Python version
- Operating system
- lossless-agent version
- Steps to reproduce
- Expected vs actual behavior

## Feature Requests

Open an issue with the feature request template. We welcome ideas!
