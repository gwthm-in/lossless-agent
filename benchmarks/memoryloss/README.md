# MemoryLoss Benchmark Results

## Overview

[MemoryLoss](https://github.com/gwthm-in/MemoryLoss) is a benchmark for evaluating how much information AI memory systems lose when compressing or summarizing conversation history. It measures **fidelity** — the percentage of questions about the original conversation that can still be answered correctly after the memory system processes it.

The benchmark tests 64 multi-turn conversations across 5 dimensions:

- **Verbatim** — Exact facts, numbers, names, dates
- **Contextual** — Understanding of why things were discussed, motivations
- **Detail** — Specific details that are easy to overlook
- **Temporal** — Order of events and what happened when
- **Synthesis** — Combining multiple pieces of information to answer

## Results

**DAG (lossless-agent) achieved 94.1% fidelity, beating the Oracle baseline (full context) at 91.9%.**

The negative information loss (-2.5%) means DAG's structured representation actually helps the LLM answer questions *more accurately* than having the raw conversation, likely because the DAG format makes information easier to locate and reason over.

| System | Fidelity | Info Loss | Verbatim | Context | Detail | Temporal | Synthesis |
|---|---|---|---|---|---|---|---|
| **DAG (lossless-agent)** | **94.1%** | **-2.5%** | **100%** | **97.3%** | **97.9%** | **83.3%** | **88.1%** |
| Oracle (full context) | 91.9% | 0.0% | 96.6% | 97.3% | 95.8% | 80.6% | 85.7% |
| Vector Search (ChromaDB) | 86.9% | 5.4% | 98.3% | 91.9% | 95.8% | 52.8% | 85.7% |
| LLM Summary | 86.9% | 5.4% | 94.9% | 78.4% | 95.8% | 75.0% | 83.3% |
| AgentMemory-style (KG+Hybrid) | 82.4% | 10.3% | 91.5% | 83.8% | 95.8% | 52.8% | 78.6% |
| Atomic Extraction | 73.4% | 20.1% | 100% | 59.5% | 91.7% | 27.8% | 66.7% |
| Truncation | 50.9% | 44.6% | 71.2% | 67.6% | 47.9% | 33.3% | 26.2% |

## Key Takeaways

1. **DAG beats Oracle** — The structured DAG format doesn't just preserve information; it makes it more accessible. The LLM can answer questions more accurately from the DAG than from the raw conversation.

2. **100% Verbatim recall** — Every exact fact, number, and date is preserved perfectly.

3. **Temporal reasoning is the hardest dimension for all systems** — Even Oracle only scores 80.6% on temporal questions. DAG's 83.3% is the best across all systems.

4. **Vector search and summaries lose ~5% of information** — Mostly in contextual understanding and temporal reasoning.

5. **Traditional memory systems (KG+Hybrid) lose 10%+** — The atomic extraction approach loses 20%, mostly in contextual and temporal dimensions.

## Files

- `result_dag.json` — Full per-case results for the DAG system (64 cases, 222 questions)

## Reproducing

See the [MemoryLoss repository](https://github.com/gwthm-in/MemoryLoss) for the full benchmark suite, dataset, and evaluation scripts.
