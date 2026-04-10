# Lossless Context Benchmark (LCB) Report

Generated: 2026-04-10 01:01:32
Model: claude-haiku-4-5 | Judge: claude-haiku-4-5
Conversations tested: 20

## Overall Results

| Metric | lossless-agent (compressed) | lossless-agent (with expand) | agentmemory |
|--------|---------------------------|------------------------------|-------------|
| Question Accuracy | 85/100 (85.0%) | 88/100 (88.0%) | 18/100 (18.0%) |
| Verbatim Recovery | 46/46 (100.0%) | N/A (same) | 30/46 (65.2%) |
| Avg Compression Ratio | 100.3% | — | 23.6% |

## By Question Type

| Type | lossless-agent (compressed) | lossless-agent (expanded) | agentmemory | Winner |
|------|---------------------------|--------------------------|-------------|--------|
| contextual | 15/20 (75%) | 17/20 (85%) | 4/20 (20%) | lossless-agent |
| detail | 19/20 (95%) | 18/20 (90%) | 2/20 (10%) | lossless-agent |
| synthesis | 16/20 (80%) | 17/20 (85%) | 7/20 (35%) | lossless-agent |
| temporal | 15/20 (75%) | 16/20 (80%) | 3/20 (15%) | lossless-agent |
| verbatim | 20/20 (100%) | 20/20 (100%) | 2/20 (10%) | lossless-agent |

## Compression Efficiency

| Metric | lossless-agent | agentmemory |
|--------|---------------|-------------|
| Avg compression ratio | 100.3% | 23.6% |
| Can expand back to original | YES | NO |
| Preserves conversation flow | YES | NO (atomic facts only) |

## Key Finding

**lossless-agent's expand capability** is the differentiator. When questions require
contextual understanding (why something was said, what led to a topic, temporal flow),
expansion from the DAG recovers information that atomic memory extraction loses.

## Token Usage

| Metric | Value |
|--------|-------|
| Total input tokens | 815,229 |
| Total output tokens | 24,824 |
| Est. cost | $0.23 |
