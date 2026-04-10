# LongMemEval Benchmark Report — lossless-agent

Generated: 2026-04-09 11:42:11
Model: claude-haiku-4-5 | Judge: claude-haiku-4-5
Dataset: longmemeval_s (500 questions, 6 types)

## Overall Results

| Method | Overall | Task-Avg |
|--------|---------|----------|
| LCM (lossless-agent) | 38.4% (192/500) | 41.4% |
| Baseline (full context) | 44.4% (222/500) | 46.6% |

## By Question Type

| Type | LCM | Baseline | Delta |
|------|-----|----------|-------|
| knowledge-update | 57.7% | 73.1% | -15.4% |
| multi-session | 29.3% | 36.1% | -6.8% |
| single-session-assistant | 76.8% | 85.7% | -8.9% |
| single-session-preference | 10.0% | 6.7% | +3.3% |
| single-session-user | 58.6% | 58.6% | +0.0% |
| temporal-reasoning | 15.8% | 19.5% | -3.8% |

## Cost & Performance

| Metric | LCM | Baseline |
|--------|-----|----------|
| LLM Calls | 480 | 500 |
| Input Tokens | 56,211,298 | 59,202,256 |
| Output Tokens | 28,728 | 28,914 |
| Est. Cost | $14.09 | $14.84 |

## Comparison with Published Results

| System | Task-Avg |
|--------|----------|
| agentmemory | 96.2% |
| CortiLoop | 92.0% |
| Atlas Memory | 90.18% |
| Zep | 71.0% |
| Mem0 | 49.0% |
| **lossless-agent (LCM)** | **41.4%** |
| Baseline (no memory system) | 46.6% |

## Notes

- LCM strategy: ingest all sessions → DAG compaction → hybrid retrieval (FTS5 + compacted summaries)
- Baseline strategy: concatenate all raw sessions into context window
- Context budget: 120000 tokens (LCM), 120000 tokens (baseline)
- Self-evaluation using claude-haiku-4-5 as judge (official benchmark uses GPT-4o)
