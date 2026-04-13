# LongMemEval Benchmark Report — Semantic Retrieval Layer

Generated: 2026-04-11 03:38:35
Model: claude-haiku-4-5 | Judge: claude-haiku-4-5
Embeddings: BAAI/bge-small-en-v1.5 (dim=384, local)
Dataset: longmemeval_s (500 questions, 6 types)

## Overall Results

| Method | Overall | Task-Avg |
|--------|---------|----------|
| **LCM+Semantic** | **53.8%** (264/491) | **56.2%** |
| LCM (original, FTS5 only) | 38.4% (192/500) | 41.4% |
| Baseline (full context) | 44.4% (222/500) | 46.6% |

## By Question Type

| Type | LCM+Semantic | LCM (orig) | Delta | Baseline | Delta |
|------|-------------|------------|-------|----------|-------|
| knowledge-update | 75.6% | 57.7% | +17.9% | 73.1% | +2.6% |
| multi-session | 45.1% | 29.3% | +15.8% | 36.1% | +9.0% |
| single-session-assistant | 89.3% | 76.8% | +12.5% | 85.7% | +3.6% |
| single-session-preference | 19.0% | 10.0% | +9.0% | 6.7% | +12.4% |
| single-session-user | 84.3% | 58.6% | +25.7% | 58.6% | +25.7% |
| temporal-reasoning | 24.1% | 15.8% | +8.3% | 19.5% | +4.5% |

## Cost & Performance

| Metric | Value |
|--------|-------|
| LLM Calls | 500 |
| Input Tokens | 61,417,806 |
| Output Tokens | 29,271 |
| Est. Cost | $15.39 |
| Total Time | 497.1 min |
| Avg per question | 59.8s |

## Comparison with Published Results

| System | Task-Avg |
|--------|----------|
| agentmemory | 96.2% |
| CortiLoop | 92.0% |
| Atlas Memory | 90.18% |
| Zep | 71.0% |
| Mem0 | 49.0% |
| **lossless-agent (LCM+Semantic)** | **56.2%** |
| lossless-agent (LCM original) | 41.4% |
| Baseline (no memory system) | 46.6% |

## Notes

- LCM+Semantic: embed all messages → vector similarity search + FTS5 keyword search + full session context → answer
- Embedding: BAAI/bge-small-en-v1.5 (dim=384, local via fastembed, no API cost)
- Vector search: in-memory numpy cosine similarity (top-30, min_score=0.35)
- Context budget: 120000 tokens
- Self-evaluation using claude-haiku-4-5 as judge
