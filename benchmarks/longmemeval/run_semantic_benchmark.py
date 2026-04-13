#!/usr/bin/env python3
"""LongMemEval benchmark — LCM with semantic retrieval via pgvector.

Compares semantic retrieval (embeddings + vector search) against original
FTS5 + brute-force and the raw baseline.

Strategy:
1. Ingest all sessions → embed each message into pgvector
2. At query time: embed question → vector search for top-K messages
3. Augment with FTS5 hits + session context → answer

Uses local fastembed (mixedbread-ai/mxbai-embed-large-v1, dim=1024) for embeddings.
Uses Postgres+pgvector for vector storage.
Uses claude-haiku-4-5 via litellm proxy for LLM calls.

Usage:
    .venv/bin/python benchmarks/longmemeval/run_semantic_benchmark.py
"""
from __future__ import annotations

# Load .env for API credentials
from pathlib import Path as _P
_env_path = _P(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                import os as _os
                _os.environ.setdefault(_k.strip(), _v.strip())

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tempfile  # noqa: E402
import shutil  # noqa: E402
import traceback  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Tuple  # noqa: E402

# Add parent to path for lossless-agent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lossless_agent.store import Database, ConversationStore, MessageStore  # noqa: E402
from lossless_agent.tools import lcm_grep  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/GowthamSai/Documents/repos/gwthm-in/LongMemEval/data/longmemeval_s"
OUTPUT_DIR = Path(__file__).parent
SEMANTIC_RESULTS = OUTPUT_DIR / "results_semantic_lcm.jsonl"
EVAL_RESULTS_SEMANTIC = OUTPUT_DIR / "results_semantic_eval.jsonl"
REPORT_PATH = OUTPUT_DIR / "BENCHMARK_REPORT_SEMANTIC.md"

# Previously completed results (for comparison)
PREV_EVAL = OUTPUT_DIR / "results_full_eval.jsonl"

MODEL = "claude-haiku-4-5"
PROVIDER = "anthropic"
JUDGE_MODEL = "claude-haiku-4-5"
CONTEXT_BUDGET = 120000  # tokens

# Embedding config (local fastembed)
EMBEDDING_DIM = 1024
EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

# Semantic search config
SEMANTIC_TOP_K = 30  # top messages from vector search
MIN_SIMILARITY = 0.35  # cosine similarity threshold


# ---------------------------------------------------------------------------
# Local Embedder (fastembed, batch-capable)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from fastembed import TextEmbedding
        _EMBED_MODEL = TextEmbedding(EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using local fastembed model."""
    model = _get_embed_model()
    embeddings = list(model.embed(texts, batch_size=64))
    return [e.tolist() for e in embeddings]


def embed_single(text: str) -> List[float]:
    """Embed a single text."""
    return embed_batch([text])[0]


# ---------------------------------------------------------------------------
# In-memory vector search (numpy, no Postgres needed per-question)
# ---------------------------------------------------------------------------

class InMemoryVectorIndex:
    """Simple cosine-similarity search using numpy.
    
    For the benchmark, each question gets its own index (~500 messages).
    This avoids Postgres connection overhead per question.
    """
    
    def __init__(self):
        self._embeddings = []  # list of numpy arrays
        self._metadata = []    # list of dicts (content, session_id, role, etc.)
    
    def add(self, embedding: List[float], metadata: dict):
        self._embeddings.append(np.array(embedding, dtype=np.float32))
        self._metadata.append(metadata)
    
    def search(self, query_embedding: List[float], top_k: int = 10, min_score: float = 0.3) -> List[Tuple[dict, float]]:
        if not self._embeddings:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        
        # Stack all embeddings into matrix
        matrix = np.stack(self._embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_norm = matrix / norms
        
        # Cosine similarities
        scores = matrix_norm @ query_norm
        
        # Get top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            results.append((self._metadata[idx], score))
        
        return results


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------

class LLMBackend:
    def __init__(self, model: str, provider: str = "anthropic"):
        self.model = model
        self.provider = provider
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_cost = 0.0

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        client = self._get_client()
        self.total_calls += 1
        loop = asyncio.get_event_loop()

        resp = await loop.run_in_executor(None, lambda: client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ))
        self.total_input_tokens += resp.usage.input_tokens
        self.total_output_tokens += resp.usage.output_tokens
        self.total_cost += (resp.usage.input_tokens * 0.25 + resp.usage.output_tokens * 1.25) / 1_000_000
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Key term extraction
# ---------------------------------------------------------------------------

def extract_key_terms(question: str) -> List[str]:
    stop_words = {
        "what", "when", "where", "who", "how", "which", "why", "did",
        "does", "was", "were", "is", "are", "the", "a", "an", "my",
        "your", "i", "me", "we", "you", "do", "have", "has", "had",
        "that", "this", "with", "for", "from", "about", "after",
        "before", "first", "last", "most", "recently", "ever",
        "any", "some", "can", "could", "would", "should", "tell",
        "name", "mentioned", "discussed", "talked", "said", "told",
    }
    words = question.lower().replace("?", "").replace(",", "").replace(".", "").split()
    terms = [w for w in words if w not in stop_words and len(w) > 2]
    bigrams = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms) - 1)]
    return bigrams[:3] + terms[:5]


# ---------------------------------------------------------------------------
# Strategy: Semantic Retrieval
# ---------------------------------------------------------------------------

async def run_semantic_question(question: dict, llm: LLMBackend, idx: int, total: int) -> dict:
    qid = question["question_id"]
    qtype = question["question_type"]
    qtxt = question["question"]
    sessions = question["haystack_sessions"]
    session_ids = question["haystack_session_ids"]
    session_dates = question.get("haystack_dates", [])

    print(f"\n[SEM {idx+1}/{total}] {qid} ({qtype})")
    print(f"  Q: {qtxt[:100]}")

    t0_total = time.time()

    # PHASE 1: Ingest + embed all messages
    t0 = time.time()
    all_messages = []  # (content, session_id, role, msg_idx)
    all_texts = []
    
    for session, sid in zip(sessions, session_ids):
        for msg_idx, turn in enumerate(session):
            content = turn["content"]
            if content.strip():
                all_messages.append({
                    "content": content,
                    "session_id": sid,
                    "role": turn["role"],
                    "msg_idx": msg_idx,
                })
                all_texts.append(content[:512])  # truncate long messages for embedding
    
    # Batch embed all messages
    embeddings = embed_batch(all_texts)
    
    # Build in-memory vector index
    index = InMemoryVectorIndex()
    for emb, meta in zip(embeddings, all_messages):
        index.add(emb, meta)
    
    time_embed = time.time() - t0

    # Also set up SQLite for FTS5
    tmp_dir = tempfile.mkdtemp(prefix=f"sem_{qid}_")
    db_path = os.path.join(tmp_dir, "lcm.db")
    db = Database(db_path)
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    
    for session, sid in zip(sessions, session_ids):
        conv = conv_store.get_or_create(f"session_{sid}")
        for turn in session:
            msg_store.append(conv.id, turn["role"], turn["content"],
                           len(turn["content"].split()))

    # PHASE 2: Semantic + FTS5 hybrid retrieval
    t0 = time.time()
    context_parts = []
    tokens_used = 0
    seen_content = set()

    # 2a: Vector search — embed question, find similar messages
    query_emb = embed_single(qtxt)
    semantic_hits = index.search(query_emb, top_k=SEMANTIC_TOP_K, min_score=MIN_SIMILARITY)
    
    semantic_count = 0
    for meta, score in semantic_hits:
        content = meta["content"]
        content_key = content[:200]  # dedup key
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        
        session_label = f"Session {meta['session_id']}"
        role = meta["role"]
        context_parts.append(f"[SEMANTIC score={score:.3f} {session_label}] [{role}]: {content}")
        tokens_used += len(content.split())
        semantic_count += 1
        
        if tokens_used >= CONTEXT_BUDGET // 2:
            break

    # 2b: FTS5 keyword search
    fts_results = lcm_grep(db, qtxt, scope="all", limit=20)
    key_terms = extract_key_terms(qtxt)
    for term in key_terms[:3]:
        try:
            term_results = lcm_grep(db, term, scope="all", limit=10)
            fts_results.extend(term_results)
        except Exception:
            pass

    fts_count = 0
    for r in fts_results:
        content = r.content_snippet if hasattr(r, 'content_snippet') else r.content
        if not content:
            continue
        content_key = content[:200]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        context_parts.append(f"[FTS HIT] {content}")
        tokens_used += len(content.split())
        fts_count += 1

    # 2c: Fill remaining budget with full session context (round-robin)
    for sid, session in zip(session_ids, sessions):
        if tokens_used >= CONTEXT_BUDGET:
            break
        session_text = f"--- Session {sid} ---\n"
        for turn in session:
            session_text += f"[{turn['role']}]: {turn['content']}\n"
        stokens = len(session_text.split())
        if tokens_used + stokens <= CONTEXT_BUDGET:
            context_parts.append(session_text)
            tokens_used += stokens

    retrieved_context = "\n\n".join(context_parts)
    time_retrieve = time.time() - t0

    # PHASE 3: Answer
    t0 = time.time()
    date_info = ""
    if session_dates:
        date_info = "\nConversation timeline:\n"
        for sid, date in zip(session_ids, session_dates):
            date_info += f"  Session {sid}: {date}\n"

    answer_prompt = f"""You are a personal AI assistant answering a question about past conversations with your user.
The retrieved context below contains relevant excerpts from your conversation history.
The most relevant excerpts are labeled with [SEMANTIC] or [FTS HIT] scores.
Answer based ONLY on what you find in the context. Be precise and specific.
If the context contains the answer, state it directly. If the question asks about something
that was updated/changed, give the MOST RECENT value.
Only say you don't know if the context is completely irrelevant.

Question date: {question.get('question_date', 'unknown')}
{date_info}

Retrieved conversation context:
{retrieved_context}

Question: {qtxt}

Answer concisely and directly in 1-2 sentences. State the specific answer first."""

    hypothesis = await llm.complete(answer_prompt, max_tokens=256)
    time_answer = time.time() - t0

    time_total = time.time() - t0_total

    print(f"  A: {hypothesis[:100]}")
    print(f"  semantic={semantic_count} fts={fts_count} msgs={len(all_messages)} embed={time_embed:.1f}s retrieve={time_retrieve:.1f}s answer={time_answer:.1f}s total={time_total:.1f}s")

    # Cleanup
    db.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "question_id": qid,
        "hypothesis": hypothesis,
        "_question_type": qtype,
        "_semantic_hits": semantic_count,
        "_fts_hits": fts_count,
        "_sessions": len(sessions),
        "_messages": len(all_messages),
        "_tokens_used": tokens_used,
        "_time_embed": time_embed,
        "_time_retrieve": time_retrieve,
        "_time_answer": time_answer,
        "_time_total": time_total,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate(predictions: List[dict], ref_data: List[dict], llm: LLMBackend, label: str) -> dict:
    ref_by_id = {q["question_id"]: q for q in ref_data}

    correct = 0
    total = 0
    by_type = {}
    eval_details = []

    for pred in predictions:
        qid = pred["question_id"]
        ref = ref_by_id.get(qid)
        if not ref:
            continue

        qtype = ref["question_type"]
        is_abstention = "_abs" in qid

        if is_abstention:
            judge_prompt = f"""The user asked: "{ref['question']}"
The correct behavior is to say the information is not available/unknown.
The model responded: "{pred['hypothesis']}"

Did the model correctly identify that it cannot answer? Answer ONLY "yes" or "no"."""
        else:
            judge_prompt = f"""Question: "{ref['question']}"
Correct answer: "{ref['answer']}"
Model response: "{pred['hypothesis']}"

Does the model response contain the correct answer? Be lenient — if the key fact is present even with extra text, count it. Answer ONLY "yes" or "no"."""

        verdict = await llm.complete(judge_prompt, max_tokens=10)
        is_correct = "yes" in verdict.lower()

        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if is_correct:
            by_type[qtype]["correct"] += 1
            correct += 1
        total += 1

        eval_details.append({
            "question_id": qid,
            "question_type": qtype,
            "correct": is_correct,
            "hypothesis": pred["hypothesis"],
            "answer": ref["answer"],
        })

        _status = "Y" if is_correct else "N"
        if total % 50 == 0:
            print(f"  Evaluated {total}/{len(predictions)}... ({correct}/{total} correct so far)")

    overall = correct / total * 100 if total else 0
    type_accs = {}
    for qtype, counts in by_type.items():
        type_accs[qtype] = counts["correct"] / counts["total"] * 100 if counts["total"] else 0

    task_avg = sum(type_accs.values()) / len(type_accs) if type_accs else 0

    return {
        "label": label,
        "overall": overall,
        "correct": correct,
        "total": total,
        "task_average": task_avg,
        "by_type": {k: {"accuracy": v, **by_type[k]} for k, v in type_accs.items()},
        "details": eval_details,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(sem_eval: dict, sem_llm: LLMBackend, prev_eval: dict = None, timing: dict = None):
    lines = []
    lines.append("# LongMemEval Benchmark Report — Semantic Retrieval Layer")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {MODEL} | Judge: {JUDGE_MODEL}")
    lines.append(f"Embeddings: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM}, local)")
    lines.append("Dataset: longmemeval_s (500 questions, 6 types)")
    lines.append("")

    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Method | Overall | Task-Avg |")
    lines.append("|--------|---------|----------|")
    lines.append(f"| **LCM+Semantic** | **{sem_eval['overall']:.1f}%** ({sem_eval['correct']}/{sem_eval['total']}) | **{sem_eval['task_average']:.1f}%** |")
    
    if prev_eval:
        lcm = prev_eval.get("lcm", {})
        base = prev_eval.get("baseline", {})
        if lcm:
            lines.append(f"| LCM (original, FTS5 only) | {lcm.get('overall', 0):.1f}% ({lcm.get('correct', 0)}/{lcm.get('total', 0)}) | {lcm.get('task_average', 0):.1f}% |")
        if base:
            lines.append(f"| Baseline (full context) | {base.get('overall', 0):.1f}% ({base.get('correct', 0)}/{base.get('total', 0)}) | {base.get('task_average', 0):.1f}% |")
    lines.append("")

    lines.append("## By Question Type")
    lines.append("")
    
    header = "| Type | LCM+Semantic |"
    sep = "|------|-------------|"
    if prev_eval:
        lcm = prev_eval.get("lcm", {})
        base = prev_eval.get("baseline", {})
        if lcm:
            header += " LCM (orig) | Delta |"
            sep += "------------|-------|"
        if base:
            header += " Baseline | Delta |"
            sep += "----------|-------|"
    lines.append(header)
    lines.append(sep)

    all_types = sorted(sem_eval["by_type"].keys())
    for t in all_types:
        sem_acc = sem_eval["by_type"].get(t, {}).get("accuracy", 0)
        row = f"| {t} | {sem_acc:.1f}% |"
        if prev_eval:
            lcm = prev_eval.get("lcm", {})
            base = prev_eval.get("baseline", {})
            if lcm:
                orig_acc = lcm.get("by_type", {}).get(t, {}).get("accuracy", 0)
                delta = sem_acc - orig_acc
                sign = "+" if delta >= 0 else ""
                row += f" {orig_acc:.1f}% | {sign}{delta:.1f}% |"
            if base:
                base_acc = base.get("by_type", {}).get(t, {}).get("accuracy", 0)
                delta = sem_acc - base_acc
                sign = "+" if delta >= 0 else ""
                row += f" {base_acc:.1f}% | {sign}{delta:.1f}% |"
        lines.append(row)
    lines.append("")

    lines.append("## Cost & Performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| LLM Calls | {sem_llm.total_calls:,} |")
    lines.append(f"| Input Tokens | {sem_llm.total_input_tokens:,} |")
    lines.append(f"| Output Tokens | {sem_llm.total_output_tokens:,} |")
    lines.append(f"| Est. Cost | ${sem_llm.total_cost:.2f} |")
    if timing:
        lines.append(f"| Total Time | {timing.get('total', 0)/60:.1f} min |")
        lines.append(f"| Avg per question | {timing.get('avg_per_q', 0):.1f}s |")
    lines.append("")

    lines.append("## Comparison with Published Results")
    lines.append("")
    lines.append("| System | Task-Avg |")
    lines.append("|--------|----------|")
    lines.append("| agentmemory | 96.2% |")
    lines.append("| CortiLoop | 92.0% |")
    lines.append("| Atlas Memory | 90.18% |")
    lines.append("| Zep | 71.0% |")
    lines.append("| Mem0 | 49.0% |")
    lines.append(f"| **lossless-agent (LCM+Semantic)** | **{sem_eval['task_average']:.1f}%** |")
    if prev_eval and prev_eval.get("lcm"):
        lines.append(f"| lossless-agent (LCM original) | {prev_eval['lcm'].get('task_average', 0):.1f}% |")
    if prev_eval and prev_eval.get("baseline"):
        lines.append(f"| Baseline (no memory system) | {prev_eval['baseline'].get('task_average', 0):.1f}% |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- LCM+Semantic: embed all messages → vector similarity search + FTS5 keyword search + full session context → answer")
    lines.append(f"- Embedding: {EMBEDDING_MODEL_NAME} (dim={EMBEDDING_DIM}, local via fastembed, no API cost)")
    lines.append(f"- Vector search: in-memory numpy cosine similarity (top-{SEMANTIC_TOP_K}, min_score={MIN_SIMILARITY})")
    lines.append(f"- Context budget: {CONTEXT_BUDGET} tokens")
    lines.append("- Self-evaluation using claude-haiku-4-5 as judge")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to {REPORT_PATH}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing_results(path: Path) -> List[dict]:
    results = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


async def main():
    print("=" * 60)
    print("LongMemEval Benchmark — LCM + Semantic Retrieval")
    print("=" * 60)

    # Warm up the embedding model
    print("\nLoading local embedding model...")
    t0 = time.time()
    _get_embed_model()
    print(f"Embedding model loaded in {time.time() - t0:.1f}s")

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions")

    # ---- RUN SEMANTIC ----
    print("\n" + "=" * 60)
    print("PHASE 1: Semantic Retrieval Strategy")
    print("=" * 60)

    sem_llm = LLMBackend(MODEL, PROVIDER)
    sem_results = load_existing_results(SEMANTIC_RESULTS)
    existing_ids = {r["question_id"] for r in sem_results}
    print(f"Existing results: {len(sem_results)} (skipping)")

    t_start = time.time()
    errors = 0
    for i, question in enumerate(data):
        if question["question_id"] in existing_ids:
            continue
        try:
            result = await run_semantic_question(question, sem_llm, i, len(data))
            sem_results.append(result)
            with open(SEMANTIC_RESULTS, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}")
            traceback.print_exc()
            if errors > 20:
                print("Too many errors, aborting.")
                break
        
        # Progress report every 50 questions
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            done = len(sem_results) - len(existing_ids)
            if done > 0:
                per_q = elapsed / done
                remaining = (len(data) - len(sem_results)) * per_q
                print(f"\n  === Progress: {len(sem_results)}/{len(data)}, {per_q:.1f}s/q, ETA: {remaining/60:.0f}min ===\n")

    t_phase1 = time.time() - t_start
    print(f"\nSemantic: {len(sem_results)} results, {sem_llm.total_calls} LLM calls, ${sem_llm.total_cost:.2f}, {t_phase1/60:.1f}min")

    # ---- EVALUATE ----
    print("\n" + "=" * 60)
    print("PHASE 2: Evaluation")
    print("=" * 60)

    judge_llm = LLMBackend(JUDGE_MODEL, PROVIDER)

    print("\nEvaluating semantic results...")
    t_eval_start = time.time()
    sem_eval = await evaluate(sem_results, data, judge_llm, "semantic")
    t_eval = time.time() - t_eval_start

    # Save eval
    with open(EVAL_RESULTS_SEMANTIC, "w") as f:
        json.dump({"semantic": sem_eval}, f, indent=2)

    # Load previous results for comparison
    prev_eval = None
    if PREV_EVAL.exists():
        with open(PREV_EVAL) as f:
            prev_eval = json.load(f)

    # ---- REPORT ----
    print("\n" + "=" * 60)
    print("PHASE 3: Report")
    print("=" * 60)

    timing = {
        "total": t_phase1 + t_eval,
        "avg_per_q": t_phase1 / max(len(sem_results), 1),
    }
    report = write_report(sem_eval, sem_llm, prev_eval, timing)
    print(report)

    print(f"\nTotal time: {(t_phase1 + t_eval)/60:.1f} min")
    print(f"Total cost: ${sem_llm.total_cost + judge_llm.total_cost:.2f}")
    print("\nDONE!")


if __name__ == "__main__":
    asyncio.run(main())
