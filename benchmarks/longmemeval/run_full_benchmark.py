#!/usr/bin/env python3
"""Full LongMemEval benchmark — autonomous, no human approval needed.

Two strategies compared:
1. LCM: Ingest → Compact → Retrieve via DAG → Answer
2. Baseline: Stuff all sessions into context → Answer

The key fix from the 30% accuracy run: instead of relying on FTS5 keyword
search (which misses semantic matches), we use a hybrid approach:
- FTS5 search for direct hits
- ALSO retrieve full context from ALL sessions (compacted summaries fit in budget)
- The DAG summaries preserve key facts even after compaction

Usage:
    python run_full_benchmark.py
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

import asyncio
import json
import os
import sys
import time
import tempfile
import shutil
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Add parent to path for lossless-agent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lossless_agent.adapters.simple import SimpleAdapter
from lossless_agent.config import LCMConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/GowthamSai/Documents/repos/gwthm-in/LongMemEval/data/longmemeval_s"
OUTPUT_DIR = Path(__file__).parent
LCM_RESULTS = OUTPUT_DIR / "results_full_lcm.jsonl"
BASELINE_RESULTS = OUTPUT_DIR / "results_full_baseline.jsonl"
EVAL_RESULTS = OUTPUT_DIR / "results_full_eval.jsonl"
REPORT_PATH = OUTPUT_DIR / "BENCHMARK_REPORT.md"

MODEL = "claude-haiku-4-5"
PROVIDER = "anthropic"
JUDGE_MODEL = "claude-haiku-4-5"
CONTEXT_BUDGET = 120000  # tokens — use most of haiku's 200K window
MAX_BASELINE_TOKENS = 120000


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
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
        return self._client

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        client = self._get_client()
        self.total_calls += 1
        loop = asyncio.get_event_loop()

        if self.provider == "anthropic":
            resp = await loop.run_in_executor(None, lambda: client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ))
            self.total_input_tokens += resp.usage.input_tokens
            self.total_output_tokens += resp.usage.output_tokens
            # Haiku pricing: $0.25/M input, $1.25/M output
            self.total_cost += (resp.usage.input_tokens * 0.25 + resp.usage.output_tokens * 1.25) / 1_000_000
            return resp.content[0].text
        elif self.provider == "openai":
            resp = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ))
            if resp.usage:
                self.total_input_tokens += resp.usage.prompt_tokens
                self.total_output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content


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
# Strategy 1: LCM (ingest → compact → hybrid retrieve → answer)
# ---------------------------------------------------------------------------

async def run_lcm_question(question: dict, llm: LLMBackend, idx: int, total: int) -> dict:
    qid = question["question_id"]
    qtype = question["question_type"]
    qtxt = question["question"]
    answer = question["answer"]
    sessions = question["haystack_sessions"]
    session_ids = question["haystack_session_ids"]
    session_dates = question.get("haystack_dates", [])

    print(f"\n[LCM {idx+1}/{total}] {qid} ({qtype})")
    print(f"  Q: {qtxt[:100]}")

    tmp_dir = tempfile.mkdtemp(prefix=f"lcm_{qid}_")
    db_path = os.path.join(tmp_dir, "lcm.db")

    async def summarize_fn(prompt: str) -> str:
        return await llm.complete(prompt, max_tokens=512)

    adapter = SimpleAdapter(db_path, summarize_fn)

    try:
        # PHASE 1: Ingest
        t0 = time.time()
        for session, sid in zip(sessions, session_ids):
            messages = [{"role": t["role"], "content": t["content"],
                        "token_count": len(t["content"].split())} for t in session]
            await adapter.ingest(f"session_{sid}", messages)
        time_ingest = time.time() - t0

        # PHASE 2: Compact
        t0 = time.time()
        total_summaries = 0
        for sid in session_ids:
            n = await adapter.compact(f"session_{sid}")
            total_summaries += n
        time_compact = time.time() - t0

        # PHASE 3: Hybrid Retrieve
        # Key fix: retrieve from ALL sessions (compacted), not just FTS5 hits
        t0 = time.time()
        context_parts = []
        tokens_used = 0

        # First: FTS5 search for direct keyword hits (high signal)
        search_results = await adapter.search(qtxt)
        key_terms = extract_key_terms(qtxt)
        for term in key_terms[:3]:
            try:
                term_results = await adapter.search(term)
                search_results.extend(term_results)
            except Exception:
                pass

        # Deduplicate search results
        seen = set()
        for r in search_results:
            content = r.get("content_snippet", r.get("content", ""))
            if content and content not in seen:
                seen.add(content)
                context_parts.append(f"[SEARCH HIT] {content}")
                tokens_used += len(content.split())

        # Second: retrieve compacted context from ALL sessions
        # After compaction, summaries are much smaller — we can fit many
        for sid in session_ids:
            if tokens_used >= CONTEXT_BUDGET:
                break
            remaining = CONTEXT_BUDGET - tokens_used
            ctx = await adapter.retrieve(f"session_{sid}", budget_tokens=min(remaining, 3000))
            if ctx:
                context_parts.append(f"--- Session {sid} ---\n{ctx}")
                tokens_used += len(ctx.split())

        retrieved_context = "\n\n".join(context_parts)
        time_retrieve = time.time() - t0

        # PHASE 4: Answer
        t0 = time.time()
        date_info = ""
        if session_dates:
            date_info = "\nConversation timeline:\n"
            for sid, date in zip(session_ids, session_dates):
                date_info += f"  Session {sid}: {date}\n"

        answer_prompt = f"""You are a personal AI assistant answering a question about past conversations with your user.
The retrieved context below contains relevant excerpts from your conversation history.
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

        print(f"  A: {hypothesis[:100]}")
        print(f"  Times: ingest={time_ingest:.1f}s compact={time_compact:.1f}s retrieve={time_retrieve:.1f}s answer={time_answer:.1f}s")

        return {
            "question_id": qid,
            "hypothesis": hypothesis,
            "_question_type": qtype,
            "_retrieval_hits": len(search_results),
            "_sessions": len(sessions),
            "_summaries": total_summaries,
            "_tokens_used": tokens_used,
            "_time_total": time_ingest + time_compact + time_retrieve + time_answer,
        }

    finally:
        adapter.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Strategy 2: Baseline (stuff everything, no LCM)
# ---------------------------------------------------------------------------

async def run_baseline_question(question: dict, llm: LLMBackend, idx: int, total: int) -> dict:
    qid = question["question_id"]
    qtype = question["question_type"]
    qtxt = question["question"]
    sessions = question["haystack_sessions"]
    session_ids = question["haystack_session_ids"]
    session_dates = question.get("haystack_dates", [])

    print(f"\n[BASE {idx+1}/{total}] {qid} ({qtype})")
    print(f"  Q: {qtxt[:100]}")

    # Concatenate all sessions up to token limit
    context_parts = []
    total_tokens = 0
    for sid, session in zip(session_ids, sessions):
        session_text = f"--- Session {sid} ---\n"
        for turn in session:
            session_text += f"[{turn['role']}]: {turn['content']}\n"
        context_parts.append(session_text)
        total_tokens += len(session_text.split())
        if total_tokens > MAX_BASELINE_TOKENS:
            break

    full_context = "\n".join(context_parts)

    date_info = ""
    if session_dates:
        date_info = "\nConversation timeline:\n"
        for sid, date in zip(session_ids, session_dates):
            date_info += f"  Session {sid}: {date}\n"

    prompt = f"""You are a personal AI assistant answering a question about past conversations with your user.
Use the conversation history below to answer. Be precise and specific.
If the question asks about something that was updated/changed, give the MOST RECENT value.
Only say you don't know if you truly cannot find the answer.

Question date: {question.get('question_date', 'unknown')}
{date_info}

Conversation history:
{full_context}

Question: {qtxt}

Answer concisely and directly in 1-2 sentences. State the specific answer first."""

    hypothesis = await llm.complete(prompt, max_tokens=256)
    print(f"  A: {hypothesis[:100]}")

    return {
        "question_id": qid,
        "hypothesis": hypothesis,
        "_question_type": qtype,
        "_context_tokens": total_tokens,
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

        status = "Y" if is_correct else "N"
        print(f"  [{status}] {qid} ({qtype})")

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

def write_report(lcm_eval: dict, baseline_eval: dict, lcm_llm: LLMBackend, baseline_llm: LLMBackend):
    lines = []
    lines.append("# LongMemEval Benchmark Report — lossless-agent")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {MODEL} | Judge: {JUDGE_MODEL}")
    lines.append(f"Dataset: longmemeval_s (500 questions, 6 types)")
    lines.append("")

    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Method | Overall | Task-Avg |")
    lines.append("|--------|---------|----------|")
    lines.append(f"| LCM (lossless-agent) | {lcm_eval['overall']:.1f}% ({lcm_eval['correct']}/{lcm_eval['total']}) | {lcm_eval['task_average']:.1f}% |")
    lines.append(f"| Baseline (full context) | {baseline_eval['overall']:.1f}% ({baseline_eval['correct']}/{baseline_eval['total']}) | {baseline_eval['task_average']:.1f}% |")
    lines.append("")

    lines.append("## By Question Type")
    lines.append("")
    lines.append("| Type | LCM | Baseline | Delta |")
    lines.append("|------|-----|----------|-------|")
    all_types = sorted(set(list(lcm_eval["by_type"].keys()) + list(baseline_eval["by_type"].keys())))
    for t in all_types:
        lcm_acc = lcm_eval["by_type"].get(t, {}).get("accuracy", 0)
        base_acc = baseline_eval["by_type"].get(t, {}).get("accuracy", 0)
        delta = lcm_acc - base_acc
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {t} | {lcm_acc:.1f}% | {base_acc:.1f}% | {sign}{delta:.1f}% |")
    lines.append("")

    lines.append("## Cost & Performance")
    lines.append("")
    lines.append(f"| Metric | LCM | Baseline |")
    lines.append(f"|--------|-----|----------|")
    lines.append(f"| LLM Calls | {lcm_llm.total_calls:,} | {baseline_llm.total_calls:,} |")
    lines.append(f"| Input Tokens | {lcm_llm.total_input_tokens:,} | {baseline_llm.total_input_tokens:,} |")
    lines.append(f"| Output Tokens | {lcm_llm.total_output_tokens:,} | {baseline_llm.total_output_tokens:,} |")
    lines.append(f"| Est. Cost | ${lcm_llm.total_cost:.2f} | ${baseline_llm.total_cost:.2f} |")
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
    lines.append(f"| **lossless-agent (LCM)** | **{lcm_eval['task_average']:.1f}%** |")
    lines.append(f"| Baseline (no memory system) | {baseline_eval['task_average']:.1f}% |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- LCM strategy: ingest all sessions → DAG compaction → hybrid retrieval (FTS5 + compacted summaries)")
    lines.append("- Baseline strategy: concatenate all raw sessions into context window")
    lines.append(f"- Context budget: {CONTEXT_BUDGET} tokens (LCM), {MAX_BASELINE_TOKENS} tokens (baseline)")
    lines.append("- Self-evaluation using claude-haiku-4-5 as judge (official benchmark uses GPT-4o)")
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
    print("LongMemEval Full Benchmark — lossless-agent")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions")

    # ---- RUN LCM ----
    print("\n" + "=" * 60)
    print("PHASE 1: LCM Strategy")
    print("=" * 60)

    lcm_llm = LLMBackend(MODEL, PROVIDER)
    lcm_results = load_existing_results(LCM_RESULTS)
    existing_ids = {r["question_id"] for r in lcm_results}
    print(f"Existing LCM results: {len(lcm_results)} (skipping)")

    for i, question in enumerate(data):
        if question["question_id"] in existing_ids:
            continue
        try:
            result = await run_lcm_question(question, lcm_llm, i, len(data))
            lcm_results.append(result)
            with open(LCM_RESULTS, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nLCM: {len(lcm_results)} results, {lcm_llm.total_calls} LLM calls, ${lcm_llm.total_cost:.2f}")

    # ---- RUN BASELINE ----
    print("\n" + "=" * 60)
    print("PHASE 2: Baseline Strategy")
    print("=" * 60)

    baseline_llm = LLMBackend(MODEL, PROVIDER)
    baseline_results = load_existing_results(BASELINE_RESULTS)
    existing_ids = {r["question_id"] for r in baseline_results}
    print(f"Existing baseline results: {len(baseline_results)} (skipping)")

    for i, question in enumerate(data):
        if question["question_id"] in existing_ids:
            continue
        try:
            result = await run_baseline_question(question, baseline_llm, i, len(data))
            baseline_results.append(result)
            with open(BASELINE_RESULTS, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nBaseline: {len(baseline_results)} results, {baseline_llm.total_calls} LLM calls, ${baseline_llm.total_cost:.2f}")

    # ---- EVALUATE ----
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation")
    print("=" * 60)

    judge_llm = LLMBackend(JUDGE_MODEL, PROVIDER)

    print("\nEvaluating LCM results...")
    lcm_eval = await evaluate(lcm_results, data, judge_llm, "lcm")

    print("\nEvaluating baseline results...")
    baseline_eval = await evaluate(baseline_results, data, judge_llm, "baseline")

    # Save eval details
    with open(EVAL_RESULTS, "w") as f:
        json.dump({"lcm": lcm_eval, "baseline": baseline_eval}, f, indent=2)

    # ---- REPORT ----
    print("\n" + "=" * 60)
    print("PHASE 4: Report")
    print("=" * 60)

    report = write_report(lcm_eval, baseline_eval, lcm_llm, baseline_llm)
    print(report)

    print("\n\nDONE!")


if __name__ == "__main__":
    asyncio.run(main())
