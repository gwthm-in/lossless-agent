#!/usr/bin/env python3
"""LongMemEval Benchmark for lossless-agent.

Measures how well LCM's DAG-based compaction preserves information from
long conversation histories. The benchmark:

1. Ingests each question's haystack sessions into lossless-agent
2. Compacts them (simulating context pressure)
3. Uses lcm_grep + expansion to retrieve relevant context
4. Has an LLM answer based on retrieved context
5. Outputs JSONL for LongMemEval evaluation

Usage:
    python run_benchmark.py \
        --data ../LongMemEval/data/longmemeval_s \
        --output results.jsonl \
        --model claude-haiku-4-5 \
        --limit 10  # optional: run on N questions only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add parent to path for lossless-agent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lossless_agent.adapters.simple import SimpleAdapter


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------

class LLMBackend:
    """Handles LLM calls for summarization and answering."""

    def __init__(self, model: str, provider: str = "anthropic"):
        self.model = model
        self.provider = provider
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def _get_client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call LLM and return response text."""
        client = self._get_client()
        self.total_calls += 1

        if self.provider == "anthropic":
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ))
            self.total_input_tokens += resp.usage.input_tokens
            self.total_output_tokens += resp.usage.output_tokens
            return resp.content[0].text

        elif self.provider == "openai":
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ))
            usage = resp.usage
            if usage:
                self.total_input_tokens += usage.prompt_tokens
                self.total_output_tokens += usage.completion_tokens
            return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    data_path: str
    output_path: str
    model: str = "claude-haiku-4-5"
    provider: str = "anthropic"
    judge_model: str = "claude-haiku-4-5"  # for self-eval (official uses gpt-4o)
    judge_provider: str = "anthropic"
    limit: Optional[int] = None
    context_budget: int = 8000  # tokens for retrieved context
    compaction_budget: int = 4000  # aggressive compaction to test recall
    skip_existing: bool = True
    verbose: bool = False
    question_types: Optional[List[str]] = None


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    question: str
    answer: str
    hypothesis: str
    retrieval_context: str
    num_sessions_ingested: int
    num_summaries_created: int
    retrieval_hits: int
    time_ingest: float
    time_compact: float
    time_retrieve: float
    time_answer: float


async def run_single_question(
    question: dict,
    llm: LLMBackend,
    config: BenchmarkConfig,
    question_idx: int,
    total: int,
) -> QuestionResult:
    """Process a single LongMemEval question through the LCM pipeline."""

    qid = question["question_id"]
    qtype = question["question_type"]
    qtxt = question["question"]
    answer = question["answer"]
    sessions = question["haystack_sessions"]
    session_ids = question["haystack_session_ids"]
    session_dates = question.get("haystack_dates", [])

    print(f"\n[{question_idx+1}/{total}] {qid} ({qtype})")
    print(f"  Q: {qtxt[:100]}...")
    print(f"  A: {answer[:100]}...")
    print(f"  Sessions: {len(sessions)}, Answer sessions: {question['answer_session_ids']}")

    # Create a temp DB for this question (isolated)
    tmp_dir = tempfile.mkdtemp(prefix=f"lcm_bench_{qid}_")
    db_path = os.path.join(tmp_dir, "lcm.db")

    async def summarize_fn(prompt: str) -> str:
        return await llm.complete(prompt, max_tokens=512)

    adapter = SimpleAdapter(db_path, summarize_fn)

    try:
        # ----- PHASE 1: Ingest all sessions -----
        t0 = time.time()
        total_turns = 0
        for i, (session, sid) in enumerate(zip(sessions, session_ids)):
            session_key = f"session_{sid}"
            messages = []
            for turn in session:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"],
                    "token_count": len(turn["content"].split()),
                })
                total_turns += 1
            await adapter.ingest(session_key, messages)
        time_ingest = time.time() - t0
        print(f"  Ingested: {len(sessions)} sessions, {total_turns} turns ({time_ingest:.1f}s)")

        # ----- PHASE 2: Compact all sessions -----
        t0 = time.time()
        total_summaries = 0
        for sid in session_ids:
            session_key = f"session_{sid}"
            n = await adapter.compact(session_key)
            total_summaries += n
        time_compact = time.time() - t0
        print(f"  Compacted: {total_summaries} summaries created ({time_compact:.1f}s)")

        # ----- PHASE 3: Retrieve relevant context -----
        t0 = time.time()

        # Strategy: Use lcm_grep across all sessions, then assemble context
        # from the best-matching sessions
        search_results = await adapter.search(qtxt)

        # Also search for key terms from the question
        key_terms = extract_key_terms(qtxt)
        for term in key_terms[:3]:
            term_results = await adapter.search(term)
            search_results.extend(term_results)

        # Deduplicate by content
        seen = set()
        unique_results = []
        for r in search_results:
            content = r.get("content_snippet", r.get("content", ""))
            if content not in seen:
                seen.add(content)
                unique_results.append(r)

        # Get the full context from the most relevant sessions
        # Find which sessions had hits
        hit_sessions = set()
        for r in unique_results:
            conv_id = r.get("conversation_id")
            if conv_id:
                hit_sessions.add(conv_id)

        # Assemble context from matching sessions
        context_parts = []
        tokens_used = 0

        # First: add direct search hits
        for r in unique_results[:20]:
            snippet = r.get("content_snippet", r.get("content", ""))
            if snippet and tokens_used < config.context_budget:
                context_parts.append(snippet)
                tokens_used += len(snippet.split())

        # Second: retrieve full context from the best-matching sessions
        for sid in session_ids:
            if tokens_used >= config.context_budget:
                break
            session_key = f"session_{sid}"
            ctx = await adapter.retrieve(session_key, budget_tokens=config.context_budget - tokens_used)
            if ctx:
                context_parts.append(f"--- Session {sid} ---\n{ctx}")
                tokens_used += len(ctx.split())

        retrieved_context = "\n\n".join(context_parts)
        time_retrieve = time.time() - t0
        print(f"  Retrieved: {len(unique_results)} hits, {tokens_used} tokens ({time_retrieve:.1f}s)")

        # ----- PHASE 4: Answer the question -----
        t0 = time.time()

        # Include session dates for temporal reasoning
        date_info = ""
        if session_dates:
            date_info = "\n\nConversation timeline:\n"
            for sid, date in zip(session_ids, session_dates):
                date_info += f"  Session {sid}: {date}\n"

        answer_prompt = f"""You are a personal AI assistant answering a question about past conversations with your user.
The retrieved context below contains relevant excerpts from your conversation history.
Answer based on what you find in the context. Give your best answer even if context is partial.
Only say you don't know if the context is completely irrelevant to the question.

Question date: {question.get('question_date', 'unknown')}
{date_info}

Retrieved conversation context:
{retrieved_context}

Question: {qtxt}

Answer concisely and directly in 1-2 sentences. State the answer first."""

        hypothesis = await llm.complete(answer_prompt, max_tokens=256)
        time_answer = time.time() - t0
        print(f"  Answer: {hypothesis[:100]}...")

        return QuestionResult(
            question_id=qid,
            question_type=qtype,
            question=qtxt,
            answer=answer,
            hypothesis=hypothesis,
            retrieval_context=retrieved_context[:500],  # truncate for logging
            num_sessions_ingested=len(sessions),
            num_summaries_created=total_summaries,
            retrieval_hits=len(unique_results),
            time_ingest=time_ingest,
            time_compact=time_compact,
            time_retrieve=time_retrieve,
            time_answer=time_answer,
        )

    finally:
        adapter.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_key_terms(question: str) -> List[str]:
    """Extract meaningful search terms from a question."""
    # Remove common question words
    stop_words = {
        "what", "when", "where", "who", "how", "which", "why", "did",
        "does", "was", "were", "is", "are", "the", "a", "an", "my",
        "your", "i", "me", "we", "you", "do", "have", "has", "had",
        "that", "this", "with", "for", "from", "about", "after",
        "before", "first", "last", "most", "recently", "ever",
        "any", "some", "can", "could", "would", "should",
    }
    words = question.lower().replace("?", "").replace(",", "").split()
    terms = [w for w in words if w not in stop_words and len(w) > 2]

    # Try bigrams for more specific searches
    bigrams = []
    for i in range(len(terms) - 1):
        bigrams.append(f"{terms[i]} {terms[i+1]}")

    return bigrams[:2] + terms[:5]


async def run_benchmark(config: BenchmarkConfig):
    """Run the full LongMemEval benchmark."""

    print(f"Loading data from {config.data_path}...")
    with open(config.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions")

    # Filter by question type if specified
    if config.question_types:
        data = [q for q in data if q["question_type"] in config.question_types]
        print(f"Filtered to {len(data)} questions of types: {config.question_types}")

    # Apply limit
    if config.limit:
        data = data[:config.limit]
        print(f"Limited to {config.limit} questions")

    # Load existing results to skip
    existing_ids = set()
    if config.skip_existing and os.path.exists(config.output_path):
        with open(config.output_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    existing_ids.add(obj["question_id"])
        print(f"Skipping {len(existing_ids)} already-completed questions")

    llm = LLMBackend(config.model, config.provider)

    results: List[QuestionResult] = []
    errors = []

    for i, question in enumerate(data):
        if question["question_id"] in existing_ids:
            continue

        try:
            result = await run_single_question(question, llm, config, i, len(data))
            results.append(result)

            # Write result to JSONL immediately (append mode)
            with open(config.output_path, "a") as f:
                f.write(json.dumps({
                    "question_id": result.question_id,
                    "hypothesis": result.hypothesis,
                    # Extra fields for analysis (LongMemEval eval ignores these)
                    "_question_type": result.question_type,
                    "_retrieval_hits": result.retrieval_hits,
                    "_num_summaries": result.num_summaries_created,
                    "_time_total": (
                        result.time_ingest + result.time_compact +
                        result.time_retrieve + result.time_answer
                    ),
                }) + "\n")

        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append({"question_id": question["question_id"], "error": str(e)})
            import traceback
            traceback.print_exc()

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Questions processed: {len(results)}")
    print(f"Errors: {len(errors)}")
    if results:
        avg_ingest = sum(r.time_ingest for r in results) / len(results)
        avg_compact = sum(r.time_compact for r in results) / len(results)
        avg_retrieve = sum(r.time_retrieve for r in results) / len(results)
        avg_answer = sum(r.time_answer for r in results) / len(results)
        avg_hits = sum(r.retrieval_hits for r in results) / len(results)
        avg_summaries = sum(r.num_summaries_created for r in results) / len(results)
        print("\nAverage times:")
        print(f"  Ingest:   {avg_ingest:.2f}s")
        print(f"  Compact:  {avg_compact:.2f}s")
        print(f"  Retrieve: {avg_retrieve:.2f}s")
        print(f"  Answer:   {avg_answer:.2f}s")
        print(f"  Total:    {avg_ingest + avg_compact + avg_retrieve + avg_answer:.2f}s")
        print(f"\nAverage retrieval hits: {avg_hits:.1f}")
        print(f"Average summaries created: {avg_summaries:.1f}")

    print("\nLLM usage:")
    print(f"  Calls: {llm.total_calls}")
    print(f"  Input tokens: {llm.total_input_tokens:,}")
    print(f"  Output tokens: {llm.total_output_tokens:,}")

    print(f"\nResults written to: {config.output_path}")

    if errors:
        err_path = config.output_path.replace(".jsonl", "_errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors written to: {err_path}")


# ---------------------------------------------------------------------------
# Self-Evaluation (when GPT-4o isn't available)
# ---------------------------------------------------------------------------

async def self_evaluate(config: BenchmarkConfig, reference_path: str):
    """Run evaluation using our own LLM as judge (unofficial but useful)."""

    print(f"\nRunning self-evaluation with {config.judge_model}...")

    # Load reference data
    with open(reference_path) as f:
        ref_data = json.load(f)
    ref_by_id = {q["question_id"]: q for q in ref_data}

    # Load predictions
    predictions = []
    with open(config.output_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    llm = LLMBackend(config.judge_model, config.judge_provider)

    correct = 0
    total = 0
    by_type = {}

    eval_results_path = config.output_path.replace(".jsonl", f"_eval_{config.judge_model}.jsonl")

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

Did the model correctly identify that it cannot answer this question? Answer yes or no only."""
        elif qtype == "temporal-reasoning":
            judge_prompt = f"""Question: "{ref['question']}"
Correct answer: "{ref['answer']}"
Model response: "{pred['hypothesis']}"

Does the model response contain the correct answer? For temporal questions, allow off-by-one errors in day/week/month counts. Answer yes or no only."""
        elif qtype == "knowledge-update":
            judge_prompt = f"""Question: "{ref['question']}"
Correct (updated) answer: "{ref['answer']}"
Model response: "{pred['hypothesis']}"

Does the model response contain the correct updated answer? It's OK if it also mentions old information, as long as the updated answer is present. Answer yes or no only."""
        else:
            judge_prompt = f"""Question: "{ref['question']}"
Correct answer: "{ref['answer']}"
Model response: "{pred['hypothesis']}"

Does the model response contain the correct answer? Answer yes or no only."""

        verdict = await llm.complete(judge_prompt, max_tokens=10)
        is_correct = "yes" in verdict.lower()

        if qtype not in by_type:
            by_type[qtype] = {"correct": 0, "total": 0}
        by_type[qtype]["total"] += 1
        if is_correct:
            by_type[qtype]["correct"] += 1
            correct += 1
        total += 1

        # Write eval result
        with open(eval_results_path, "a") as f:
            f.write(json.dumps({
                "question_id": qid,
                "hypothesis": pred["hypothesis"],
                "autoeval_label": {"model": config.judge_model, "label": is_correct},
            }) + "\n")

        status = "✓" if is_correct else "✗"
        print(f"  {status} {qid} ({qtype})")

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS (judge: {config.judge_model})")
    print(f"{'=' * 60}")
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print("\nBy question type:")
    type_accs = []
    for qtype, counts in sorted(by_type.items()):
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        type_accs.append(acc)
        print(f"  {qtype}: {counts['correct']}/{counts['total']} = {acc:.1f}%")
    if type_accs:
        print(f"\nTask-averaged accuracy: {sum(type_accs)/len(type_accs):.1f}%")
    print(f"\nEval results: {eval_results_path}")


# ---------------------------------------------------------------------------
# Baseline: No compaction (just stuff everything into context)
# ---------------------------------------------------------------------------

async def run_baseline(config: BenchmarkConfig):
    """Run a no-LCM baseline: just concatenate all sessions and ask."""

    print("Running NO-LCM BASELINE (full context stuffing)...")

    with open(config.data_path) as f:
        data = json.load(f)

    if config.question_types:
        data = [q for q in data if q["question_type"] in config.question_types]
    if config.limit:
        data = data[:config.limit]

    llm = LLMBackend(config.model, config.provider)
    baseline_path = config.output_path.replace(".jsonl", "_baseline.jsonl")

    for i, question in enumerate(data):
        qid = question["question_id"]
        qtxt = question["question"]
        sessions = question["haystack_sessions"]
        session_dates = question.get("haystack_dates", [])
        session_ids = question["haystack_session_ids"]

        print(f"\n[{i+1}/{len(data)}] {qid} ({question['question_type']})")

        # Just concatenate everything
        context_parts = []
        total_tokens = 0
        for sid, session in zip(session_ids, sessions):
            session_text = f"--- Session {sid} ---\n"
            for turn in session:
                session_text += f"[{turn['role']}]: {turn['content']}\n"
            context_parts.append(session_text)
            total_tokens += len(session_text.split())

            # Cap at ~100k tokens to avoid API limits
            if total_tokens > 100000:
                break

        full_context = "\n".join(context_parts)

        date_info = ""
        if session_dates:
            date_info = "\nConversation timeline:\n"
            for sid, date in zip(session_ids, session_dates):
                date_info += f"  Session {sid}: {date}\n"

        prompt = f"""You are answering a question about past conversations.
Use the conversation history below to answer.

Question date: {question.get('question_date', 'unknown')}
{date_info}

Conversation history:
{full_context}

Question: {qtxt}

Answer concisely and directly."""

        try:
            hypothesis = await llm.complete(prompt, max_tokens=256)
            print(f"  Answer: {hypothesis[:100]}...")

            with open(baseline_path, "a") as f:
                f.write(json.dumps({
                    "question_id": qid,
                    "hypothesis": hypothesis,
                    "_question_type": question["question_type"],
                    "_context_tokens": total_tokens,
                }) + "\n")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nBaseline results: {baseline_path}")
    print(f"LLM usage: {llm.total_calls} calls, {llm.total_input_tokens:,} input, {llm.total_output_tokens:,} output tokens")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for lossless-agent")
    parser.add_argument("--data", required=True, help="Path to longmemeval JSON file")
    parser.add_argument("--output", default="results.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="claude-haiku-4-5", help="LLM model for summarization + answering")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--limit", type=int, help="Process only N questions")
    parser.add_argument("--context-budget", type=int, default=8000, help="Token budget for retrieval")
    parser.add_argument("--types", nargs="+", help="Only run specific question types")
    parser.add_argument("--baseline", action="store_true", help="Run no-LCM baseline instead")
    parser.add_argument("--evaluate", action="store_true", help="Run self-evaluation after benchmark")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = BenchmarkConfig(
        data_path=args.data,
        output_path=args.output,
        model=args.model,
        provider=args.provider,
        limit=args.limit,
        context_budget=args.context_budget,
        question_types=args.types,
        verbose=args.verbose,
    )

    if args.baseline:
        asyncio.run(run_baseline(config))
    else:
        asyncio.run(run_benchmark(config))

    if args.evaluate:
        asyncio.run(self_evaluate(config, args.data))


if __name__ == "__main__":
    main()
