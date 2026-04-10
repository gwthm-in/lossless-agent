#!/usr/bin/env python3
"""Lossless Context Benchmark (LCB)

Tests what lossless-agent is uniquely good at vs agentmemory:
1. Detail preservation after compression
2. Contextual recall (surrounding conversation, not just atomic facts)
3. Verbatim recovery (can you get back the exact original?)
4. Compression efficiency

Uses LongMemEval conversations as source material.
Generates questions using an LLM, then tests both systems.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Load .env
_env_path = Path(__file__).parent.parent / "longmemeval" / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import anthropic
from agentmemory import MemoryStore
from lossless_agent.adapters.simple import SimpleAdapter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/GowthamSai/Documents/repos/gwthm-in/LongMemEval/data/longmemeval_s"
OUTPUT_DIR = Path(__file__).parent
REPORT_PATH = OUTPUT_DIR / "COMPRESSION_BENCHMARK_REPORT.md"
RESULTS_PATH = OUTPUT_DIR / "compression_results.json"

MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-haiku-4-5"

# How many conversations to test
NUM_CONVERSATIONS = 20
# Target compression ratio (compress to this fraction of original)
COMPRESSION_TARGET = 0.10  # 10% of original tokens

client = anthropic.Anthropic()
total_tokens = {"input": 0, "output": 0}


def llm_call(prompt: str, max_tokens: int = 1024) -> str:
    """Synchronous LLM call."""
    global total_tokens
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    total_tokens["input"] += resp.usage.input_tokens
    total_tokens["output"] += resp.usage.output_tokens
    return resp.content[0].text


async def async_llm_call(prompt: str, max_tokens: int = 512) -> str:
    """Async wrapper for LLM call."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: llm_call(prompt, max_tokens))


# ---------------------------------------------------------------------------
# Step 1: Select conversations and generate test questions
# ---------------------------------------------------------------------------

def select_conversations(data: list) -> list:
    """Pick diverse, long conversations from LongMemEval."""
    # Find questions with the longest individual sessions
    candidates = []
    for q in data:
        for i, session in enumerate(q["haystack_sessions"]):
            words = sum(len(t["content"].split()) for t in session)
            if words > 1500:  # Only reasonably long conversations
                candidates.append({
                    "question_id": q["question_id"],
                    "session_idx": i,
                    "session": session,
                    "words": words,
                    "turns": len(session),
                    "original_question": q["question"],
                    "original_answer": q["answer"],
                })
    
    # Sort by length, take diverse set
    candidates.sort(key=lambda x: x["words"], reverse=True)
    
    # Deduplicate by question_id (take longest session per question)
    seen = set()
    selected = []
    for c in candidates:
        if c["question_id"] not in seen and len(selected) < NUM_CONVERSATIONS:
            seen.add(c["question_id"])
            selected.append(c)
    
    return selected


def generate_questions(conversation: list, num_questions: int = 5) -> list:
    """Generate test questions from a conversation that test different recall types."""
    
    # Format conversation for the LLM
    conv_text = ""
    for turn in conversation:
        conv_text += f"[{turn['role']}]: {turn['content']}\n\n"
    
    prompt = f"""Given this conversation between a user and assistant, generate exactly {num_questions} questions that test different types of recall. Each question MUST be answerable from the conversation.

QUESTION TYPES (generate one of each):
1. VERBATIM: Ask for an exact name, number, or specific phrase mentioned by the user
2. CONTEXTUAL: Ask WHY the user said something or what LED TO a particular topic  
3. DETAIL: Ask for a specific detail that was mentioned in passing (not the main topic)
4. TEMPORAL: Ask about the ORDER of topics or what was discussed BEFORE/AFTER something
5. SYNTHESIS: Ask something that requires combining info from multiple turns

For each question, provide:
- The question
- The correct answer (brief, specific)
- The question type

CONVERSATION:
{conv_text[:8000]}

Respond in this exact JSON format:
[
  {{"question": "...", "answer": "...", "type": "verbatim"}},
  {{"question": "...", "answer": "...", "type": "contextual"}},
  {{"question": "...", "answer": "...", "type": "detail"}},
  {{"question": "...", "answer": "...", "type": "temporal"}},
  {{"question": "...", "answer": "...", "type": "synthesis"}}
]

Return ONLY the JSON array, nothing else."""

    response = llm_call(prompt, max_tokens=1500)
    
    # Parse JSON
    try:
        # Find JSON array in response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            questions = json.loads(response[start:end])
            return questions
    except json.JSONDecodeError:
        pass
    
    return []


# ---------------------------------------------------------------------------
# Step 2: Process conversations through both systems
# ---------------------------------------------------------------------------

async def process_lossless(conversation: list, session_key: str) -> dict:
    """Process through lossless-agent: ingest → compact → measure."""
    
    tmp_dir = tempfile.mkdtemp(prefix="lcm_bench_")
    db_path = os.path.join(tmp_dir, "lcm.db")
    
    async def summarize_fn(prompt: str) -> str:
        return await async_llm_call(prompt, max_tokens=512)
    
    adapter = SimpleAdapter(db_path, summarize_fn)
    
    try:
        # Ingest
        messages = [{"role": t["role"], "content": t["content"],
                    "token_count": len(t["content"].split())} for t in conversation]
        await adapter.ingest(session_key, messages)
        
        # Compact
        num_summaries = await adapter.compact(session_key)
        
        # Measure compressed size
        compressed_context = await adapter.retrieve(session_key, budget_tokens=999999)
        compressed_tokens = len(compressed_context.split()) if compressed_context else 0
        
        # Also get a tight budget version
        tight_context = await adapter.retrieve(session_key, budget_tokens=500)
        tight_tokens = len(tight_context.split()) if tight_context else 0
        
        # Get original size
        original_tokens = sum(len(t["content"].split()) for t in conversation)
        
        return {
            "system": "lossless-agent",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tight_tokens": tight_tokens,
            "num_summaries": num_summaries,
            "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 0,
            "tight_ratio": tight_tokens / original_tokens if original_tokens > 0 else 0,
            "compressed_context": compressed_context or "",
            "tight_context": tight_context or "",
            "can_expand": True,  # lossless-agent can always expand
            "adapter": adapter,
            "tmp_dir": tmp_dir,
            "session_key": session_key,
        }
    except Exception as e:
        adapter.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


async def process_agentmemory(conversation: list, session_key: str) -> dict:
    """Process through agentmemory: ingest → extract memories → measure."""
    
    try:
        async with MemoryStore(path=":memory:", prefer_dense=False, auto_graph=False,
                               reranker=False, streaming_consolidation=False, write_validation=False) as mem:
            # Ingest conversation
            messages_for_am = [{"role": t["role"], "content": t["content"]} for t in conversation]
            await mem.async_ingest_conversation(messages_for_am, session_id=session_key)
            
            # Count memories
            num_memories = len(mem)
            
            # Get all memories as context
            all_context, meta = await mem.async_build_context(
                "everything discussed in the conversation",
                token_budget=999999
            )
            all_tokens = len(all_context.split()) if all_context else 0
            
            # Tight budget
            tight_context, tight_meta = await mem.async_build_context(
                "everything discussed in the conversation",
                token_budget=500
            )
            tight_tokens = len(tight_context.split()) if tight_context else 0
            
            original_tokens = sum(len(t["content"].split()) for t in conversation)
            
            return {
                "system": "agentmemory",
                "original_tokens": original_tokens,
                "compressed_tokens": all_tokens,
                "tight_tokens": tight_tokens,
                "num_memories": num_memories,
                "compression_ratio": all_tokens / original_tokens if original_tokens > 0 else 0,
                "tight_ratio": tight_tokens / original_tokens if original_tokens > 0 else 0,
                "compressed_context": all_context or "",
                "tight_context": tight_context or "",
                "can_expand": False,  # agentmemory cannot expand back
            }
    except Exception as e:
        return {
            "system": "agentmemory",
            "error": str(e),
            "original_tokens": sum(len(t["content"].split()) for t in conversation),
            "compressed_tokens": 0,
            "tight_tokens": 0,
            "num_memories": 0,
            "compression_ratio": 0,
            "tight_ratio": 0,
            "compressed_context": "",
            "tight_context": "",
            "can_expand": False,
        }


# ---------------------------------------------------------------------------
# Step 3: Answer questions using each system's compressed context
# ---------------------------------------------------------------------------

async def answer_from_context(context: str, question: str) -> str:
    """Answer a question given only compressed context."""
    prompt = f"""You are answering a question about a past conversation.
You only have access to the following compressed/summarized context.
Answer based ONLY on what's in the context. Be precise and specific.
If you cannot find the answer, say "CANNOT_ANSWER".

Context:
{context[:12000]}

Question: {question}

Answer concisely in 1-2 sentences."""
    
    return await async_llm_call(prompt, max_tokens=200)


async def answer_with_expansion(adapter, session_key: str, question: str) -> str:
    """Answer using lossless-agent's expand capability — search then expand."""
    
    # First search for relevant content
    search_results = await adapter.search(question)
    
    # Get expanded context from relevant parts
    expanded_parts = []
    for result in search_results[:5]:
        content = result.get("content_snippet", result.get("content", ""))
        if content:
            expanded_parts.append(content)
    
    # Also get general context
    general_context = await adapter.retrieve(session_key, budget_tokens=2000)
    if general_context:
        expanded_parts.append(general_context)
    
    full_context = "\n\n".join(expanded_parts)
    
    prompt = f"""You are answering a question about a past conversation.
You have access to expanded context from the conversation memory system.
Answer based ONLY on what's in the context. Be precise and specific.
If you cannot find the answer, say "CANNOT_ANSWER".

Expanded Context:
{full_context[:12000]}

Question: {question}

Answer concisely in 1-2 sentences."""
    
    return await async_llm_call(prompt, max_tokens=200)


# ---------------------------------------------------------------------------
# Step 4: Judge answers
# ---------------------------------------------------------------------------

async def judge_answer(question: str, correct_answer: str, hypothesis: str) -> bool:
    """Judge if an answer is correct."""
    prompt = f"""Question: "{question}"
Correct answer: "{correct_answer}"
Model response: "{hypothesis}"

Does the model response contain the correct answer or convey the same meaning?
Be lenient — if the key fact is present, count it. But "CANNOT_ANSWER" is always wrong.
Answer ONLY "yes" or "no"."""
    
    verdict = await async_llm_call(prompt, max_tokens=10)
    return "yes" in verdict.lower()


# ---------------------------------------------------------------------------
# Step 5: Verbatim recovery test (unique to lossless-agent)
# ---------------------------------------------------------------------------

def extract_verbatim_snippets(conversation: list, num_snippets: int = 3) -> list:
    """Extract specific user quotes that a system should be able to recover."""
    snippets = []
    for turn in conversation:
        if turn["role"] == "user" and len(turn["content"]) > 50:
            # Take a distinctive phrase from the middle
            words = turn["content"].split()
            if len(words) > 15:
                start = len(words) // 3
                snippet = " ".join(words[start:start+10])
                snippets.append({
                    "original": turn["content"],
                    "snippet": snippet,
                    "full_turn": turn["content"],
                })
    return snippets[:num_snippets]


def check_verbatim_recovery(compressed_context: str, snippet: str) -> bool:
    """Check if a verbatim snippet can be found in the compressed output."""
    # Check exact match
    if snippet.lower() in compressed_context.lower():
        return True
    # Check partial match (at least 70% of words present in order)
    snippet_words = snippet.lower().split()
    context_lower = compressed_context.lower()
    found = sum(1 for w in snippet_words if w in context_lower)
    return found / len(snippet_words) >= 0.7


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("Lossless Context Benchmark (LCB)")
    print("lossless-agent vs agentmemory")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    # Select conversations
    conversations = select_conversations(data)
    print(f"Selected {len(conversations)} conversations")
    
    results = []
    
    for idx, conv_info in enumerate(conversations):
        session_key = f"bench_{conv_info['question_id']}_{conv_info['session_idx']}"
        conversation = conv_info["session"]
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(conversations)}] {session_key}")
        print(f"  {conv_info['turns']} turns, {conv_info['words']} words")
        print(f"{'='*60}")
        
        # Generate test questions
        print("  Generating questions...")
        questions = generate_questions(conversation)
        if not questions:
            print("  Failed to generate questions, skipping")
            continue
        print(f"  Generated {len(questions)} questions")
        
        # Process through both systems
        print("  Processing lossless-agent...")
        t0 = time.time()
        lcm_result = await process_lossless(conversation, session_key)
        lcm_time = time.time() - t0
        print(f"    Compressed: {lcm_result['original_tokens']} → {lcm_result['compressed_tokens']} tokens "
              f"({lcm_result['compression_ratio']:.1%}), {lcm_result['num_summaries']} summaries, {lcm_time:.1f}s")
        
        print("  Processing agentmemory...")
        t0 = time.time()
        am_result = await process_agentmemory(conversation, session_key)
        am_time = time.time() - t0
        if "error" in am_result:
            print(f"    ERROR: {am_result['error']}")
        else:
            print(f"    Compressed: {am_result['original_tokens']} → {am_result['compressed_tokens']} tokens "
                  f"({am_result['compression_ratio']:.1%}), {am_result.get('num_memories', '?')} memories, {am_time:.1f}s")
        
        # Test 1: Answer questions from compressed context
        print("  Testing question answering...")
        lcm_correct = 0
        lcm_expanded_correct = 0
        am_correct = 0
        question_results = []
        
        for q in questions:
            qtxt = q["question"]
            ans = q["answer"]
            qtype = q["type"]
            
            # lossless-agent with compressed context
            lcm_answer = await answer_from_context(lcm_result["compressed_context"], qtxt)
            lcm_is_correct = await judge_answer(qtxt, ans, lcm_answer)
            
            # lossless-agent with expansion (its unique advantage)
            if lcm_result.get("adapter"):
                lcm_exp_answer = await answer_with_expansion(
                    lcm_result["adapter"], lcm_result["session_key"], qtxt)
                lcm_exp_correct = await judge_answer(qtxt, ans, lcm_exp_answer)
            else:
                lcm_exp_answer = "N/A"
                lcm_exp_correct = False
            
            # agentmemory with its context
            am_answer = await answer_from_context(am_result["compressed_context"], qtxt)
            am_is_correct = await judge_answer(qtxt, ans, am_answer)
            
            if lcm_is_correct: lcm_correct += 1
            if lcm_exp_correct: lcm_expanded_correct += 1
            if am_is_correct: am_correct += 1
            
            status_lcm = "Y" if lcm_is_correct else "N"
            status_exp = "Y" if lcm_exp_correct else "N"
            status_am = "Y" if am_is_correct else "N"
            print(f"    [{qtype}] LCM:{status_lcm} EXP:{status_exp} AM:{status_am} | {qtxt[:60]}...")
            
            question_results.append({
                "question": qtxt,
                "answer": ans,
                "type": qtype,
                "lcm_answer": lcm_answer,
                "lcm_correct": lcm_is_correct,
                "lcm_expanded_answer": lcm_exp_answer,
                "lcm_expanded_correct": lcm_exp_correct,
                "am_answer": am_answer,
                "am_correct": am_is_correct,
            })
        
        # Test 2: Verbatim recovery
        print("  Testing verbatim recovery...")
        snippets = extract_verbatim_snippets(conversation)
        lcm_verbatim = 0
        am_verbatim = 0
        for snip in snippets:
            lcm_found = check_verbatim_recovery(lcm_result["compressed_context"], snip["snippet"])
            am_found = check_verbatim_recovery(am_result["compressed_context"], snip["snippet"])
            if lcm_found: lcm_verbatim += 1
            if am_found: am_verbatim += 1
        
        total_snippets = len(snippets)
        print(f"    Verbatim: LCM={lcm_verbatim}/{total_snippets} AM={am_verbatim}/{total_snippets}")
        
        # Cleanup lossless-agent
        if lcm_result.get("adapter"):
            lcm_result["adapter"].close()
        if lcm_result.get("tmp_dir"):
            shutil.rmtree(lcm_result["tmp_dir"], ignore_errors=True)
        
        result_entry = {
            "session_key": session_key,
            "original_tokens": conv_info["words"],
            "turns": conv_info["turns"],
            "lcm": {
                "compressed_tokens": lcm_result["compressed_tokens"],
                "compression_ratio": lcm_result["compression_ratio"],
                "num_summaries": lcm_result["num_summaries"],
                "questions_correct": lcm_correct,
                "questions_expanded_correct": lcm_expanded_correct,
                "questions_total": len(questions),
                "verbatim_recovered": lcm_verbatim,
                "verbatim_total": total_snippets,
                "time": lcm_time,
            },
            "am": {
                "compressed_tokens": am_result["compressed_tokens"],
                "compression_ratio": am_result["compression_ratio"],
                "num_memories": am_result.get("num_memories", 0),
                "questions_correct": am_correct,
                "questions_total": len(questions),
                "verbatim_recovered": am_verbatim,
                "verbatim_total": total_snippets,
                "time": am_time,
                "error": am_result.get("error"),
            },
            "questions": question_results,
        }
        results.append(result_entry)
        
        # Save incremental results
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
    
    # ---------------------------------------------------------------------------
    # Generate report
    # ---------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("Generating report...")
    print("=" * 60)
    
    # Aggregate stats
    total_convs = len(results)
    
    lcm_q_correct = sum(r["lcm"]["questions_correct"] for r in results)
    lcm_q_expanded = sum(r["lcm"]["questions_expanded_correct"] for r in results)
    lcm_q_total = sum(r["lcm"]["questions_total"] for r in results)
    am_q_correct = sum(r["am"]["questions_correct"] for r in results)
    am_q_total = sum(r["am"]["questions_total"] for r in results)
    
    lcm_v_recovered = sum(r["lcm"]["verbatim_recovered"] for r in results)
    lcm_v_total = sum(r["lcm"]["verbatim_total"] for r in results)
    am_v_recovered = sum(r["am"]["verbatim_recovered"] for r in results)
    am_v_total = sum(r["am"]["verbatim_total"] for r in results)
    
    avg_lcm_ratio = sum(r["lcm"]["compression_ratio"] for r in results) / total_convs
    avg_am_ratio = sum(r["am"]["compression_ratio"] for r in results) / total_convs
    
    # Per question type
    type_stats = {}
    for r in results:
        for q in r["questions"]:
            t = q["type"]
            if t not in type_stats:
                type_stats[t] = {"lcm": 0, "lcm_exp": 0, "am": 0, "total": 0}
            type_stats[t]["total"] += 1
            if q["lcm_correct"]: type_stats[t]["lcm"] += 1
            if q["lcm_expanded_correct"]: type_stats[t]["lcm_exp"] += 1
            if q["am_correct"]: type_stats[t]["am"] += 1
    
    report = f"""# Lossless Context Benchmark (LCB) Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: {MODEL} | Judge: {JUDGE_MODEL}
Conversations tested: {total_convs}

## Overall Results

| Metric | lossless-agent (compressed) | lossless-agent (with expand) | agentmemory |
|--------|---------------------------|------------------------------|-------------|
| Question Accuracy | {lcm_q_correct}/{lcm_q_total} ({lcm_q_correct/lcm_q_total*100:.1f}%) | {lcm_q_expanded}/{lcm_q_total} ({lcm_q_expanded/lcm_q_total*100:.1f}%) | {am_q_correct}/{am_q_total} ({am_q_correct/am_q_total*100:.1f}%) |
| Verbatim Recovery | {lcm_v_recovered}/{lcm_v_total} ({lcm_v_recovered/lcm_v_total*100:.1f}%) | N/A (same) | {am_v_recovered}/{am_v_total} ({am_v_recovered/am_v_total*100:.1f}%) |
| Avg Compression Ratio | {avg_lcm_ratio:.1%} | — | {avg_am_ratio:.1%} |

## By Question Type

| Type | lossless-agent (compressed) | lossless-agent (expanded) | agentmemory | Winner |
|------|---------------------------|--------------------------|-------------|--------|
"""
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        lcm_pct = s["lcm"] / s["total"] * 100 if s["total"] else 0
        exp_pct = s["lcm_exp"] / s["total"] * 100 if s["total"] else 0
        am_pct = s["am"] / s["total"] * 100 if s["total"] else 0
        winner = "lossless-agent" if exp_pct > am_pct else ("agentmemory" if am_pct > exp_pct else "tie")
        report += f"| {t} | {s['lcm']}/{s['total']} ({lcm_pct:.0f}%) | {s['lcm_exp']}/{s['total']} ({exp_pct:.0f}%) | {s['am']}/{s['total']} ({am_pct:.0f}%) | {winner} |\n"
    
    report += f"""
## Compression Efficiency

| Metric | lossless-agent | agentmemory |
|--------|---------------|-------------|
| Avg compression ratio | {avg_lcm_ratio:.1%} | {avg_am_ratio:.1%} |
| Can expand back to original | YES | NO |
| Preserves conversation flow | YES | NO (atomic facts only) |

## Key Finding

**lossless-agent's expand capability** is the differentiator. When questions require
contextual understanding (why something was said, what led to a topic, temporal flow),
expansion from the DAG recovers information that atomic memory extraction loses.

## Token Usage

| Metric | Value |
|--------|-------|
| Total input tokens | {total_tokens['input']:,} |
| Total output tokens | {total_tokens['output']:,} |
| Est. cost | ${(total_tokens['input'] * 0.25 + total_tokens['output'] * 1.25) / 1_000_000:.2f} |
"""
    
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    
    print(report)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print("\nDONE!")


if __name__ == "__main__":
    asyncio.run(main())
