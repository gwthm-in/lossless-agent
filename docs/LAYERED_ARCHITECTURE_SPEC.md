# Layered Memory Architecture Spec
## lossless-agent + agentmemory Integration

**Goal:** Beat 96.2% on LongMemEval while building a production-useful memory system.

---

## The Core Insight

These two systems solve **different halves** of the agent memory problem:

| Problem | lossless-agent solves | agentmemory solves |
|---------|----------------------|-------------------|
| "How do I keep a 200-turn conversation without losing anything?" | ✅ DAG compression | ❌ Not addressed |
| "How do I find the right fact across 50 past sessions?" | ❌ FTS5 only (41.4%) | ✅ Six-signal retrieval (96.2%) |
| "How do I expand a summary back to full detail?" | ✅ Lossless expand | ❌ Atomic memories only |
| "How do I know if a fact was updated/contradicted?" | ❌ Not addressed | ✅ Contradiction resolution |
| "How do I track entity relationships?" | ❌ Not addressed | ✅ Knowledge graph |

**Together, they form a complete memory system with no gaps.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT (Hermes, Claude, etc.)              │
│                                                             │
│  on_turn_start(question) ──────────────────┐                │
│                                            │                │
│  ┌─────────────────────────────────────────▼──────────────┐ │
│  │              LAYER 3: ORCHESTRATOR                      │ │
│  │                                                        │ │
│  │  1. Classify query type                                │ │
│  │  2. Route to retrieval engine                          │ │
│  │  3. If atomic memory insufficient → expand via DAG     │ │
│  │  4. Assemble final context                             │ │
│  └──────────┬──────────────────────────────┬──────────────┘ │
│             │                              │                │
│  ┌──────────▼──────────┐    ┌──────────────▼─────────────┐ │
│  │  LAYER 2: RETRIEVAL │    │  LAYER 1: LOSSLESS STORE   │ │
│  │  (agentmemory)      │    │  (lossless-agent)          │ │
│  │                     │    │                            │ │
│  │  • Vector index     │    │  • Full message DAG        │ │
│  │  • Knowledge graph  │◄───│  • Hierarchical summaries  │ │
│  │  • BM25 lexical     │    │  • Lossless expand         │ │
│  │  • Temporal scoring │    │  • Session continuity      │ │
│  │  • Cross-encoder    │    │                            │ │
│  │  • Consolidation    │    │                            │ │
│  └─────────────────────┘    └────────────────────────────┘ │
│                                                             │
│             SHARED: SQLite database                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Lossless Store (lossless-agent — what exists today)

### What it does
- Stores every conversation message verbatim (zero information loss)
- Compacts old messages into a DAG of hierarchical summaries
- Provides `lcm_expand` to drill back from any summary to its source messages
- FTS5 full-text search over messages and summaries

### Data model
```
Conversation (session_key)
  └── Messages (seq-ordered, role, content, token_count)
       └── Summaries (DAG: leaf → condensed → root)
            ├── summary_id, kind (leaf/condensed), depth
            ├── content (the summary text)
            ├── source_token_count (what was summarized)
            ├── earliest_at, latest_at (temporal bounds)
            └── children → [more Summaries or Messages]
```

### Key property
Any summary can be expanded back to the original messages. The "lossless" guarantee — compression doesn't destroy information, it just hides it behind a navigable hierarchy.

### Role in the layered system
- **Source of truth** for all conversation content
- **Expansion engine** when atomic memories need more context
- **Session continuity** for in-conversation context management

---

## Layer 2: Retrieval Engine (agentmemory's approach — to be built/integrated)

### What it does
- Extracts **atomic memories** from conversations (facts, preferences, events, relationships)
- Indexes them with **six parallel signals** for retrieval
- Builds a **knowledge graph** of entities and relationships
- **Consolidates** memories over time (dedup, contradiction, decay)

### Data model (atomic memories)
```
MemoryNode
  ├── id, content (atomic fact: "User lives in San Francisco")
  ├── kind (fact | preference | event | relationship | opinion)
  ├── importance (0-1), confidence (0-1)
  ├── embedding (768-dim vector from all-mpnet-base-v2)
  ├── entities (extracted: ["User", "San Francisco"])
  ├── event_time (grounded timestamp if temporal)
  ├── session_id → links back to Layer 1's session_key
  ├── source_message_ids → links back to Layer 1's messages
  ├── superseded_by → if contradicted by newer memory
  └── tier (working | episodic | semantic)

KnowledgeGraph
  ├── Entity nodes (people, places, things)
  └── Relationship edges (entity → relation → entity)
```

### Six retrieval signals
```
score = w_semantic  * cosine(query_emb, memory_emb)     # 0.30
      + w_lexical   * bm25(query, memory)                # 0.12
      + w_activation * memory.activation                  # 0.18
      + w_graph     * spreading_activation(entities)      # 0.18
      + w_importance * memory.importance * confidence      # 0.10
      + w_temporal  * gaussian(event_time, query_time)    # 0.12
```

### Key property
Atomic memories are precise and structured, enabling high-precision retrieval. But they lose conversational context — you know "User prefers async communication" but not the full conversation about why.

### Role in the layered system
- **Find the right memories** from across all sessions
- **Track entity relationships** across conversations
- **Resolve contradictions** when facts change over time
- **Rank by relevance** using multiple signals

---

## Layer 3: Orchestrator (the new integration layer)

This is the glue. It coordinates Layers 1 and 2 to produce the best possible context for any query.

### Ingestion Flow (on every conversation turn)

```
User says something
        │
        ▼
   ┌─────────────┐
   │  Layer 1     │  Store raw message in DAG
   │  (lossless)  │  Trigger compaction if needed
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Layer 2     │  Extract atomic memories from new messages:
   │  (retrieval) │  - Facts, preferences, events, relationships
   │              │  - Generate embeddings
   │              │  - Update knowledge graph
   │              │  - Run consolidation (dedup, contradictions)
   │              │  - Each memory links back to source_message_ids
   └─────────────┘
```

### Retrieval Flow (when agent needs to recall)

```
Query: "What restaurant did I mention last week?"
        │
        ▼
┌───────────────────────┐
│  Step 1: CLASSIFY     │
│  Query type detection │
│  → temporal + factual │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│  Step 2: RETRIEVE (Layer 2)               │
│                                           │
│  Six-signal hybrid search:                │
│  - Vector: embed query → HNSW ANN search  │
│  - BM25: keyword search "restaurant"      │
│  - Graph: entities linked to "restaurant" │
│  - Temporal: Gaussian around "last week"  │
│  - Importance: high-importance memories   │
│  - Activation: recently accessed memories │
│                                           │
│  Cross-encoder reranks top candidates     │
│  → Returns: top 10 atomic memories        │
└───────────┬───────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│  Step 3: EXPAND (Layer 1)                 │
│                                           │
│  For each atomic memory:                  │
│  - memory.source_message_ids → find the   │
│    original messages in the DAG           │
│  - Get surrounding context (the full      │
│    conversation snippet, not just the     │
│    atomic fact)                           │
│  - If a summary covers those messages,    │
│    include the summary too                │
│                                           │
│  This is the UNIQUE VALUE of layering:    │
│  agentmemory alone returns                │
│    "User mentioned La Taqueria"           │
│  With expansion, we return                │
│    "User: I tried La Taqueria in the      │
│     Mission last Tuesday. The al pastor   │
│     burrito was incredible but the wait   │
│     was 45 minutes."                      │
└───────────┬───────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│  Step 4: ASSEMBLE                         │
│                                           │
│  Build final context within token budget: │
│  - Atomic memories (precise facts)        │
│  - Expanded context (surrounding convo)   │
│  - Temporal ordering                      │
│  - Session date labels                    │
│  - Knowledge graph connections            │
│                                           │
│  Per-question-type budgets:               │
│  - multi-session: 7,500 tokens            │
│  - temporal-reasoning: 5,000              │
│  - knowledge-update: 2,500               │
│  - single-session: 1,500-3,500           │
└───────────────────────────────────────────┘
```

---

## The Linking Mechanism (critical detail)

The key to making the layers work together is **bidirectional linking**:

### Forward link: Layer 1 → Layer 2
When Layer 2 extracts an atomic memory from a conversation, it stores:
```python
MemoryNode(
    content="User prefers async communication over meetings",
    source_session_key="session_2024_03_15",  # → Layer 1 session
    source_message_ids=[42, 43, 44],           # → Layer 1 message IDs
    source_summary_id="sum_abc123",            # → Layer 1 summary covering these
)
```

### Reverse link: Layer 2 → Layer 1
When Layer 1 compacts messages into a summary, it can notify Layer 2:
```python
# "These messages were compacted — update your source links"
orchestrator.on_compaction(
    session_key="session_2024_03_15",
    compacted_message_ids=[40, 41, 42, 43, 44, 45],
    summary_id="sum_abc123"
)
```

### Expansion via links
When a retrieved atomic memory needs more context:
```python
async def expand_memory(memory_node):
    """Get full conversation context around an atomic memory."""
    # Option A: Expand from source messages directly
    messages = await layer1.get_messages(
        memory_node.source_session_key,
        memory_node.source_message_ids,
        context_window=3  # 3 messages before and after
    )

    # Option B: Get the summary that covers these messages
    summary = await layer1.get_summary(memory_node.source_summary_id)

    # Option C: Full DAG expansion (most context, most tokens)
    full_context = await layer1.lcm_expand(memory_node.source_summary_id)

    return messages, summary, full_context
```

---

## Why This Beats Pure agentmemory

agentmemory alone scores 96.2% — but it has limitations in production:

1. **Lost conversational context**: Atomic memories strip away the conversation flow. "User prefers Python" is less useful than knowing the full discussion about why they switched from Java.

2. **No lossless guarantee**: Once conversation is processed into atomic memories, the original messages aren't preserved in a navigable structure. You can't "drill down" into a memory.

3. **No in-session compression**: agentmemory doesn't help with managing context within a long running conversation. It's a cross-session system.

4. **Extraction errors compound**: If the LLM misextracts a fact, the original context is gone. With Layer 1, you can always re-extract or verify.

The layered system adds:
- **Verification**: Any atomic memory can be traced back to the exact conversation that produced it
- **Richer context**: Retrieval finds the fact, expansion provides the story
- **In-session management**: Long conversations stay coherent without truncation
- **Recovery**: Extraction errors don't cause permanent information loss

---

## Implementation Plan

### Phase 1: Add Retrieval Layer to lossless-agent
Build the retrieval capabilities directly into lossless-agent (don't import agentmemory as dependency — implement the approach):

1. **Memory extraction pipeline**
   - On each turn, extract atomic memories from new messages
   - Structured output: {content, kind, entities, importance, event_time}
   - Store with links back to source message IDs

2. **Embedding index**
   - sentence-transformers `all-mpnet-base-v2` (768-dim)
   - HNSW index for approximate nearest neighbor
   - Embed all atomic memories at extraction time

3. **BM25/FTS5 over atomic memories**
   - Already have FTS5 — extend to index atomic memories too
   - Dual FTS5: one for raw messages/summaries, one for atomic memories

4. **Knowledge graph**
   - Entity extraction during memory creation
   - NetworkX-like graph (or SQLite-backed)
   - Spreading activation for retrieval

### Phase 2: Cross-encoder Reranking + Temporal
5. **Cross-encoder reranker**
   - `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Apply after initial candidate retrieval

6. **Temporal grounding**
   - Parse relative dates ("last Tuesday") to absolute timestamps
   - Gaussian temporal scoring in retrieval
   - Session date propagation to memories

### Phase 3: Consolidation + Orchestrator
7. **Memory consolidation**
   - Near-duplicate detection (vector similarity > 0.85)
   - Contradiction resolution (newer supersedes older)
   - Importance decay over time

8. **Orchestrator**
   - Query classification (semantic/temporal/relational/factual)
   - Per-type retrieval strategies and token budgets
   - Expand-on-demand via DAG
   - Context assembly

### Phase 4: Benchmark + Iterate
9. Re-run LongMemEval with the layered system
10. Per-type error analysis and targeted fixes
11. Iterate until we beat 96.2%

---

## What We Keep vs What We Add

### Keep from lossless-agent (Layer 1)
- [x] Full message storage (SQLite)
- [x] DAG summarization (compaction engine)
- [x] FTS5 search over messages and summaries
- [x] lcm_expand / lcm_describe / lcm_grep tools
- [x] Session management
- [x] Adapter interface for agent frameworks

### Add (Layer 2 — new)
- [ ] Atomic memory extraction (LLM-based)
- [ ] Embedding generation (sentence-transformers)
- [ ] HNSW vector index
- [ ] Knowledge graph (entities + relationships)
- [ ] Six-signal hybrid scoring
- [ ] Cross-encoder reranking
- [ ] Temporal grounding
- [ ] Memory consolidation
- [ ] Contradiction resolution

### Add (Layer 3 — new)
- [ ] Query classifier
- [ ] Retrieval orchestrator
- [ ] Expand-on-demand bridge
- [ ] Context assembler with per-type budgets
- [ ] Abstention detection

---

## File Structure (proposed)

```
src/lossless_agent/
├── store/              # Layer 1 (existing)
│   ├── message_store.py
│   ├── summary_store.py
│   └── ...
├── engine/             # Layer 1 engine (existing)
│   ├── compaction.py
│   └── ...
├── retrieval/          # Layer 2 (NEW)
│   ├── memory_node.py         # Atomic memory model
│   ├── extraction.py          # LLM-based memory extraction
│   ├── embeddings.py          # Embedding generation + HNSW
│   ├── graph.py               # Knowledge graph
│   ├── scoring.py             # Six-signal hybrid scorer
│   ├── reranker.py            # Cross-encoder reranking
│   ├── temporal.py            # Temporal grounding
│   ├── consolidation.py       # Dedup, contradictions, decay
│   └── store.py               # SQLite storage for memories
├── orchestrator/       # Layer 3 (NEW)
│   ├── classifier.py          # Query type classification
│   ├── retriever.py           # Orchestrates L1 + L2
│   ├── expander.py            # DAG expansion bridge
│   ├── assembler.py           # Context assembly
│   └── abstention.py          # "I don't know" detection
├── tools/              # Existing + new
│   ├── recall.py              # Existing (lcm_grep, etc.)
│   └── memory_recall.py       # New (hybrid recall tool)
└── adapters/           # Existing (updated)
    └── base.py                # Updated to use orchestrator
```

---

## Cost Estimate

agentmemory's full LongMemEval run: 4.3M tokens (~$15-20 with Opus)
Our current run: ~115M tokens (~$29 with Haiku) — because we stuff everything

The layered approach should be CHEAPER than our current approach because:
- Precise retrieval means smaller context (2,500-7,500 vs 120,000 tokens per question)
- Extraction is one-time cost at ingestion
- Embeddings are local (sentence-transformers, free)
- Only the generator LLM call and extraction LLM call cost money

---

## Open Questions

1. **Should we vendor agentmemory or reimplement?**
   agentmemory is MIT, 1 commit, ~60K lines. Clean code. We could:
   - (a) Fork it and integrate Layer 1 on top
   - (b) Reimplement the retrieval approach within lossless-agent
   - (c) Use it as a dependency
   Option (b) gives most control but most work. Option (a) is fastest.

2. **Extraction model choice**
   agentmemory uses the generator (Opus) for extraction. Could use cheaper models (Haiku) since extraction is simpler than reasoning.

3. **Embedding model**
   `all-mpnet-base-v2` is good but old. Consider `bge-large-en-v1.5` or `gte-large` for better quality. Trade-off: speed vs accuracy.

4. **Graph backend**
   NetworkX (in-memory) vs SQLite-backed. For production persistence, SQLite is better. For benchmark speed, NetworkX is fine.

5. **Where does abstention detection live?**
   agentmemory gets 100% on abstention (30/30). This needs a confidence threshold — if no retrieved memory scores above X, say "I don't know."
