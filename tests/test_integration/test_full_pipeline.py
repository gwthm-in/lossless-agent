"""End-to-end integration tests for the full lossless-agent pipeline.

Tests cover conversation creation, message persistence, leaf and condensed
compaction, DAG structure verification, context assembly, recall tools,
incremental compaction, multi-conversation isolation, and data integrity.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.engine.compaction import CompactionConfig, CompactionEngine
from lossless_agent.engine.assembler import AssemblerConfig, ContextAssembler
from lossless_agent.tools.recall import lcm_grep, lcm_describe, lcm_expand

# ---------------------------------------------------------------------------
# Realistic message content for debugging a Python application
# ---------------------------------------------------------------------------

_USER_MSGS = [
    "I'm getting a TypeError in my Flask app when I try to serialize a datetime object to JSON. The error says 'Object of type datetime is not JSON serializable'. How do I fix this?",
    "I tried using str() on the datetime but now the frontend can't parse the date format. What format should I use?",
    "Now I'm seeing a different error: 'AttributeError: NoneType object has no attribute strftime'. It seems like some of my database records have NULL timestamps.",
    "I added a None check but the API is still returning 500 errors. Let me share the traceback I'm seeing in the logs.",
    "Here's the traceback: File '/app/views.py', line 42, in get_users ... sqlalchemy.exc.OperationalError: database is locked. This is a different issue entirely.",
    "I'm using SQLite with multiple Gunicorn workers. Is that the problem? Should I switch to PostgreSQL?",
    "OK I switched to PostgreSQL but now I'm getting connection pool exhaustion errors. The app creates a new connection for every request.",
    "I refactored to use SQLAlchemy's connection pooling with pool_size=10 and max_overflow=20. But I'm concerned about connection leaks.",
    "Found the leak - there's a code path in the error handler that doesn't close the session. Let me fix that and run the load test again.",
    "Load test results: p50 latency dropped from 800ms to 120ms, p99 from 12s to 450ms. No more connection errors. But I'm seeing memory growth over time.",
    "I profiled with tracemalloc and found that the user session cache is growing unbounded. Each session stores the full user preferences dict.",
    "I replaced the dict cache with an LRU cache limited to 1000 entries. Memory is stable now at around 200MB. But I still need to handle the cache invalidation.",
    "Implemented cache invalidation using PostgreSQL LISTEN/NOTIFY. When a user updates their preferences, a trigger fires and the app evicts the cache entry.",
    "I'm writing tests for the cache invalidation. How do I test the LISTEN/NOTIFY flow in pytest? I need to simulate the notification.",
    "Used pytest-postgresql fixture and a helper that directly sends NOTIFY via a separate connection. Tests pass. Now I need to add CI/CD.",
    "Set up GitHub Actions with a PostgreSQL service container. The test suite runs in about 3 minutes. But the Docker image build is taking 8 minutes.",
    "Optimized the Dockerfile with multi-stage builds and better layer caching. Build time dropped to 2 minutes. Now looking at deploying to AWS.",
    "Should I use ECS Fargate or EKS for this Flask app? It's a simple REST API with about 20 endpoints.",
    "Going with ECS Fargate for simplicity. Set up a Terraform module for the VPC, ALB, and ECS service. But the health check keeps failing.",
    "The health check was hitting the root path which requires auth. Added a /health endpoint that returns 200 without auth. Deployment is working now.",
    "Users are reporting slow response times in the morning. Looking at CloudWatch metrics, CPU spikes to 90% around 9 AM.",
    "Added auto-scaling based on CPU utilization with target 60%. Min 2 tasks, max 10. The scaling policy kicks in within 2 minutes of load increase.",
    "One more issue: the static assets are being served by Flask which is inefficient. Should I use CloudFront or nginx?",
    "Set up CloudFront with S3 origin for static assets. Response time for assets dropped from 200ms to 15ms. The Flask app only handles API calls now.",
    "Final optimization: added Redis caching for the most frequently accessed API endpoints. Cache hit rate is 85% and average response time is now 45ms.",
]

_ASSISTANT_MSGS = [
    "The issue is that Python's json module doesn't know how to serialize datetime objects. You have a few options:\n1. Use a custom JSON encoder\n2. Convert datetimes to ISO 8601 strings\n3. Use flask.jsonify which handles some types\n\nI'd recommend creating a custom encoder:\n```python\nclass DateTimeEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, datetime):\n            return obj.isoformat()\n        return super().default(obj)\n```",
    "Use ISO 8601 format: `datetime.isoformat()` which produces strings like '2024-01-15T10:30:00'. Most JavaScript Date parsers handle this natively. You can also add timezone info with `datetime.isoformat() + 'Z'` for UTC.",
    "You need to handle the None case before calling strftime. A defensive approach:\n```python\ndef format_timestamp(dt):\n    if dt is None:\n        return None\n    return dt.isoformat()\n```\nAlso consider adding a NOT NULL constraint to your database column with a default value.",
    "The 500 error might be coming from a different code path. Check your error logging middleware - sometimes the error handler itself can raise exceptions. Add try/except around the response serialization and log the full traceback with `logging.exception()`.",
    "Yes, 'database is locked' is a classic SQLite issue with concurrent writers. SQLite uses file-level locking so only one writer can proceed at a time. With multiple Gunicorn workers you'll definitely hit this. Options:\n1. Switch to PostgreSQL (recommended for production)\n2. Use WAL mode (`PRAGMA journal_mode=WAL`)\n3. Reduce to a single worker (not ideal)",
    "PostgreSQL is the right choice for a multi-worker setup. For the connection, use SQLAlchemy's `create_engine` with a connection pool. Don't create raw connections per request.",
    "Connection pool exhaustion usually means connections aren't being returned to the pool. Make sure you're using `session.close()` or a context manager. Configure: `create_engine(url, pool_size=10, max_overflow=20, pool_timeout=30, pool_recycle=1800)`",
    "For detecting connection leaks, enable SQLAlchemy's pool event listeners:\n```python\nfrom sqlalchemy import event\n@event.listens_for(engine, 'checkout')\ndef receive_checkout(dbapi_conn, conn_record, conn_proxy):\n    logging.debug('Connection checked out: %s', id(dbapi_conn))\n```\nThis will help you trace which code paths aren't returning connections.",
    "That's a common pattern - error handlers need the same cleanup discipline as happy paths. Use a `finally` block or better yet, use Flask's `@app.teardown_appcontext` to ensure sessions are always cleaned up regardless of how the request ends.",
    "Those are excellent improvements! The memory growth is worth investigating. Use `tracemalloc` to take snapshots:\n```python\nimport tracemalloc\ntracemalloc.start()\n# ... later ...\nsnapshot = tracemalloc.take_snapshot()\ntop_stats = snapshot.statistics('lineno')\n```",
    "An unbounded cache is a classic memory leak pattern. Use `functools.lru_cache` or a library like `cachetools` with TTL support. For user preferences, an LRU cache with max 1000-2000 entries and a 5-minute TTL would be reasonable.",
    "Good choice with LRU cache. For cache invalidation, you have several strategies:\n1. Time-based TTL (simple but stale data)\n2. Event-driven invalidation via DB triggers\n3. Pub/sub with Redis\n\nSince you're already on PostgreSQL, LISTEN/NOTIFY is a great lightweight option.",
    "LISTEN/NOTIFY is elegant for this use case. The trigger should fire on UPDATE of the user_preferences table and send the user_id as payload. Your app maintains a background listener that processes invalidation events.",
    "For testing LISTEN/NOTIFY, use a separate database connection that sends the NOTIFY:\n```python\ndef test_cache_invalidation(db_session, cache):\n    # Populate cache\n    cache.get_preferences(user_id=1)\n    # Simulate DB update trigger\n    db_session.execute(\"NOTIFY user_prefs_changed, '1'\")\n    db_session.commit()\n    time.sleep(0.1)  # Allow listener to process\n    assert 1 not in cache._store\n```",
    "For the CI/CD pipeline with PostgreSQL, add this to your GitHub Actions workflow:\n```yaml\nservices:\n  postgres:\n    image: postgres:15\n    env:\n      POSTGRES_PASSWORD: test\n    ports:\n      - 5432:5432\n```\nThen set your test DATABASE_URL to point to this service.",
    "8 minutes is too long. Use multi-stage builds:\n```dockerfile\nFROM python:3.11-slim AS builder\nCOPY requirements.txt .\nRUN pip install --prefix=/install -r requirements.txt\n\nFROM python:3.11-slim\nCOPY --from=builder /install /usr/local\nCOPY . /app\n```\nAlso add a `.dockerignore` to exclude tests, docs, and .git.",
    "For a simple REST API with 20 endpoints, ECS Fargate is a better fit than EKS. EKS adds Kubernetes complexity you don't need. Fargate is serverless containers - you define the task and AWS handles the infrastructure.",
    "For your Terraform ECS setup, make sure the ALB health check path matches a lightweight endpoint. Common pattern:\n```python\n@app.route('/health')\ndef health():\n    return jsonify(status='ok'), 200\n```\nSet the health check interval to 30s and healthy threshold to 2.",
    "Auto-scaling configuration looks good. For ECS, use target tracking with CPU at 60%:\n```json\n{\n  \"targetTrackingScalingPolicyConfiguration\": {\n    \"targetValue\": 60.0,\n    \"predefinedMetricSpecification\": {\n      \"predefinedMetricType\": \"ECSServiceAverageCPUUtilization\"\n    }\n  }\n}\n```",
    "CloudFront is the better choice since you're already on AWS. Set up an S3 bucket for static files with CloudFront as CDN. Configure cache behaviors: static assets get cached for 24h, API calls pass through to the ALB with no caching.",
    "For the health endpoint, don't put it behind authentication middleware. In Flask:\n```python\n@app.route('/health')\ndef health():\n    return {'status': 'healthy'}, 200\n```\nMake sure your ALB target group health check points to this path.",
    "Excellent setup! For the morning CPU spikes, this is a classic 'thundering herd' problem. Auto-scaling helps but also consider:\n1. Scheduled scaling: pre-scale before 9 AM\n2. Connection warming: keep-alive connections to the database\n3. Cache warming: pre-populate caches before peak hours",
    "That auto-scaling config is solid. Consider also adding scaling based on request count per target. CPU alone can lag behind sudden traffic surges. Use step scaling for faster response to sudden spikes.",
    "Great question. Never serve static files from Flask in production. CloudFront + S3 is ideal for AWS:\n1. Upload assets to S3 during CI/CD\n2. CloudFront caches at edge locations globally\n3. Set Cache-Control headers for long TTLs\n4. Use cache busting with file hashes in filenames",
    "Redis caching is a great final layer. Use a pattern like:\n```python\ndef get_user(user_id):\n    cached = redis.get(f'user:{user_id}')\n    if cached:\n        return json.loads(cached)\n    user = db.query(User).get(user_id)\n    redis.setex(f'user:{user_id}', 300, json.dumps(user.to_dict()))\n    return user.to_dict()\n```\n85% hit rate is very good for a REST API.",
]


def _make_summarize_fn() -> AsyncMock:
    """Return an AsyncMock summarizer that generates plausible summaries."""

    async def _summarize(text: str) -> str:
        # Pull some keywords from the text to make the summary searchable
        words = text.split()
        snippet = " ".join(words[:20]) if len(words) > 20 else " ".join(words)
        return (
            f"Summary: The conversation covered debugging and optimizing "
            f"a Python Flask application. Topics included JSON serialization, "
            f"database connection pooling, caching strategies, and deployment. "
            f"Key snippet: {snippet}"
        )

    mock = AsyncMock(side_effect=_summarize)
    return mock


def _make_db(tmp_path, name: str = "test.db") -> Database:
    return Database(str(tmp_path / name))


def _add_messages(msg_store, conv_id, count=50, token_count=100):
    """Add alternating user/assistant messages with realistic content."""
    msgs = []
    for i in range(count):
        if i % 2 == 0:
            role = "user"
            content = _USER_MSGS[i // 2 % len(_USER_MSGS)]
        else:
            role = "assistant"
            content = _ASSISTANT_MSGS[i // 2 % len(_ASSISTANT_MSGS)]
        msg = msg_store.append(
            conversation_id=conv_id,
            role=role,
            content=content,
            token_count=token_count,
        )
        msgs.append(msg)
    return msgs


# ===================================================================
# Test 1: Full lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_full_lifecycle(tmp_path):
    """Create conversation, add messages, compact leaves + condensed,
    assemble context, and verify the full DAG structure."""
    db = _make_db(tmp_path)
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    summarize_fn = _make_summarize_fn()

    # Use aggressive compaction settings for testing
    config = CompactionConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1000,  # ~10 messages at 100 tokens each
        leaf_min_fanout=4,
        condensed_min_fanout=3,
    )
    engine = CompactionEngine(msg_store, sum_store, summarize_fn, config)

    # 1. Create conversation and add 50 messages
    conv = conv_store.get_or_create("test-session-lifecycle")
    msgs = _add_messages(msg_store, conv.id, count=50, token_count=100)
    assert len(msgs) == 50
    assert msg_store.count(conv.id) == 50

    # 2. Run leaf compaction multiple times to create several leaf summaries
    leaf_summaries = []
    for _ in range(10):
        leaf = await engine.compact_leaf(conv.id)
        if leaf is None:
            break
        leaf_summaries.append(leaf)

    assert len(leaf_summaries) >= 3, (
        f"Expected at least 3 leaf summaries, got {len(leaf_summaries)}"
    )

    # Verify leaf summaries are linked to source messages
    for leaf in leaf_summaries:
        assert leaf.kind == "leaf"
        assert leaf.depth == 0
        source_ids = sum_store.get_source_message_ids(leaf.summary_id)
        assert len(source_ids) >= config.leaf_min_fanout

    # 3. Run condensed compaction
    condensed = await engine.compact_condensed(conv.id, depth=0)
    assert condensed is not None, "Expected condensed summary to be created"
    assert condensed.kind == "condensed"
    assert condensed.depth == 1

    # 4. Verify DAG structure
    child_ids = sum_store.get_child_ids(condensed.summary_id)
    assert len(child_ids) >= config.condensed_min_fanout
    for child_id in child_ids:
        child = sum_store.get_by_id(child_id)
        assert child is not None
        assert child.kind == "leaf"
        assert child.depth == 0

    # 5. Assemble context
    assembler_config = AssemblerConfig(
        max_context_tokens=10000,
        summary_budget_ratio=0.5,
        fresh_tail_count=4,
    )
    assembler = ContextAssembler(msg_store, sum_store, assembler_config)
    assembled = assembler.assemble(conv.id)

    assert len(assembled.messages) > 0, "Expected tail messages"
    assert assembled.total_tokens <= assembler_config.max_context_tokens
    assert assembled.total_tokens > 0

    # Verify context string
    context_str = assembler.format_context(assembled)
    assert len(context_str) > 0

    # All messages still in DB
    assert msg_store.count(conv.id) == 50

    db.close()


# ===================================================================
# Test 2: Recall tools after compaction
# ===================================================================


@pytest.mark.asyncio
async def test_recall_tools_after_compaction(tmp_path):
    """After compaction, verify lcm_grep, lcm_describe, and lcm_expand work."""
    db = _make_db(tmp_path)
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    summarize_fn = _make_summarize_fn()

    config = CompactionConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1000,
        leaf_min_fanout=4,
        condensed_min_fanout=3,
    )
    engine = CompactionEngine(msg_store, sum_store, summarize_fn, config)

    conv = conv_store.get_or_create("test-recall-tools")
    _add_messages(msg_store, conv.id, count=30, token_count=100)

    # Create leaf summaries
    leaves = []
    for _ in range(10):
        leaf = await engine.compact_leaf(conv.id)
        if leaf is None:
            break
        leaves.append(leaf)
    assert len(leaves) >= 3

    # Create condensed
    condensed = await engine.compact_condensed(conv.id, depth=0)
    assert condensed is not None

    # lcm_grep: search messages
    grep_results = lcm_grep(db, "Flask", scope="messages", conversation_id=conv.id)
    assert len(grep_results) > 0
    assert all(r.type == "message" for r in grep_results)

    # lcm_grep: search summaries
    grep_summary = lcm_grep(db, "debugging", scope="summaries", conversation_id=conv.id)
    assert len(grep_summary) > 0
    assert all(r.type == "summary" for r in grep_summary)

    # lcm_describe: leaf summary
    leaf_desc = lcm_describe(db, leaves[0].summary_id)
    assert leaf_desc is not None
    assert leaf_desc.kind == "leaf"
    assert leaf_desc.source_message_count >= config.leaf_min_fanout

    # lcm_describe: condensed summary
    cond_desc = lcm_describe(db, condensed.summary_id)
    assert cond_desc is not None
    assert cond_desc.kind == "condensed"
    assert len(cond_desc.child_ids) >= config.condensed_min_fanout

    # lcm_expand: leaf -> source messages
    leaf_expand = lcm_expand(db, leaves[0].summary_id, is_sub_agent=True)
    assert leaf_expand is not None
    assert leaf_expand.kind == "leaf"
    assert len(leaf_expand.children) >= config.leaf_min_fanout

    # lcm_expand: condensed -> child summaries
    cond_expand = lcm_expand(db, condensed.summary_id, is_sub_agent=True)
    assert cond_expand is not None
    assert cond_expand.kind == "condensed"
    assert len(cond_expand.children) >= config.condensed_min_fanout
    # Children should be Summary objects
    for child in cond_expand.children:
        assert hasattr(child, "summary_id")
        assert child.kind == "leaf"

    db.close()


# ===================================================================
# Test 3: Incremental compaction over time
# ===================================================================


@pytest.mark.asyncio
async def test_incremental_compaction_over_time(tmp_path):
    """Simulate a growing conversation with incremental compaction after
    each batch of messages."""
    db = _make_db(tmp_path)
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    summarize_fn = _make_summarize_fn()

    config = CompactionConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=800,
        leaf_min_fanout=4,
        condensed_min_fanout=3,
        context_threshold=0.3,  # aggressive threshold so compaction triggers
    )
    engine = CompactionEngine(msg_store, sum_store, summarize_fn, config)
    context_limit = 5000  # low limit to force frequent compaction

    conv = conv_store.get_or_create("test-incremental")

    total_added = 0
    all_summaries_over_time = []
    batch_size = 10

    for batch_num in range(5):
        # Add a batch of messages
        for i in range(batch_size):
            idx = total_added
            role = "user" if idx % 2 == 0 else "assistant"
            content_list = _USER_MSGS if role == "user" else _ASSISTANT_MSGS
            content = content_list[(idx // 2) % len(content_list)]
            msg_store.append(
                conversation_id=conv.id,
                role=role,
                content=content,
                token_count=100,
            )
            total_added += 1

        # Run incremental compaction
        created = await engine.run_incremental(conv.id, context_limit)
        all_summaries_over_time.extend(created)

    # Verify no messages are lost
    assert msg_store.count(conv.id) == total_added

    # Verify DAG grew (at least some summaries were created)
    all_summaries = sum_store.get_by_conversation(conv.id)
    assert len(all_summaries) > 0

    # Verify FTS finds content across time periods
    # Early message content
    results_early = lcm_grep(db, "TypeError", conversation_id=conv.id)
    assert len(results_early) > 0

    # Later message content
    results_late = lcm_grep(db, "CloudFront", conversation_id=conv.id)
    assert len(results_late) > 0

    db.close()


# ===================================================================
# Test 4: Multiple conversations isolated
# ===================================================================


@pytest.mark.asyncio
async def test_multiple_conversations_isolated(tmp_path):
    """Two conversations must not leak summaries or search results."""
    db = _make_db(tmp_path)
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    summarize_fn = _make_summarize_fn()

    config = CompactionConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=800,
        leaf_min_fanout=4,
        condensed_min_fanout=3,
    )
    engine = CompactionEngine(msg_store, sum_store, summarize_fn, config)

    # Conversation A: about Flask debugging
    conv_a = conv_store.get_or_create("session-alpha")
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        content = (
            f"Alpha message {i}: discussing Flask route configuration and "
            f"Jinja2 template rendering with alphaspecific keyword."
        )
        msg_store.append(conv_a.id, role, content, token_count=100)

    # Conversation B: about data science
    conv_b = conv_store.get_or_create("session-beta")
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        content = (
            f"Beta message {i}: discussing pandas DataFrame operations and "
            f"matplotlib plotting with betaspecific keyword."
        )
        msg_store.append(conv_b.id, role, content, token_count=100)

    # Compact each independently
    for _ in range(5):
        await engine.compact_leaf(conv_a.id)
        await engine.compact_leaf(conv_b.id)

    # Verify summaries are scoped
    summaries_a = sum_store.get_by_conversation(conv_a.id)
    summaries_b = sum_store.get_by_conversation(conv_b.id)

    for s in summaries_a:
        assert s.conversation_id == conv_a.id
    for s in summaries_b:
        assert s.conversation_id == conv_b.id

    # Verify grep scoped to conversation works
    results_a = lcm_grep(db, "alphaspecific", conversation_id=conv_a.id)
    assert len(results_a) > 0
    for r in results_a:
        assert r.conversation_id == conv_a.id

    results_b = lcm_grep(db, "betaspecific", conversation_id=conv_b.id)
    assert len(results_b) > 0
    for r in results_b:
        assert r.conversation_id == conv_b.id

    # Cross-check: alpha keyword should NOT appear in beta-scoped search
    cross_results = lcm_grep(db, "alphaspecific", conversation_id=conv_b.id)
    assert len(cross_results) == 0

    db.close()


# ===================================================================
# Test 5: Nothing lost
# ===================================================================


@pytest.mark.asyncio
async def test_nothing_lost(tmp_path):
    """After multiple compaction passes, every message is still recoverable
    through the DAG."""
    db = _make_db(tmp_path)
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    summarize_fn = _make_summarize_fn()

    config = CompactionConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=800,
        leaf_min_fanout=4,
        condensed_min_fanout=3,
    )
    engine = CompactionEngine(msg_store, sum_store, summarize_fn, config)

    conv = conv_store.get_or_create("test-nothing-lost")
    original_msgs = _add_messages(msg_store, conv.id, count=100, token_count=80)
    original_ids = {m.id for m in original_msgs}
    assert len(original_ids) == 100

    # Run leaf compaction until exhausted
    leaf_summaries = []
    for _ in range(50):  # upper bound
        leaf = await engine.compact_leaf(conv.id)
        if leaf is None:
            break
        leaf_summaries.append(leaf)

    # Run condensed compaction
    condensed_summaries = []
    depth = 0
    for _ in range(10):
        c = await engine.compact_condensed(conv.id, depth)
        if c is None:
            break
        condensed_summaries.append(c)
        depth += 1

    # Verify: every leaf summary's source messages are recoverable
    recovered_msg_ids = set()
    for leaf in leaf_summaries:
        expand_result = lcm_expand(db, leaf.summary_id, is_sub_agent=True)
        assert expand_result is not None
        assert expand_result.kind == "leaf"
        for child in expand_result.children:
            assert hasattr(child, "id")  # It's a Message
            recovered_msg_ids.add(child.id)
            assert child.id in original_ids

    # Verify: every condensed summary's children are present
    for cond in condensed_summaries:
        expand_result = lcm_expand(db, cond.summary_id, is_sub_agent=True)
        assert expand_result is not None
        assert expand_result.kind == "condensed"
        assert len(expand_result.children) >= config.condensed_min_fanout
        for child in expand_result.children:
            assert hasattr(child, "summary_id")  # It's a Summary

    # The compacted message IDs should be a subset of original
    compacted_ids = set(sum_store.get_compacted_message_ids(conv.id))
    assert compacted_ids.issubset(original_ids)
    assert compacted_ids == recovered_msg_ids

    # All 100 messages still exist in DB
    assert msg_store.count(conv.id) == 100
    all_msgs = msg_store.get_messages(conv.id)
    assert {m.id for m in all_msgs} == original_ids

    db.close()
