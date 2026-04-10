# ADR-001: Double `to_tsvector` Evaluation in Postgres FTS Queries

**Status:** Accepted  
**Date:** 2026-04-10  
**Context:** Postgres full-text search in `recall.py` and `summary_store.py`

## Decision

Our Postgres FTS queries compute `to_tsvector('english', coalesce(content, ''))` twice per query — once in the `WHERE` clause for filtering and once in the `ORDER BY` clause for `ts_rank` relevance scoring:

```sql
SELECT ...
FROM messages
WHERE to_tsvector('english', coalesce(content, '')) @@ plainto_tsquery('english', $1)
ORDER BY ts_rank(to_tsvector('english', coalesce(content, '')),
                 plainto_tsquery('english', $2)) DESC
LIMIT $3
```

We accept this as the standard Postgres pattern rather than adding a stored `tsvector` column.

## Rationale

**Why the GIN index handles the `WHERE` efficiently:**  
The GIN index (`CREATE INDEX ... USING gin(to_tsvector('english', content))`) is an expression index that matches the exact `to_tsvector(...)` call in the `WHERE` clause. Postgres recognises this and uses the index for filtering — no full table scan.

**Why the `ORDER BY` double-evaluation is acceptable:**  
The `ts_rank()` call in `ORDER BY` operates only on rows that already passed the `WHERE` filter. For typical conversation/summary tables (thousands to low millions of rows), the filtered result set before `LIMIT` is small. Computing `to_tsvector` on those few rows is negligible.

**Why we don't use a stored `tsvector` column:**  
- Adds schema complexity (extra column, trigger to keep it in sync on INSERT/UPDATE)
- Doubles write-time cost for every message and summary insertion
- Requires migration for existing databases
- The performance gain is only meaningful at scales well beyond our expected usage (millions of rows with broad FTS matches)

**This is the documented Postgres pattern:**  
The PostgreSQL docs and community best practices treat expression-index-backed `WHERE` + inline `ts_rank` in `ORDER BY` as the standard approach for full-text search with relevance ranking.

## Consequences

- Queries may recompute `to_tsvector` on the filtered result set — acceptable for our scale
- If we later observe FTS queries becoming slow on very large databases (>10M rows with broad matches), we can add a stored `tsvector` column as an optimisation without changing the query interface
- No additional schema migration burden for users adopting Postgres backend
