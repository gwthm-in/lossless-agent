"""pgvector-backed store for summary and message embeddings.

Used for:
1. Cross-session semantic retrieval: embed summary nodes at compaction time
2. Raw vector retrieval: embed raw messages at ingestion time for hybrid search

Requires: psycopg2-binary + pgvector Postgres extension.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorStore:
    """Store and search summary and message embeddings using pgvector.

    Maintains its own psycopg2 connection (separate from the main
    PostgresDatabase) so it can use the vector extension without
    disrupting the existing connection adapter.

    Schema:
        summary_embeddings(
            summary_id TEXT PK,
            conversation_id INTEGER,
            embedding vector(dim),
            created_at TIMESTAMPTZ
        )
        message_embeddings(
            message_id TEXT PK,
            conversation_id INTEGER,
            embedding vector(msg_dim),
            created_at TIMESTAMPTZ
        )
    """

    def __init__(self, dsn: str, dim: int = 1536, msg_dim: int = 384) -> None:
        import psycopg2  # noqa: F401 — fail fast if not installed

        self._dsn = dsn
        self._dim = dim
        self._msg_dim = msg_dim
        self._conn: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self):
        import psycopg2

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = False
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        # Step 1: try to create the extension.
        # On managed Postgres (RDS, Supabase, etc.) this requires SUPERUSER;
        # the extension is typically pre-installed by the platform admin.
        # We attempt it in an isolated transaction so a failure doesn't
        # poison subsequent DDL.
        cur = self._conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            logger.debug(
                "CREATE EXTENSION vector failed (%s) — assuming extension is "
                "already installed by a superuser",
                exc,
            )
        finally:
            cur.close()

        # Step 2: create the table (requires the extension to be present).
        cur = self._conn.cursor()
        try:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS summary_embeddings (
                    summary_id      TEXT    PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    embedding       vector({self._dim}),
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS summary_embeddings_conv_idx
                ON summary_embeddings(conversation_id)
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

        # Step 3: HNSW index (pgvector >= 0.5.0).  Fall back to IVFFlat on
        # older versions — both support vector_cosine_ops; IVFFlat just
        # requires a pre-built list count.
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS summary_embeddings_ann_idx
                ON summary_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.debug("HNSW index unavailable (pgvector < 0.5?), trying IVFFlat")
            cur2 = self._conn.cursor()
            try:
                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS summary_embeddings_ann_idx
                    ON summary_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
                self._conn.commit()
            except Exception as exc2:
                self._conn.rollback()
                logger.warning(
                    "Could not create ANN index on summary_embeddings: %s — "
                    "search will fall back to exact scan",
                    exc2,
                )
            finally:
                cur2.close()
        finally:
            cur.close()

        # Step 4: message_embeddings table for raw vector retrieval
        cur = self._conn.cursor()
        try:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id      TEXT    PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    embedding       vector({self._msg_dim}),
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS message_embeddings_conv_idx
                ON message_embeddings(conversation_id)
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

        # Step 5: HNSW index on message_embeddings
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS message_embeddings_ann_idx
                ON message_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            logger.debug("HNSW index unavailable for message_embeddings, trying IVFFlat")
            cur2 = self._conn.cursor()
            try:
                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS message_embeddings_ann_idx
                    ON message_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
                self._conn.commit()
            except Exception as exc2:
                self._conn.rollback()
                logger.warning(
                    "Could not create ANN index on message_embeddings: %s — "
                    "search will fall back to exact scan",
                    exc2,
                )
            finally:
                cur2.close()
        finally:
            cur.close()

    @staticmethod
    def _vec_literal(embedding: List[float]) -> str:
        """Format a Python float list as a pgvector literal '[x,y,z,...]'."""
        return "[" + ",".join(str(v) for v in embedding) + "]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        summary_id: str,
        conversation_id: int,
        embedding: List[float],
    ) -> None:
        """Upsert a summary embedding."""
        conn = self._get_conn()
        vec = self._vec_literal(embedding)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO summary_embeddings (summary_id, conversation_id, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (summary_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                """,
                (summary_id, conversation_id, vec),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        exclude_conversation_id: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Return (summary_id, cosine_similarity) for nearest neighbours.

        Excludes embeddings from *exclude_conversation_id* (the current
        session) so cross-session search returns memories from other
        conversations only.
        """
        conn = self._get_conn()
        vec = self._vec_literal(query_embedding)
        cur = conn.cursor()
        try:
            if exclude_conversation_id is not None:
                cur.execute(
                    """
                    SELECT summary_id,
                           1.0 - (embedding <=> %s::vector) AS similarity
                    FROM summary_embeddings
                    WHERE conversation_id != %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec, exclude_conversation_id, vec, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT summary_id,
                           1.0 - (embedding <=> %s::vector) AS similarity
                    FROM summary_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vec, vec, top_k),
                )
            rows = cur.fetchall()
        finally:
            cur.close()
        return [(row[0], float(row[1])) for row in rows]

    # ------------------------------------------------------------------
    # Message embedding API (raw vector retrieval)
    # ------------------------------------------------------------------

    def store_message(
        self,
        message_id: str,
        conversation_id: int,
        embedding: List[float],
    ) -> None:
        """Upsert a message embedding."""
        conn = self._get_conn()
        vec = self._vec_literal(embedding)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO message_embeddings (message_id, conversation_id, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (message_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                """,
                (message_id, conversation_id, vec),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def store_messages_batch(
        self,
        items: List[tuple],
    ) -> None:
        """Batch upsert message embeddings.

        Args:
            items: List of (message_id, conversation_id, embedding) tuples.
        """
        if not items:
            return
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            for message_id, conversation_id, embedding in items:
                vec = self._vec_literal(embedding)
                cur.execute(
                    """
                    INSERT INTO message_embeddings (message_id, conversation_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (message_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding
                    """,
                    (message_id, conversation_id, vec),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def search_messages(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        exclude_conversation_id: Optional[int] = None,
        conversation_ids: Optional[List[int]] = None,
        min_score: float = 0.35,
    ) -> List[Tuple[str, float]]:
        """Return (message_id, cosine_similarity) for nearest neighbours.

        Args:
            query_embedding: Query vector to search against.
            top_k: Maximum results to return.
            exclude_conversation_id: Exclude messages from this conversation.
            conversation_ids: If set, only search within these conversations.
            min_score: Minimum cosine similarity threshold.
        """
        conn = self._get_conn()
        vec = self._vec_literal(query_embedding)
        cur = conn.cursor()
        try:
            where_clauses = []
            params: list = []

            if exclude_conversation_id is not None:
                where_clauses.append("conversation_id != %s")
                params.append(exclude_conversation_id)

            if conversation_ids is not None:
                placeholders = ",".join(["%s"] * len(conversation_ids))
                where_clauses.append(f"conversation_id IN ({placeholders})")
                params.extend(conversation_ids)

            # Push min_score into the query so LIMIT applies after filtering,
            # guaranteeing we return up to top_k *qualifying* results.
            if min_score > 0:
                where_clauses.append("1.0 - (embedding <=> %s::vector) >= %s")
                params.extend([vec, min_score])

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            sql = f"""
                SELECT message_id,
                       1.0 - (embedding <=> %s::vector) AS similarity
                FROM message_embeddings
                {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(sql, [vec] + params + [vec, top_k])
            rows = cur.fetchall()
        finally:
            cur.close()

        return [(row[0], float(row[1])) for row in rows]

    def delete_message(self, message_id: str) -> None:
        """Delete a message embedding."""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM message_embeddings WHERE message_id = %s",
                (message_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def message_embedding_count(self) -> int:
        """Return the number of message embeddings stored."""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM message_embeddings")
            return int(cur.fetchone()[0])
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Summary embedding API (existing)
    # ------------------------------------------------------------------

    def delete(self, summary_id: str) -> None:
        """Delete an embedding by summary_id."""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM summary_embeddings WHERE summary_id = %s",
                (summary_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
