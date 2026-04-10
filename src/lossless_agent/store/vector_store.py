"""pgvector-backed store for summary node embeddings.

Used for cross-session semantic retrieval: embed summary nodes at compaction
time, then search across conversations at query time.

Requires: psycopg2-binary + pgvector Postgres extension.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorStore:
    """Store and search summary embeddings using pgvector.

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
    """

    def __init__(self, dsn: str, dim: int = 1536) -> None:
        import psycopg2  # noqa: F401 — fail fast if not installed

        self._dsn = dsn
        self._dim = dim
        self._conn = None

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
        cur = self._conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
            # HNSW index for fast approximate nearest-neighbour search
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS summary_embeddings_hnsw_idx
                ON summary_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
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
