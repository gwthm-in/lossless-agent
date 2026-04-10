"""Database factory — returns SQLite or Postgres backend based on config."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from lossless_agent.store.database import Database

if TYPE_CHECKING:
    from lossless_agent.config import LCMConfig
    from lossless_agent.store.postgres_database import PostgresDatabase


def create_database(config: "LCMConfig") -> Union[Database, "PostgresDatabase"]:
    """Return a Database (SQLite) or PostgresDatabase depending on config.

    If ``config.database_dsn`` is set (or ``LCM_DATABASE_DSN`` env var),
    a Postgres connection is returned.  Otherwise a SQLite file-backed
    Database is returned using ``config.resolved_db_path``.

    Raises:
        ImportError: if a DSN is provided but psycopg2 is not installed.
            Install with: pip install 'lossless-agent[postgres]'
    """
    if config.database_dsn:
        try:
            from lossless_agent.store.postgres_database import PostgresDatabase
            return PostgresDatabase(config.database_dsn)
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. "
                "Install with: pip install 'lossless-agent[postgres]'"
            ) from exc
    return Database(config.resolved_db_path)
