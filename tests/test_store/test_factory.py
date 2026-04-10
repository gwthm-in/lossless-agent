"""Tests for the database factory (SQLite vs Postgres routing)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from lossless_agent.config import LCMConfig
from lossless_agent.store.database import Database
from lossless_agent.store.factory import create_database


class TestCreateDatabaseSQLite:
    """create_database returns a SQLite Database when no DSN is set."""

    def test_returns_database_when_no_dsn(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        config = LCMConfig(db_path=db_path, database_dsn="")
        db = create_database(config)
        assert isinstance(db, Database)
        db.close()


class TestCreateDatabasePostgres:
    """create_database returns PostgresDatabase when DSN is set (mocked)."""

    def test_returns_postgres_database_when_dsn_set(self):
        mock_pg_db = MagicMock()
        mock_pg_db.backend = "postgres"

        with patch(
            "lossless_agent.store.factory.PostgresDatabase",
            create=True,
        ):
            # Patch the import inside the factory function
            mock_module = MagicMock()
            mock_module.PostgresDatabase = MagicMock(return_value=mock_pg_db)

            with patch.dict(
                "sys.modules",
                {"lossless_agent.store.postgres_database": mock_module},
            ):
                config = LCMConfig(database_dsn="postgresql://localhost/test")
                db = create_database(config)
                assert db is mock_pg_db
                mock_module.PostgresDatabase.assert_called_once_with(
                    "postgresql://localhost/test"
                )

    def test_raises_import_error_when_psycopg2_missing(self):
        """create_database raises ImportError with helpful message when DSN set but psycopg2 missing."""
        # Temporarily remove psycopg2 from sys.modules and make import fail
        saved = {}
        for mod_name in list(sys.modules):
            if "psycopg2" in mod_name or "postgres_database" in mod_name:
                saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict(
                "sys.modules",
                {
                    "psycopg2": None,
                    "psycopg2.extensions": None,
                    "lossless_agent.store.postgres_database": None,
                },
            ):
                # Force reimport to hit the ImportError
                config = LCMConfig(database_dsn="postgresql://localhost/test")
                with pytest.raises(ImportError, match="psycopg2"):
                    create_database(config)
        finally:
            sys.modules.update(saved)


class TestDatabaseDSNFromEnv:
    """config.database_dsn is populated from the LCM_DATABASE_DSN env var."""

    def test_dsn_from_env_var(self, monkeypatch):
        monkeypatch.setenv("LCM_DATABASE_DSN", "postgresql://user:pass@host/db")
        config = LCMConfig.from_env()
        assert config.database_dsn == "postgresql://user:pass@host/db"

    def test_dsn_empty_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("LCM_DATABASE_DSN", raising=False)
        config = LCMConfig.from_env()
        assert config.database_dsn == ""
