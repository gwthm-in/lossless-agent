"""Shared fixtures for store tests."""

import pytest

from lossless_agent.store.database import Database

# Check if Postgres is available
_pg_available = False
try:
    import psycopg2
    _conn = psycopg2.connect(dbname="lossless_agent_test", host="localhost")
    _conn.close()
    _pg_available = True
except Exception:
    pass


@pytest.fixture
def db():
    """Provide a fresh in-memory SQLite Database for each test."""
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def db_file(tmp_path):
    """Provide a fresh file-backed SQLite Database for each test."""
    path = str(tmp_path / "test.db")
    database = Database(path)
    yield database
    database.close()


def _make_pg_db():
    """Create a fresh PostgresDatabase, cleaning up first."""
    import psycopg2 as _pg2
    from lossless_agent.store.postgres_database import PostgresDatabase

    _conn = _pg2.connect(dbname="lossless_agent_test", host="localhost")
    _conn.autocommit = True
    _cur = _conn.cursor()
    _cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    for (t,) in _cur.fetchall():
        _cur.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
    _conn.close()

    return PostgresDatabase(dsn="dbname=lossless_agent_test host=localhost")


def _teardown_pg_db(database):
    """Drop all tables and close a PostgresDatabase."""
    try:
        database._raw_conn.rollback()
    except Exception:
        pass
    try:
        import psycopg2 as _pg2
        _conn2 = _pg2.connect(dbname="lossless_agent_test", host="localhost")
        _conn2.autocommit = True
        _cur2 = _conn2.cursor()
        _cur2.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        for (t,) in _cur2.fetchall():
            _cur2.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
        _conn2.close()
    except Exception:
        pass
    try:
        database.close()
    except Exception:
        pass


@pytest.fixture
def pg_db():
    """Provide a fresh PostgresDatabase for each test."""
    if not _pg_available:
        pytest.skip("PostgreSQL not available locally")
    database = _make_pg_db()
    yield database
    _teardown_pg_db(database)


# Build the list of backend params
_backend_params = ["sqlite"]
if _pg_available:
    _backend_params.append("postgres")


@pytest.fixture(params=_backend_params, ids=lambda x: x)
def any_db(request):
    """Parametrized fixture providing Database for all available backends.

    Tests using this fixture run once per backend (SQLite always, Postgres if available).
    """
    if request.param == "sqlite":
        database = Database(":memory:")
        yield database
        database.close()
    elif request.param == "postgres":
        database = _make_pg_db()
        yield database
        _teardown_pg_db(database)
