"""The Turso driver adapter.

RAGX drives Turso through its own ``sqlite+aiolibsql`` dialect on top of the
``libsql`` package. Two pieces of that adapter are easy to get wrong and are
pinned here:

* **Error classification.** ``libsql`` raises a bare ``ValueError`` for every
  SQL failure and defines none of the PEP-249 exception classes, so the shim
  reconstructs them from the message. If that mapping drifts, a duplicate insert
  stops being an ``IntegrityError`` and application code that catches it breaks
  silently.
* **Connect arguments.** The auth token must be a ``connect()`` keyword.
  Passing it as a URL query parameter -- which is how the URL is written -- makes
  Turso reject the handshake with 401.

The messages asserted below are the real ones the driver produced against a live
Turso database, not invented strings.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine.url import make_url

from app.db import libsql_dbapi
from app.db.libsql_dialect import SQLiteDialect_aiolibsql, register

# Verbatim driver output, captured from a live Turso instance.
UNIQUE_VIOLATION = (
    'Hrana: `stream error: `Error { message: "SQLite error: UNIQUE constraint '
    'failed: docs.name", code: "SQLITE_CONSTRAINT_UNIQUE" }`'
)
MISSING_TABLE = (
    'Hrana: `stream error: `Error { message: "SQLite error: no such table: '
    'docs", code: "SQLITE_UNKNOWN" }`'
)
BAD_SYNTAX = (
    'Hrana: `stream error: `Error { message: "SQL string could not be parsed: '
    'near ID", code: "SQLITE_UNKNOWN" }`'
)
UNAUTHORIZED = (
    'Hrana: `api error: `status=401 Unauthorized, body={"error":"Unauthorized: '
    'empty JWT token"}`'
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (UNIQUE_VIOLATION, libsql_dbapi.IntegrityError),
        (MISSING_TABLE, libsql_dbapi.OperationalError),
        (BAD_SYNTAX, libsql_dbapi.OperationalError),
        (UNAUTHORIZED, libsql_dbapi.OperationalError),
        ("Cannot operate on a closed database.", libsql_dbapi.ProgrammingError),
    ],
)
def test_driver_errors_map_to_dbapi_classes(message, expected) -> None:
    assert libsql_dbapi._classify(ValueError(message)) is expected


def test_constraint_violation_is_not_mistaken_for_a_network_error() -> None:
    """The transport envelope must not mask the SQL failure inside it.

    Every driver message is wrapped in "Hrana: ...", which is also a marker of
    transport trouble. A constraint violation that got classified as
    OperationalError would additionally look like a dropped connection to
    is_disconnect(), causing the pool to discard a perfectly healthy connection.
    """
    error = libsql_dbapi._classify(ValueError(UNIQUE_VIOLATION))
    assert error is libsql_dbapi.IntegrityError
    assert not issubclass(error, libsql_dbapi.OperationalError)


def test_exception_hierarchy_follows_pep249() -> None:
    assert issubclass(libsql_dbapi.IntegrityError, libsql_dbapi.DatabaseError)
    assert issubclass(libsql_dbapi.OperationalError, libsql_dbapi.DatabaseError)
    assert issubclass(libsql_dbapi.ProgrammingError, libsql_dbapi.DatabaseError)
    assert issubclass(libsql_dbapi.DatabaseError, libsql_dbapi.Error)
    assert issubclass(libsql_dbapi.InterfaceError, libsql_dbapi.Error)


def test_translation_preserves_the_original_message_and_cause() -> None:
    original = ValueError(UNIQUE_VIOLATION)
    translated = libsql_dbapi._translate(original)
    assert isinstance(translated, libsql_dbapi.IntegrityError)
    assert "UNIQUE constraint failed" in str(translated)
    assert translated.__cause__ is original


def test_connect_renames_arguments_for_the_driver(monkeypatch) -> None:
    """The driver spells two arguments with a leading underscore."""
    captured: dict = {}

    class _FakeDriver:
        Error = libsql_dbapi._libsql.Error

        @staticmethod
        def connect(database, **kwargs):
            captured["database"] = database
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(libsql_dbapi, "_libsql", _FakeDriver)
    libsql_dbapi.connect(
        "https://db.turso.io", auth_token="tok", check_same_thread=False, uri=True
    )

    assert captured["database"] == "https://db.turso.io"
    assert captured["auth_token"] == "tok"
    assert captured["_check_same_thread"] is False
    assert captured["_uri"] is True
    assert "check_same_thread" not in captured, "DBAPI spelling must not leak through"
    assert "uri" not in captured


def test_auth_token_moves_from_url_query_to_connect_kwarg() -> None:
    """A token left in the URL makes Turso answer 401, so it must be extracted."""
    dialect = SQLiteDialect_aiolibsql()
    args, options = dialect.create_connect_args(
        make_url("sqlite+aiolibsql://ragx.turso.io?secure=true&authToken=secret-token")
    )

    assert args == ["https://ragx.turso.io"]
    assert "secret-token" not in args[0], "token must never remain in the URL"
    assert options["auth_token"] == "secret-token"
    # Calls are dispatched across a thread pool.
    assert options["check_same_thread"] is False


def test_secure_false_selects_plain_http() -> None:
    dialect = SQLiteDialect_aiolibsql()
    args, _ = dialect.create_connect_args(
        make_url("sqlite+aiolibsql://localhost:8080?secure=false")
    )
    assert args == ["http://localhost:8080"]


def test_transport_failures_count_as_disconnects() -> None:
    dialect = SQLiteDialect_aiolibsql()
    dialect.dbapi = libsql_dbapi

    dropped = libsql_dbapi.OperationalError("Hrana: connection reset by peer")
    assert dialect.is_disconnect(dropped, None, None)

    # A constraint violation is not a connection problem: recycling the
    # connection would be wasted work and would hide a real application bug.
    violation = libsql_dbapi.IntegrityError(UNIQUE_VIOLATION)
    assert not dialect.is_disconnect(violation, None, None)


def test_ragx_dialect_wins_the_registry_name() -> None:
    """`sqlalchemy-libsql` claims the same name but cannot import on Windows."""
    from sqlalchemy.dialects import registry

    register()
    resolved = registry.load("sqlite.aiolibsql")
    assert resolved is SQLiteDialect_aiolibsql
