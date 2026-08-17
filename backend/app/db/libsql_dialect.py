"""SQLAlchemy async dialect for Turso, built on the ``libsql`` driver.

Registered as ``sqlite+aiolibsql``, which is the URL scheme
``Settings._turso_url()`` produces -- so this is a drop-in replacement for the
dialect of the same name in ``sqlalchemy-libsql``, whose Rust dependency has no
Windows wheel. Because Turso speaks SQLite, everything above this layer (models,
queries, migrations) is identical to the local-file path.

Only four things differ from SQLAlchemy's stock pysqlite dialect:

* the DBAPI is :mod:`app.db.libsql_dbapi` rather than :mod:`sqlite3`;
* connection arguments carry a host and an auth token instead of a filename;
* ``on_connect`` is disabled, because the driver has no ``create_function`` and
  so cannot register pysqlite's Python helper functions;
* ``is_disconnect`` also recognises transport failures, since this connection is
  a network round-trip rather than a local file handle.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from sqlalchemy import pool
from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite
from sqlalchemy.engine import AdaptedConnection
from sqlalchemy.util.concurrency import await_only

from app.db import libsql_dbapi


def _off_loop(fn, *args, **kwargs):
    """Run a blocking driver call in a worker thread, awaited by SQLAlchemy.

    Two things make this necessary rather than optional:

    * ``libsql`` is synchronous, but every call here is a network round-trip to
      Turso. Running it inline would stall the event loop -- and therefore every
      other in-flight request -- for the duration of the round-trip.
    * SQLAlchemy's async engine runs dialect code inside ``greenlet_spawn(...,
      _require_await=True)``, which raises ``AwaitRequired`` unless the
      operation actually suspends. ``await_only`` on a real awaitable is what
      satisfies that contract.
    """
    return await_only(asyncio.to_thread(lambda: fn(*args, **kwargs)))


class AsyncAdapt_libsql_cursor:
    """Presents the synchronous libsql cursor to SQLAlchemy's async engine.

    Result rows are buffered inside the worker thread immediately after
    execution, so ``fetchone``/``fetchmany``/``fetchall`` are pure in-memory
    operations. Fetching lazily instead would cost one thread hop per call --
    the same design SQLAlchemy's own aiosqlite adapter uses.
    """

    server_side = False

    def __init__(self, adapt_connection: AsyncAdapt_libsql_connection) -> None:
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self._cursor = self._connection.cursor()
        self._rows: deque = deque()

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def arraysize(self) -> int:
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        self._cursor.arraysize = value

    def _run_and_buffer(self, method, operation, parameters) -> None:
        def call():
            if parameters is None:
                method(operation)
            else:
                method(operation, parameters)
            # Drain inside the worker thread: one hop instead of one per row.
            return self._cursor.fetchall() if self._cursor.description else None

        rows = _off_loop(call)
        self._rows = deque(rows) if rows else deque()

    def execute(self, operation, parameters=None):
        self._run_and_buffer(self._cursor.execute, operation, parameters)
        return self

    def executemany(self, operation, seq_of_parameters):
        self._run_and_buffer(self._cursor.executemany, operation, seq_of_parameters)
        return self

    def fetchone(self):
        return self._rows.popleft() if self._rows else None

    def fetchmany(self, size: int | None = None):
        size = self.arraysize if size is None else size
        return [self._rows.popleft() for _ in range(min(size, len(self._rows)))]

    def fetchall(self):
        rows = list(self._rows)
        self._rows.clear()
        return rows

    def close(self) -> None:
        self._rows.clear()
        # Closing a cursor is local bookkeeping, so it needs no worker thread.
        self._cursor.close()

    def __iter__(self):
        while self._rows:
            yield self._rows.popleft()

    def setinputsizes(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass


class AsyncAdapt_libsql_connection(AdaptedConnection):
    """Async facade over one synchronous libsql connection."""

    def __init__(self, dbapi: Any, connection: Any) -> None:
        self.dbapi = dbapi
        self._connection = connection

    def cursor(self, server_side: bool = False) -> AsyncAdapt_libsql_cursor:
        return AsyncAdapt_libsql_cursor(self)

    def execute(self, operation, parameters=None):
        return self.cursor().execute(operation, parameters)

    def executemany(self, operation, seq_of_parameters):
        return self.cursor().executemany(operation, seq_of_parameters)

    def commit(self) -> None:
        _off_loop(self._connection.commit)

    def rollback(self) -> None:
        _off_loop(self._connection.rollback)

    def close(self) -> None:
        _off_loop(self._connection.close)

    def terminate(self) -> None:
        # Called when the pool discards a connection it cannot reset. There is
        # no abort primitive, and raising here would mask the original error.
        try:
            self._connection.close()
        except Exception:
            pass

    @property
    def isolation_level(self):
        return self._connection.isolation_level

    @isolation_level.setter
    def isolation_level(self, value) -> None:
        self._connection.isolation_level = value

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction


class _AsyncLibsqlDBAPI:
    """The DBAPI module SQLAlchemy sees: sync shim plus async connections.

    Attribute lookups (exception classes, ``paramstyle``, ``sqlite_version_info``)
    fall through to :mod:`app.db.libsql_dbapi`; only ``connect`` is replaced, so
    the exceptions SQLAlchemy matches on are the same objects the shim raises.
    """

    def __init__(self, sync_dbapi: Any) -> None:
        self._sync_dbapi = sync_dbapi

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sync_dbapi, name)

    def connect(self, *args: Any, **kwargs: Any) -> AsyncAdapt_libsql_connection:
        connection = _off_loop(self._sync_dbapi.connect, *args, **kwargs)
        return AsyncAdapt_libsql_connection(self, connection)


class SQLiteDialect_aiolibsql(SQLiteDialect_pysqlite):
    driver = "aiolibsql"
    supports_statement_cache = True
    is_async = True

    @classmethod
    def import_dbapi(cls):
        return _AsyncLibsqlDBAPI(libsql_dbapi)

    @classmethod
    def get_pool_class(cls, url):
        # Turso is remote, so connections are expensive to establish and worth
        # pooling -- unlike a local SQLite file, where SQLAlchemy defaults to a
        # simpler pool.
        return pool.AsyncAdaptedQueuePool

    def on_connect(self):
        # pysqlite's hook registers Python callables (regexp, floor, ...) via
        # create_function(), which libsql does not implement. Returning None
        # skips it; the affected SQL functions are not used by RAGX.
        return None

    def create_connect_args(self, url) -> tuple[list[Any], dict[str, Any]]:
        query = dict(url.query)

        # The token must be a connect() keyword -- passing it as a URL query
        # parameter makes Turso reject the handshake with 401.
        auth_token = ""
        for key in ("authToken", "auth_token"):
            value = query.pop(key, None)
            if value:
                auth_token = value if isinstance(value, str) else value[0]

        secure = str(query.pop("secure", "true")).lower() not in ("false", "0", "")

        # Calls are dispatched to a thread pool, so a connection is legitimately
        # used from different threads over its lifetime. The pool still hands it
        # to one caller at a time, so this relaxes a check that does not apply
        # rather than removing real protection.
        options: dict[str, Any] = {"check_same_thread": False}
        if auth_token:
            options["auth_token"] = auth_token
        if "timeout" in query:
            options["timeout"] = float(query.pop("timeout"))

        if not url.host:
            # A local libSQL/SQLite file: no host, no token.
            return ([url.database or ":memory:"], options)

        netloc = f"{url.host}:{url.port}" if url.port else url.host
        scheme = "https" if secure else "http"
        target = f"{scheme}://{netloc}"
        if url.database:
            target = f"{target}/{url.database}"

        return ([target], options)

    def is_disconnect(self, e, connection, cursor) -> bool:
        if super().is_disconnect(e, connection, cursor):
            return True
        if not isinstance(e, libsql_dbapi.Error):
            return False
        message = str(e).lower()
        return any(marker in message for marker in libsql_dbapi.DISCONNECT_MARKERS)


dialect = SQLiteDialect_aiolibsql


def register() -> None:
    """Make ``sqlite+aiolibsql://`` resolve to this dialect.

    Registering explicitly (rather than via entry points) means this
    implementation wins even when ``sqlalchemy-libsql`` is also installed --
    that package registers the same name from its ``__init__``, and its module
    imports ``libsql_experimental`` at import time, which is exactly the
    dependency that cannot be built here.
    """
    from sqlalchemy.dialects import registry  # noqa: PLC0415

    registry.register(
        "sqlite.aiolibsql", "app.db.libsql_dialect", "SQLiteDialect_aiolibsql"
    )
    registry.register(
        "sqlite.libsql", "app.db.libsql_dialect", "SQLiteDialect_aiolibsql"
    )


__all__ = ["SQLiteDialect_aiolibsql", "dialect", "register"]
