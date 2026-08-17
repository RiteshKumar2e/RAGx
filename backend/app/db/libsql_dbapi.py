"""PEP-249 conformance layer over the ``libsql`` driver.

Why this exists
---------------
Turso's own SQLAlchemy package (``sqlalchemy-libsql``) is built on
``libsql-experimental``, which has no Windows wheel -- pip compiles it from Rust
source, which needs a working MSVC toolchain. The maintained successor package,
``libsql``, *does* ship a prebuilt Windows wheel and talks to Turso over HTTP,
but it is not PEP-249 conformant in two ways SQLAlchemy depends on:

1. **Exceptions.** It defines only ``Error`` -- none of the nine standard
   subclasses -- and raises a bare ``ValueError`` for every SQL failure.
   SQLAlchemy uses that hierarchy to tell a constraint violation from a dropped
   connection, so without it a duplicate insert and a network outage are
   indistinguishable, and ``is_disconnect`` cannot recycle a stale pooled
   connection.
2. **connect() signature.** The auth token is a keyword argument, not a URL
   query parameter (passing it in the URL fails with 401), and two arguments are
   named ``_uri`` / ``_check_same_thread`` rather than the DBAPI spellings.

Everything else the dialect touches -- cursors, ``description``, ``rowcount`` on
UPDATE/DELETE, ``lastrowid``, ``executemany``, commit/rollback -- is already
conformant and is forwarded untouched. This module adapts the two gaps above and
nothing more; it is not a reimplementation of the driver.
"""

from __future__ import annotations

from typing import Any

import libsql as _libsql

apilevel = "2.0"
threadsafety = 1
paramstyle = _libsql.paramstyle
sqlite_version_info = _libsql.sqlite_version_info
sqlite_version = ".".join(str(part) for part in sqlite_version_info)

Binary = memoryview


# --- PEP-249 exception hierarchy -------------------------------------------
# The driver supplies none of these, so they are defined here with the standard
# inheritance tree. SQLAlchemy matches on these classes, not on message text.


class Warning(Exception):  # noqa: A001 - name mandated by PEP 249
    pass


class Error(Exception):
    pass


class InterfaceError(Error):
    pass


class DatabaseError(Error):
    pass


class DataError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    pass


class IntegrityError(DatabaseError):
    pass


class InternalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass


#: Message fragments that identify a failure class. Ordered: the first match
#: wins, so the more specific patterns are listed first.
#:
#: The spellings mirror what SQLite itself reports, because the driver embeds
#: the raw SQLite message inside its Hrana envelope. Where SQLite and PEP 249
#: disagree, this follows ``sqlite3`` -- SQLAlchemy's SQLite dialect is written
#: against pysqlite's behaviour, so matching it keeps the dialect's own error
#: handling correct (for example ``no such table`` is an OperationalError there,
#: not a ProgrammingError).
_ERROR_PATTERNS: tuple[tuple[tuple[str, ...], type[Error]], ...] = (
    (
        (
            "constraint failed",
            "sqlite_constraint",
            "datatype mismatch",
        ),
        IntegrityError,
    ),
    (
        (
            "cannot operate on a closed database",
            "closed connection",
            "closed cursor",
        ),
        ProgrammingError,
    ),
    (
        (
            # Transport-level: the request never reached SQLite. These are the
            # ones is_disconnect() needs to recognise so the pool discards the
            # connection instead of reusing a dead one.
            "api error",
            "unauthorized",
            "status=",
            "connection refused",
            "connection reset",
            "connection closed",
            "timed out",
            "timeout",
            "dns error",
            "tls",
            "stream not found",
            "hrana",
        ),
        OperationalError,
    ),
    (
        (
            "no such table",
            "no such column",
            "no such function",
            "no such index",
            "could not be parsed",
            "syntax error",
            "database is locked",
            "readonly",
            "disk i/o",
        ),
        OperationalError,
    ),
)

#: Fragments meaning "this connection is no longer usable". Checked by the
#: dialect's is_disconnect() so a recycled Turso connection is replaced rather
#: than handed back to the next caller.
DISCONNECT_MARKERS: tuple[str, ...] = (
    "connection refused",
    "connection reset",
    "connection closed",
    "cannot operate on a closed database",
    "stream not found",
    "dns error",
    "timed out",
    "unauthorized",
)


def _classify(exc: BaseException) -> type[Error]:
    """Map a driver exception onto the PEP-249 class it should have been."""
    message = str(exc).lower()
    # A SQLite-level complaint is reported inside the driver's transport
    # envelope, so check for the specific SQL failure before the generic
    # transport markers -- otherwise every constraint violation looks like a
    # network error, purely because "hrana" appears in the message.
    if "sqlite error" in message or "constraint failed" in message:
        for fragments, error_class in _ERROR_PATTERNS:
            if error_class is OperationalError and "api error" in fragments:
                continue  # skip the transport bucket for a SQL-level failure
            if any(fragment in message for fragment in fragments):
                return error_class
        return OperationalError

    for fragments, error_class in _ERROR_PATTERNS:
        if any(fragment in message for fragment in fragments):
            return error_class
    return DatabaseError


def _translate(exc: BaseException) -> Error:
    """Re-raise a driver error as its PEP-249 equivalent, keeping the message."""
    error_class = _classify(exc)
    translated = error_class(str(exc))
    translated.__cause__ = exc
    return translated


#: Exceptions worth translating. ValueError is what the driver actually raises
#: for SQL failures; libsql.Error is included for completeness in case a future
#: release starts using it.
_DRIVER_ERRORS = (ValueError, _libsql.Error, RuntimeError)


class Cursor:
    """Thin pass-through cursor that normalises exceptions."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.arraysize = 1

    # -- attributes the driver already implements correctly ----------------
    @property
    def description(self):
        return self._inner.description

    @property
    def rowcount(self) -> int:
        return self._inner.rowcount

    @property
    def lastrowid(self):
        return self._inner.lastrowid

    def execute(self, sql: str, parameters: Any = ()) -> Cursor:
        try:
            self._inner.execute(sql, parameters)
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc
        return self

    def executemany(self, sql: str, seq_of_parameters: Any) -> Cursor:
        try:
            self._inner.executemany(sql, seq_of_parameters)
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc
        return self

    def fetchone(self):
        try:
            return self._inner.fetchone()
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc

    def fetchmany(self, size: int | None = None):
        try:
            return self._inner.fetchmany(self.arraysize if size is None else size)
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc

    def fetchall(self):
        try:
            return self._inner.fetchall()
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc

    def close(self) -> None:
        try:
            self._inner.close()
        except _DRIVER_ERRORS:
            # A cursor that cannot be closed is already unusable; raising here
            # would mask the error that caused the caller to bail out.
            pass

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def setinputsizes(self, sizes: Any) -> None:  # pragma: no cover - no-op per PEP 249
        pass

    def setoutputsize(self, size: Any, column: Any = None) -> None:  # pragma: no cover
        pass


class Connection:
    """Thin pass-through connection that normalises exceptions."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def cursor(self) -> Cursor:
        try:
            return Cursor(self._inner.cursor())
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc

    def execute(self, sql: str, parameters: Any = ()) -> Cursor:
        cursor = self.cursor()
        return cursor.execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any) -> Cursor:
        cursor = self.cursor()
        return cursor.executemany(sql, seq_of_parameters)

    def commit(self) -> None:
        try:
            self._inner.commit()
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc

    def rollback(self) -> None:
        try:
            self._inner.rollback()
        except _DRIVER_ERRORS as exc:
            raise _translate(exc) from exc

    def close(self) -> None:
        try:
            self._inner.close()
        except _DRIVER_ERRORS:
            pass

    @property
    def in_transaction(self) -> bool:
        return bool(getattr(self._inner, "in_transaction", False))

    @property
    def isolation_level(self):
        return getattr(self._inner, "isolation_level", None)

    @isolation_level.setter
    def isolation_level(self, value) -> None:
        # SQLAlchemy sets this to take over transaction control. The driver
        # exposes it read-only, so accept the assignment and let SQLAlchemy
        # drive BEGIN/COMMIT explicitly -- which is what setting it to None
        # asks for anyway.
        try:
            self._inner.isolation_level = value
        except (AttributeError, TypeError):
            pass

    def __getattr__(self, name: str) -> Any:
        # Forward anything not adapted above (e.g. driver-specific helpers)
        # rather than hiding parts of the driver behind this wrapper.
        return getattr(self._inner, name)


def connect(
    database: str,
    *,
    auth_token: str = "",
    timeout: float | None = None,
    check_same_thread: bool | None = None,
    uri: bool | None = None,
    **kwargs: Any,
) -> Connection:
    """Open a connection, translating DBAPI argument names for the driver."""
    driver_kwargs: dict[str, Any] = dict(kwargs)
    if auth_token:
        driver_kwargs["auth_token"] = auth_token
    if timeout is not None:
        driver_kwargs["timeout"] = timeout
    if check_same_thread is not None:
        driver_kwargs["_check_same_thread"] = check_same_thread
    if uri is not None:
        driver_kwargs["_uri"] = uri

    try:
        return Connection(_libsql.connect(database, **driver_kwargs))
    except _DRIVER_ERRORS as exc:
        raise _translate(exc) from exc


__all__ = [
    "apilevel",
    "threadsafety",
    "paramstyle",
    "sqlite_version",
    "sqlite_version_info",
    "Binary",
    "Warning",
    "Error",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
    "DISCONNECT_MARKERS",
    "Connection",
    "Cursor",
    "connect",
]
