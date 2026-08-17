"""Database URL resolution and Turso driver degradation.

The driver's availability is forced with a monkeypatch rather than read from the
machine running the tests. Both branches matter and both must stay covered: the
degradation path is exactly what runs on a host where the driver is missing, so
it cannot be left untested simply because this host has it installed.

The rules pinned here:

* driver present -> use Turso
* development, driver missing -> fall back to local SQLite with a loud,
  surfaced warning
* production, driver missing  -> hard failure, because silently writing to a
  different database than the operator configured is a data-integrity problem
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.db import session as session_module
from app.db.session import _resolve_database_url


@pytest.fixture
def turso_settings():
    def _make(environment: str = "development") -> Settings:
        return Settings(
            environment=environment,
            database_url="",
            turso_database_url="libsql://ragx-test.turso.io",
            turso_auth_token="token",
        )

    return _make


@pytest.fixture
def driver(monkeypatch):
    """Force the outcome of the libSQL driver probe."""

    def _set(available: bool) -> None:
        monkeypatch.setattr(
            session_module, "libsql_driver_available", lambda: available
        )

    return _set


def test_sqlite_needs_no_driver_check() -> None:
    url, warning = _resolve_database_url(Settings(database_url="", turso_database_url=""))
    assert url.startswith("sqlite+aiosqlite://")
    assert warning is None


def test_development_falls_back_to_sqlite_with_warning(turso_settings, driver) -> None:
    driver(False)
    url, warning = _resolve_database_url(turso_settings("development"))
    assert url.startswith("sqlite+aiosqlite://"), "must not use the unusable Turso dialect"
    assert warning, "degradation must be reported, never silent"
    assert "Turso" in warning
    assert "will NOT go to Turso" in warning
    # The message must tell the operator how to fix it.
    assert "pip install" in warning
    assert "libsql" in warning


def test_production_refuses_to_switch_databases(turso_settings, driver) -> None:
    driver(False)
    with pytest.raises(RuntimeError) as excinfo:
        _resolve_database_url(turso_settings("production"))
    message = str(excinfo.value)
    assert "production" in message
    assert "will not silently use a different database" in message


def test_turso_used_directly_when_driver_present(turso_settings, driver) -> None:
    driver(True)
    url, warning = _resolve_database_url(turso_settings("development"))
    assert url.startswith("sqlite+aiolibsql://")
    assert warning is None
