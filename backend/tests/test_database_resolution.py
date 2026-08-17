"""Database URL resolution and Turso driver degradation.

The libSQL async driver has no Windows wheel, so a Turso-configured machine may
legitimately not be able to install it. The rules pinned here:

* development -> fall back to local SQLite with a loud, surfaced warning
* production  -> hard failure, because silently writing to a different database
                 than the operator configured is a data-integrity problem
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.db.session import _resolve_database_url, libsql_driver_available


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


def test_sqlite_needs_no_driver_check() -> None:
    url, warning = _resolve_database_url(Settings(database_url="", turso_database_url=""))
    assert url.startswith("sqlite+aiosqlite://")
    assert warning is None


@pytest.mark.skipif(
    libsql_driver_available(), reason="libSQL driver installed; degradation path not reachable"
)
def test_development_falls_back_to_sqlite_with_warning(turso_settings) -> None:
    url, warning = _resolve_database_url(turso_settings("development"))
    assert url.startswith("sqlite+aiosqlite://"), "must not use the unusable Turso dialect"
    assert warning, "degradation must be reported, never silent"
    assert "Turso" in warning
    assert "will NOT go to Turso" in warning
    # The message must tell the operator how to fix it.
    assert "sqlalchemy-libsql" in warning


@pytest.mark.skipif(
    libsql_driver_available(), reason="libSQL driver installed; degradation path not reachable"
)
def test_production_refuses_to_switch_databases(turso_settings) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _resolve_database_url(turso_settings("production"))
    message = str(excinfo.value)
    assert "production" in message
    assert "will not silently use a different database" in message


@pytest.mark.skipif(
    not libsql_driver_available(), reason="libSQL driver not installed on this platform"
)
def test_turso_used_directly_when_driver_present(turso_settings) -> None:
    url, warning = _resolve_database_url(turso_settings("development"))
    assert url.startswith("sqlite+aiolibsql://")
    assert warning is None
