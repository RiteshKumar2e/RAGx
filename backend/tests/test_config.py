"""Configuration resolution and the public settings snapshot."""

from __future__ import annotations

from pathlib import Path

from app.core.config import BACKEND_ROOT, Settings


def test_relative_data_paths_anchor_to_backend_dir() -> None:
    """Data paths must not depend on the working directory.

    The server can be started from the repository root or from ``backend/``,
    and ``.env`` may live in either place. Resolving against the CWD would
    silently create a second, empty data tree.
    """
    settings = Settings(
        qdrant_path="./data/qdrant",
        storage_local_path="data/objects",
        graph_fallback_path="./backend/data/graph/graph.json",
    )
    assert Path(settings.qdrant_path) == (BACKEND_ROOT / "data" / "qdrant").resolve()
    assert Path(settings.storage_local_path) == (BACKEND_ROOT / "data" / "objects").resolve()
    assert Path(settings.graph_fallback_path) == (
        BACKEND_ROOT / "data" / "graph" / "graph.json"
    ).resolve()


def test_absolute_data_paths_are_left_alone() -> None:
    absolute = str(Path.cwd().resolve() / "custom-store")
    assert Settings(qdrant_path=absolute).qdrant_path == absolute


def test_sync_sqlite_dsn_is_upgraded_to_async() -> None:
    settings = Settings(database_url="sqlite:///./local.db")
    assert settings.sqlalchemy_url.startswith("sqlite+aiosqlite://")


def test_async_sqlite_dsn_is_left_alone() -> None:
    url = "sqlite+aiosqlite:///./local.db"
    assert Settings(database_url=url).sqlalchemy_url == url


def test_sqlite_is_the_zero_infrastructure_default() -> None:
    settings = Settings(database_url="")
    assert settings.sqlalchemy_url.startswith("sqlite+aiosqlite://")
    assert settings.uses_turso is False
    assert settings.database_flavour == "sqlite"


# --------------------------------------------------------------- Turso / libSQL
def test_turso_url_uses_the_async_libsql_dialect() -> None:
    settings = Settings(
        database_url="",
        turso_database_url="libsql://ragx-anmol.turso.io",
        turso_auth_token="secret-token",
    )
    url = settings.sqlalchemy_url
    assert url.startswith("sqlite+aiolibsql://ragx-anmol.turso.io")
    assert "secure=true" in url
    assert settings.database_flavour == "turso"
    assert settings.uses_turso is True


def test_turso_accepts_a_bare_hostname() -> None:
    """`turso db show --url` prints libsql://…, but users often paste the host."""
    with_scheme = Settings(database_url="", turso_database_url="libsql://db-org.turso.io", turso_auth_token="t")
    bare = Settings(database_url="", turso_database_url="db-org.turso.io", turso_auth_token="t")
    assert with_scheme.database_flavour == "turso"
    assert with_scheme.sqlalchemy_url == bare.sqlalchemy_url


def test_turso_dsn_in_database_url_is_routed_to_libsql() -> None:
    settings = Settings(database_url="libsql://db-org.turso.io", turso_auth_token="t")
    assert settings.sqlalchemy_url.startswith("sqlite+aiolibsql://db-org.turso.io")
    assert settings.database_flavour == "turso"


def test_turso_auth_token_is_url_encoded() -> None:
    """JWT-style tokens contain characters that must not break the query string."""
    settings = Settings(
        database_url="",
        turso_database_url="db-org.turso.io",
        turso_auth_token="ab+cd/ef=gh&ij",
    )
    url = settings.sqlalchemy_url
    assert "ab%2Bcd%2Fef%3Dgh%26ij" in url
    # The raw token must not appear unescaped — that would split the query string.
    assert "ab+cd/ef=gh&ij" not in url


def test_turso_token_absent_from_public_snapshot() -> None:
    settings = Settings(
        database_url="",
        turso_database_url="libsql://db-org.turso.io",
        turso_auth_token="SECRET-TURSO-TOKEN",
    )
    snapshot = repr(settings.public_snapshot())
    assert "SECRET-TURSO-TOKEN" not in snapshot
    assert snapshot.count("turso") >= 0  # flavour may be reported, token never


def test_explicit_database_url_beats_turso() -> None:
    settings = Settings(
        database_url="sqlite+aiosqlite:///./override.db",
        turso_database_url="libsql://db-org.turso.io",
    )
    assert settings.sqlalchemy_url == "sqlite+aiosqlite:///./override.db"
    assert settings.uses_turso is False


def test_public_snapshot_contains_no_credentials() -> None:
    """The snapshot feeds the Settings page; it must expose state, not secrets."""
    settings = Settings(
        gemini_api_key="SECRET-GEMINI",
        groq_api_key="SECRET-GROQ",
        turso_auth_token="SECRET-TURSO",
        neo4j_password="SECRET-NEO",
        qdrant_api_key="SECRET-QDRANT",
        ragx_api_key="SECRET-RAGX",
    )
    rendered = repr(settings.public_snapshot())
    for secret in ("SECRET-GEMINI", "SECRET-GROQ", "SECRET-TURSO", "SECRET-NEO", "SECRET-QDRANT", "SECRET-RAGX"):
        assert secret not in rendered

    # Configuration *state* is still reported.
    assert settings.public_snapshot()["llm"]["gemini_configured"] is True
    assert settings.public_snapshot()["llm"]["groq_configured"] is True


def test_allowed_extensions_parsed_as_set() -> None:
    settings = Settings(allowed_extensions=".pdf, .TXT ,.csv")
    assert settings.allowed_extension_set == {".pdf", ".txt", ".csv"}


def test_cors_origins_parsed_as_list() -> None:
    settings = Settings(cors_origins="http://a.test, http://b.test ,")
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]
