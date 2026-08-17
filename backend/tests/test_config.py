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


def test_sync_postgres_dsn_is_upgraded_to_async() -> None:
    settings = Settings(database_url="postgresql://u:p@localhost:5432/ragx")
    assert settings.sqlalchemy_url.startswith("postgresql+asyncpg://")
    assert settings.uses_postgres is True


def test_sqlite_is_the_zero_infrastructure_default() -> None:
    settings = Settings(database_url="", postgres_host="")
    assert settings.sqlalchemy_url.startswith("sqlite+aiosqlite://")
    assert settings.uses_postgres is False


def test_public_snapshot_contains_no_credentials() -> None:
    """The snapshot feeds the Settings page; it must expose state, not secrets."""
    settings = Settings(
        gemini_api_key="SECRET-GEMINI",
        groq_api_key="SECRET-GROQ",
        postgres_password="SECRET-PG",
        neo4j_password="SECRET-NEO",
        qdrant_api_key="SECRET-QDRANT",
        ragx_api_key="SECRET-RAGX",
    )
    rendered = repr(settings.public_snapshot())
    for secret in ("SECRET-GEMINI", "SECRET-GROQ", "SECRET-PG", "SECRET-NEO", "SECRET-QDRANT", "SECRET-RAGX"):
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
