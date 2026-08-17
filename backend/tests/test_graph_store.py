"""Graph-store backend selection and URI validation.

A malformed ``NEO4J_URI`` used to select the Neo4j backend anyway, which then
failed on every call — Graph RAG silently returned nothing while the graph
reported itself as configured. These tests pin the corrected behaviour.
"""

from __future__ import annotations

import pytest

from app.indexing.graph_store import NetworkXGraphStore, validate_neo4j_uri


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://localhost:7687",
        "neo4j://localhost:7687",
        "neo4j+s://b7f9149b.databases.neo4j.io",
        "neo4j+ssc://host",
        "bolt+s://host:7687",
        "  bolt://localhost:7687  ",  # surrounding whitespace is tolerated
    ],
)
def test_valid_neo4j_uris_accepted(uri: str) -> None:
    assert validate_neo4j_uri(uri) is None


def test_empty_uri_reported() -> None:
    assert "empty" in validate_neo4j_uri("").lower()


def test_aura_instance_id_gets_a_targeted_message() -> None:
    """Pasting the Aura instance ID instead of the URI is the common mistake."""
    problem = validate_neo4j_uri("b7f9149b")
    assert problem is not None
    assert "instance ID" in problem
    # The message must contain the exact URI the user should paste instead.
    assert "neo4j+s://b7f9149b.databases.neo4j.io" in problem


@pytest.mark.parametrize("uri", ["localhost:7687", "http://localhost:7474", "just-a-name"])
def test_unsupported_scheme_reported(uri: str) -> None:
    problem = validate_neo4j_uri(uri)
    assert problem is not None
    assert "scheme" in problem.lower()


def test_misconfigured_neo4j_falls_back_to_embedded_store(monkeypatch) -> None:
    """A bad URI must degrade to the embedded store, not select a broken Neo4j."""
    import app.indexing.graph_store as module
    from app.core.config import get_settings

    settings = get_settings()
    original = settings.neo4j_uri
    monkeypatch.setattr(module, "_store", None)
    settings.neo4j_uri = "b7f9149b"
    try:
        store = module.get_graph_store()
        assert isinstance(store, NetworkXGraphStore)
        assert store.config_warning is not None
        assert "instance ID" in store.config_warning
    finally:
        settings.neo4j_uri = original
        module._store = None


@pytest.mark.anyio
async def test_fallback_health_explains_why(monkeypatch, tmp_path) -> None:
    store = NetworkXGraphStore(path=tmp_path / "graph.json")
    store.config_warning = "NEO4J_URI is set to 'b7f9149b', which looks like a Neo4j Aura instance ID"
    health = await store.health()
    assert health["healthy"] is True
    assert health["misconfigured_neo4j"] is True
    assert "instance ID" in health["note"]
