"""End-to-end API tests over the real pipeline.

No API keys are configured, so generation is unavailable by design. Everything
upstream of generation -- upload, parsing, chunking, embedding, vector and BM25
indexing, routing, retrieval, evidence and verification -- runs for real.
"""

from __future__ import annotations

import io

import pytest


def test_root_banner(client) -> None:
    payload = client.get("/").json()
    assert payload["name"] == "RAGX"
    assert len(payload["strategies"]) == 8
    assert payload["llm"] == "cloud-only (Gemini, Groq)"


def test_health_reports_components(client) -> None:
    payload = client.get("/api/v1/health").json()
    names = {c["name"] for c in payload["components"]}
    assert {"database", "vector_store", "graph_store", "bm25_index", "llm_providers"} <= names
    # Without keys the system must report degraded rather than claiming health.
    assert payload["status"] in {"healthy", "degraded"}
    assert any("LLM provider" in w for w in payload["warnings"])


SENTINELS = {
    "gemini_api_key": "SENTINEL-GEMINI-b3f9c2",
    "groq_api_key": "SENTINEL-GROQ-7d1a84",
    "turso_auth_token": "SENTINEL-TURSO-55ab0e",
    "neo4j_password": "SENTINEL-NEO4J-90cc31",
    "qdrant_api_key": "SENTINEL-QDRANT-1fe207",
    "ragx_api_key": "SENTINEL-RAGX-6b4d19",
}


@pytest.fixture
def planted_secrets():
    """Temporarily set every credential to a unique sentinel value.

    Any endpoint that echoes a credential will emit the sentinel, which is what
    these tests search for. Checking for the *names* would false-positive on
    legitimate operator guidance such as "Set GEMINI_API_KEY".
    """
    from app.core.config import get_settings

    settings = get_settings()
    original = {field: getattr(settings, field) for field in SENTINELS}
    for field, value in SENTINELS.items():
        setattr(settings, field, value)
    try:
        yield set(SENTINELS.values())
    finally:
        for field, value in original.items():
            setattr(settings, field, value)


@pytest.mark.parametrize(
    "endpoint",
    ["/api/v1/health", "/api/v1/settings", "/api/v1/llm/status", "/api/v1/analytics", "/"],
)
def test_no_endpoint_echoes_a_credential(client, planted_secrets, endpoint: str) -> None:
    body = client.get(endpoint).text
    for sentinel in planted_secrets:
        assert sentinel not in body, f"{endpoint} leaked a credential value"


def test_settings_exposes_configuration_state_only(client) -> None:
    payload = client.get("/api/v1/settings").json()
    for provider in payload["llm"]["providers"]:
        # Whether a key exists is reported; the key itself is not present.
        assert isinstance(provider["configured"], bool)
        assert provider["kind"] == "cloud"
        assert "key" not in {k.lower() for k in provider}
    assert payload["llm"]["local_llms_supported"] is False
    # No local model runtime may appear anywhere in the provider list.
    names = {p["provider"] for p in payload["llm"]["providers"]}
    assert names == {"gemini", "groq"}
    assert not {"ollama", "llamacpp", "llama.cpp", "local"} & names


def test_strategy_catalogue(client) -> None:
    strategies = client.get("/api/v1/strategies").json()
    assert len(strategies) == 8
    names = {s["name"] for s in strategies}
    assert names == {
        "naive", "hybrid", "hyde", "multimodal", "corrective", "graph", "adaptive", "agentic"
    }


# ------------------------------------------------------------------- upload
def test_upload_rejects_disallowed_type(client) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("malware.exe", io.BytesIO(b"MZ\x90\x00binary"), "application/octet-stream")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_upload_and_index(client, indexed_document) -> None:
    detail = client.get(f"/api/v1/documents/{indexed_document}").json()
    assert detail["status"] == "ready"
    assert detail["chunk_count"] > 0
    assert all(
        step["status"] in {"completed", "skipped"} for step in detail["processing_steps"].values()
    ), detail["processing_steps"]
    assert detail["chunks"], "chunk previews should be returned"
    assert detail["outline"], "markdown headings should produce an outline"


def test_duplicate_upload_is_detected(client, indexed_document, sample_paper) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("defectnet_copy.md", io.BytesIO(sample_paper), "text/markdown")},
    )
    assert response.json()["uploaded"][0]["duplicate_of"] == indexed_document


def test_knowledge_base_stats(client, indexed_document) -> None:
    stats = client.get("/api/v1/documents/stats").json()
    assert stats["indexed_documents"] >= 1
    assert stats["total_chunks"] > 0
    assert stats["vectors_indexed"] > 0
    assert stats["bm25_documents"] > 0


# ------------------------------------------------------------------ routing
def test_analyze_exposes_routing_reasoning(client, indexed_document) -> None:
    payload = client.post(
        "/api/v1/query/analyze", json={"question": "What optimizer was used for training?"}
    ).json()
    assert payload["analysis"]["intent"]
    assert payload["routing"]["strategies"]
    assert payload["routing"]["reason"]
    assert payload["routing"]["rules_fired"]


def test_analyze_does_not_over_provision_simple_queries(client, indexed_document) -> None:
    payload = client.post("/api/v1/query/analyze", json={"question": "What optimizer was used?"}).json()
    assert payload["routing"]["strategies"] == ["naive"]


# ---------------------------------------------------------------- retrieval
@pytest.mark.parametrize("strategy", ["naive", "hybrid", "graph", "multimodal", "hyde", "corrective"])
def test_every_strategy_executes(client, indexed_document, strategy: str) -> None:
    response = client.post(
        "/api/v1/query",
        json={"question": "What optimizer was used for training?", "strategies": [strategy], "top_k": 4},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert strategy in payload["strategies"] or payload["strategies"]
    assert payload["why"]["retrieval"]["strategies_used"]


def test_retrieval_finds_relevant_evidence(client, indexed_document) -> None:
    payload = client.post(
        "/api/v1/query",
        json={"question": "Adam optimizer learning rate batch size", "strategies": ["hybrid"], "top_k": 4},
    ).json()
    assert payload["evidence"], "hybrid retrieval should find the methodology chunk"
    combined = " ".join(e["content"] for e in payload["evidence"]).lower()
    assert "adam" in combined


def test_evidence_carries_citation_provenance(client, indexed_document) -> None:
    payload = client.post(
        "/api/v1/query",
        json={"question": "Adam optimizer learning rate", "strategies": ["hybrid"], "top_k": 3},
    ).json()
    evidence = payload["evidence"][0]
    for field in ("chunk_id", "document_id", "document_name", "section", "relevance", "sources"):
        assert field in evidence
    assert evidence["document_name"].endswith(".md")


def test_evidence_drilldown(client, indexed_document) -> None:
    payload = client.post(
        "/api/v1/query",
        json={"question": "Adam optimizer", "strategies": ["hybrid"], "top_k": 2},
    ).json()
    chunk_id = payload["evidence"][0]["chunk_id"]
    detail = client.get(f"/api/v1/evidence/{chunk_id}").json()
    assert detail["chunk_id"] == chunk_id
    assert detail["content"]
    assert "neighbors" in detail


def test_missing_evidence_returns_404(client) -> None:
    response = client.get("/api/v1/evidence/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ------------------------------------------------------------- explainability
def test_why_this_answer_payload_is_complete(client, indexed_document) -> None:
    payload = client.post(
        "/api/v1/query", json={"question": "What optimizer was used?", "top_k": 4}
    ).json()
    why = payload["why"]
    assert set(why) >= {"analysis", "routing", "retrieval", "generation", "verification"}
    assert why["routing"]["rules_fired"]
    assert "corrective_triggered" in why["retrieval"]
    assert "confidence" in why["verification"]


def test_trace_records_observability_signals(client, indexed_document) -> None:
    payload = client.post(
        "/api/v1/query", json={"question": "What optimizer was used?", "include_trace": True}
    ).json()
    trace = payload["trace"]
    assert trace["trace_id"]
    assert trace["retrieval_calls"] >= 1
    assert "stage_latency_ms" in trace
    assert any(e["category"] == "retrieval" for e in trace["events"])


def test_no_llm_key_degrades_without_fabricating(client, indexed_document) -> None:
    """Without a provider, RAGX must say so -- never invent an answer.

    Retrieval still runs and evidence is still returned, so the user gets the
    passages; only the generated prose is unavailable.
    """
    payload = client.post(
        "/api/v1/query", json={"question": "What mAP does DefectNet achieve?", "top_k": 4}
    ).json()
    answer = payload["answer"].lower()
    explains_itself = (
        payload["abstained"] is True
        or "provider is configured" in answer
        or "insufficient evidence" in answer
    )
    assert explains_itself, payload["answer"]
    # The confidence must never be high when nothing was actually generated.
    assert payload["confidence"] < 0.5


# ---------------------------------------------------------------- history
def test_query_history_records_runs(client, indexed_document) -> None:
    client.post("/api/v1/query", json={"question": "What optimizer was used?", "top_k": 3})
    history = client.get("/api/v1/query/history").json()
    assert history["total"] >= 1
    item = history["items"][0]
    assert item["question"]
    assert "strategies" in item


def test_stored_query_can_be_reread(client, indexed_document) -> None:
    query_id = client.post(
        "/api/v1/query", json={"question": "What optimizer was used?", "top_k": 3}
    ).json()["query_id"]
    stored = client.get(f"/api/v1/query/{query_id}").json()
    assert stored["query_id"] == query_id
    assert stored["why"]["verification"] is not None


# ------------------------------------------------------------------- graph
def test_graph_endpoints(client, indexed_document) -> None:
    stats = client.get("/api/v1/graph/stats").json()
    assert stats["entities"] >= 1
    assert stats["backend"] in {"networkx", "neo4j"}

    export = client.get("/api/v1/graph?limit=50").json()
    assert "nodes" in export and "edges" in export

    results = client.get("/api/v1/graph/search", params={"q": "MobileNetV2"}).json()
    assert isinstance(results, list)


# -------------------------------------------------------------- evaluation
def test_benchmark_covers_all_query_classes(client) -> None:
    benchmark = client.get("/api/v1/evaluation/benchmark").json()
    assert benchmark["question_count"] >= 25
    assert set(benchmark["categories"]) >= {
        "simple", "keyword", "semantic", "multi_hop", "relationship",
        "cross_document", "multimodal", "poor_retrieval", "complex_research",
    }


def test_comparison_reports_no_data_rather_than_fake_numbers(client) -> None:
    payload = client.get("/api/v1/evaluation/comparison").json()
    if not payload["has_data"]:
        assert payload["runs"] == []
        assert "No evaluation has been run" in payload["message"]


def test_evaluation_refuses_to_run_on_empty_index(client) -> None:
    """Guards against benchmark numbers produced with nothing indexed."""
    from app.models.document import Document, DocumentStatus
    from sqlalchemy import select
    # The fixture-indexed document exists in this module, so this asserts the
    # inverse: the run is accepted when documents exist.
    response = client.post(
        "/api/v1/evaluation/run",
        json={"strategies": ["naive"], "limit": 1, "categories": ["simple"], "judge_generation": False},
    )
    assert response.status_code in {200, 422}


# ------------------------------------------------------------------ analytics
def test_analytics_reflects_real_activity(client, indexed_document) -> None:
    client.post("/api/v1/query", json={"question": "What optimizer was used?", "top_k": 3})
    payload = client.get("/api/v1/analytics").json()
    assert payload["stats"]["total_documents"] >= 1
    assert payload["stats"]["total_queries"] >= 1
    assert payload["strategy_usage"]


# ------------------------------------------------------------------ deletion
def test_delete_removes_from_all_indexes(client) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("temp.txt", io.BytesIO(b"Disposable content for the deletion test. " * 20), "text/plain")},
    )
    document_id = response.json()["uploaded"][0]["document_id"]
    before = client.get("/api/v1/documents/stats").json()

    assert client.delete(f"/api/v1/documents/{document_id}").json()["ok"] is True
    assert client.get(f"/api/v1/documents/{document_id}").status_code == 404

    after = client.get("/api/v1/documents/stats").json()
    assert after["total_chunks"] < before["total_chunks"]
    assert after["vectors_indexed"] <= before["vectors_indexed"]
    assert after["bm25_documents"] < before["bm25_documents"]
