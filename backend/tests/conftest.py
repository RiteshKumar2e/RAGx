"""Test configuration.

Every test runs against an isolated temporary data directory (SQLite database,
embedded Qdrant, BM25 snapshot, graph JSON, object store) so the suite never
touches a developer's real index. No API keys are set, so the tests exercise the
degraded-but-functional path: real retrieval with the deterministic hashing
embedder, and no LLM calls.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

TEMP_ROOT = Path(tempfile.mkdtemp(prefix="ragx-tests-"))

# Configure before any application module is imported.
os.environ.update(
    {
        "ENVIRONMENT": "development",
        "DATABASE_URL": f"sqlite+aiosqlite:///{(TEMP_ROOT / 'test.db').as_posix()}",
        "QDRANT_PATH": str(TEMP_ROOT / "qdrant"),
        "QDRANT_URL": "",
        "NEO4J_URI": "",
        "GRAPH_FALLBACK_PATH": str(TEMP_ROOT / "graph" / "graph.json"),
        "STORAGE_BACKEND": "local",
        "STORAGE_LOCAL_PATH": str(TEMP_ROOT / "objects"),
        "GEMINI_API_KEY": "",
        "GROQ_API_KEY": "",
        "EMBEDDING_PROVIDER": "hashing",
        "EMBEDDING_DIMENSION": "256",
        "EXTRACT_ENTITIES": "true",
        "ENABLE_OCR": "false",
        "LOG_LEVEL": "WARNING",
        "CACHE_ENABLED": "false",
    }
)

from app.core.config import BACKEND_ROOT  # noqa: E402

# BM25 persists to a fixed path derived from BACKEND_ROOT; point it at the
# temporary tree too.
import app.indexing.bm25_index as bm25_module  # noqa: E402

bm25_module.INDEX_PATH = TEMP_ROOT / "bm25" / "index.json"


def pytest_sessionfinish(session, exitstatus) -> None:
    shutil.rmtree(TEMP_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def client():
    """A TestClient with the application lifespan run (schema created, indexes warmed)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


SAMPLE_PAPER = b"""# DefectNet: Lightweight Surface Defect Detection

## Abstract
We propose DefectNet, a lightweight surface-defect detector built on MobileNetV2
with a Feature Pyramid Network neck. DefectNet is evaluated on NEU-DET and
achieves 78.4 mAP while running at 62 FPS on a single T4 GPU.

## Methodology
DefectNet uses MobileNetV2 as its backbone feature extractor. The FPN neck fuses
multi-scale features from three stages. Training used the Adam optimizer with a
learning rate of 0.001 and a batch size of 32 for 120 epochs.

## Results
On the NEU-DET benchmark DefectNet reaches 78.4 mAP, outperforming YOLOv5s which
reaches 74.1 mAP. Inference runs at 62 FPS.

## Limitations
DefectNet degrades on very small defects because the FPN neck downsamples
aggressively. Addressing this would require a higher-resolution feature branch.
"""


@pytest.fixture(scope="module")
def indexed_document(client):
    """Upload and fully process one document; yields its id."""
    import io

    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("defectnet.md", io.BytesIO(SAMPLE_PAPER), "text/markdown")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["uploaded"], payload
    document_id = payload["uploaded"][0]["document_id"]

    # BackgroundTasks run synchronously on TestClient request completion.
    detail = client.get(f"/api/v1/documents/{document_id}").json()
    assert detail["status"] == "ready", detail
    return document_id
