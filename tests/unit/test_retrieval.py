"""Retrieval ranking, ingestion persistence and scope boundary regressions."""

import json
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from app.config.settings import Settings
from app.retrieval.embeddings import HashEmbeddings
from app.retrieval.service import RetrievalService


@pytest.fixture
def settings(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(Path(__file__).resolve().parents[2] / "sample_data", corpus)
    return Settings(
        _env_file=None,
        api_tokens_json='{"test":{"user_id":"reader","scopes":["public"]}}',
        database_path=tmp_path / "runtime.db",
        corpus_path=corpus,
    )


def test_hash_embeddings_are_deterministic_normalised_and_semantically_lexical():
    embeddings = HashEmbeddings()
    query = embeddings.embed("GPU inference requirements")
    related = embeddings.embed("GPU inference requirements for Atlas")
    unrelated = embeddings.embed("Confidential annual financial planning")
    assert np.array_equal(query, HashEmbeddings().embed("GPU inference requirements"))
    assert np.linalg.norm(query) == pytest.approx(1)
    assert np.dot(query, related) > np.dot(query, unrelated)
    assert not embeddings.embed(" ").any()


async def test_ranking_and_required_metadata(settings):
    service = RetrievalService(settings)
    assert service.ingest() == {"documents": 7, "chunks": 7}
    results = await service.search(
        "Project Atlas GPU inference deployment requirements", ["public"], 3
    )
    assert "atlas-requirements" in {item["metadata"]["document_id"] for item in results}
    assert all(
        set(item["metadata"])
        == {
            "document_id",
            "document_name",
            "chunk_id",
            "source",
            "created_at",
            "security_scope",
        }
        for item in results
    )
    assert all(item["metadata"]["security_scope"] == "public" for item in results)
    assert len(results) == 3
    assert [item["score"] for item in results] == sorted(
        [item["score"] for item in results],
        reverse=True,
    )


async def test_scope_filter_runs_before_top_k_and_no_scopes_deny_all(settings):
    service = RetrievalService(settings)
    service.ingest()
    query = "confidential annual Project Atlas infrastructure budget 4.2 million GBP"
    unrestricted = await service.search(query, ["public", "finance"], 1)
    assert unrestricted[0]["metadata"]["document_id"] == "restricted-finance"
    public = await service.search(query, ["public"], 1)
    assert public
    assert all(item["metadata"]["security_scope"] == "public" for item in public)
    assert "4.2 million GBP" not in json.dumps(public)
    assert await service.search(query, []) == []
    assert await service.search("Atlas rollback approval", ["internal"], 1)


async def test_ingestion_is_stable_persistent_and_cleans_deleted_documents(settings):
    first = RetrievalService(settings)
    first.ingest()
    before = await first.search("Atlas deployment requirements", ["public"])
    assert first.ingest() == {"documents": 7, "chunks": 7}
    reopened = RetrievalService(settings)
    assert await reopened.search("Atlas deployment requirements", ["public"]) == before
    manifest_path = settings.corpus_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["documents"] = [
        item for item in manifest["documents"] if item["document_id"] != "atlas-legacy"
    ]
    manifest_path.write_text(json.dumps(manifest))
    (settings.corpus_path / "atlas-legacy.md").unlink()
    assert reopened.ingest() == {"documents": 6, "chunks": 6}
    after = await reopened.search("Atlas replica requirements", ["public"], 100)
    assert "atlas-legacy" not in {item["metadata"]["document_id"] for item in after}
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0] == 6


def test_chunking_has_bounded_sizes_and_stable_ids(settings):
    settings.chunk_size = 100
    settings.chunk_overlap = 20
    service = RetrievalService(settings)
    result = service.ingest()
    assert result["chunks"] > result["documents"]
    with sqlite3.connect(settings.database_path) as connection:
        chunks = connection.execute("SELECT chunk_id, content FROM retrieval_chunks").fetchall()
    assert len({chunk_id for chunk_id, _ in chunks}) == len(chunks)
    assert all(0 < len(content) <= settings.chunk_size for _, content in chunks)


@pytest.mark.parametrize("failure", ["missing", "unlisted", "escape", "unknown_scope", "duplicate"])
def test_manifest_fails_closed_before_index_mutation(settings, failure):
    service = RetrievalService(settings)
    service.ingest()
    manifest_path = settings.corpus_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if failure == "missing":
        manifest_path.unlink()
    elif failure == "unlisted":
        (settings.corpus_path / "unlisted.txt").write_text("Unlisted restricted content")
    elif failure == "escape":
        manifest["documents"][0]["file"] = "../outside.md"
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "unknown_scope":
        del manifest["documents"][0]["security_scope"]
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest["documents"].append(manifest["documents"][0])
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        service.ingest()
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM retrieval_chunks").fetchone()[0] == 7


async def test_damaged_scope_metadata_is_discarded(settings):
    service = RetrievalService(settings)
    service.ingest()
    with sqlite3.connect(settings.database_path) as connection:
        rows = connection.execute("SELECT chunk_id, metadata_json FROM retrieval_chunks").fetchall()
        for chunk_id, metadata_json in rows:
            metadata = json.loads(metadata_json)
            if metadata["security_scope"] == "public":
                metadata["security_scope"] = "finance"
                connection.execute(
                    "UPDATE retrieval_chunks SET metadata_json = ? WHERE chunk_id = ?",
                    (json.dumps(metadata), chunk_id),
                )
    assert await service.search("Atlas deployment", ["public"]) == []


async def test_qdrant_uses_payload_filter_before_top_k(settings):
    settings.vector_backend = "qdrant"
    client = QdrantClient(":memory:")
    service = RetrievalService(settings, qdrant_client=client)
    service.ingest()
    with patch.object(client, "query_points", wraps=client.query_points) as query:
        results = await service.search("annual Project Atlas budget 4.2 million GBP", ["public"], 1)
        filters = query.call_args.kwargs["query_filter"].must
        scope_filter = next(item for item in filters if item.key == "security_scope")
        assert isinstance(scope_filter.match, models.MatchAny)
        assert scope_filter.match.any == ["public"]
    assert results
    assert "4.2 million GBP" not in json.dumps(results)
    assert results[0]["metadata"]["security_scope"] == "public"
    finance = await service.search("annual budget", ["finance"], 1)
    assert finance[0]["metadata"]["document_id"] == "restricted-finance"
    service.ingest()
    assert client.count("enterprise_documents", exact=True).count == 7
    await service.aclose()


async def test_qdrant_reclassification_and_shortening_remove_old_scope_before_upsert(settings):
    settings.vector_backend = "qdrant"
    settings.chunk_size = 100
    settings.chunk_overlap = 20
    client = QdrantClient(":memory:")
    service = RetrievalService(settings, qdrant_client=client)
    atlas_path = settings.corpus_path / "atlas-requirements.md"
    atlas_path.write_text("Project Atlas historical publicly visible stale content. " * 10)
    service.ingest()
    old = await service.search("Atlas historical stale", ["public"], 100)
    assert sum(item["metadata"]["document_id"] == "atlas-requirements" for item in old) > 1
    # Unrelated corpora share a collection but must survive this reconciliation.
    client.upsert(
        "enterprise_documents",
        points=[
            models.PointStruct(
                id="98765312-abcd-4321-9999-123456789012",
                vector=[0.01] * 384,
                payload={"corpus_id": "other-corpus", "security_scope": "public"},
            )
        ],
    )
    manifest_path = settings.corpus_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["documents"][0]["security_scope"] = "finance"
    manifest_path.write_text(json.dumps(manifest))
    atlas_path.write_text("Project Atlas current finance requirements.")
    original_upsert = client.upsert

    def inspect_gap(*args, **kwargs):
        # This is the actual interval between deletion and replacement publishing.
        # An upsert-before-delete implementation exposes the stale public chunks here.
        assert service._search_qdrant(service.embeddings.embed("Atlas"), ["public"], 100) == []
        assert client.count("enterprise_documents", exact=True).count == 1
        return original_upsert(*args, **kwargs)

    with patch.object(client, "upsert", side_effect=inspect_gap):
        service.ingest()
    public = await service.search("Atlas historical stale", ["public"], 100)
    assert all(item["metadata"]["document_id"] != "atlas-requirements" for item in public)
    finance = await service.search("Atlas current finance requirements", ["finance"], 100)
    atlas = [item for item in finance if item["metadata"]["document_id"] == "atlas-requirements"]
    assert len(atlas) == 1
    assert atlas[0]["metadata"]["chunk_id"] == "atlas-requirements:0000"
    assert atlas[0]["content"] == "Project Atlas current finance requirements."
    await service.aclose()


async def test_qdrant_empty_manifest_clears_corpus(settings):
    settings.vector_backend = "qdrant"
    client = QdrantClient(":memory:")
    service = RetrievalService(settings, qdrant_client=client)
    service.ingest()
    manifest_path = settings.corpus_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["documents"]:
        (settings.corpus_path / entry["file"]).unlink()
    manifest_path.write_text(json.dumps({"documents": []}))
    assert service.ingest() == {"documents": 0, "chunks": 0}
    assert client.count("enterprise_documents", exact=True).count == 0
    assert await service.search("Atlas", ["public", "finance", "internal"]) == []
    await service.aclose()


async def test_qdrant_failed_replacement_does_not_restore_old_permissions(settings):
    settings.vector_backend = "qdrant"
    client = QdrantClient(":memory:")
    service = RetrievalService(settings, qdrant_client=client)
    service.ingest()
    with patch.object(client, "upsert", side_effect=RuntimeError("Unavailable vector store")):
        with pytest.raises(RuntimeError):
            service.ingest()
    assert await service.search("Atlas", ["public", "finance", "internal"]) == []
    await service.aclose()


@pytest.mark.parametrize("broken_metadata", ["{broken", "null", "[]", "{}"])
async def test_sqlite_corrupt_metadata_is_skipped(settings, broken_metadata):
    service = RetrievalService(settings)
    service.ingest()
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "UPDATE retrieval_chunks SET metadata_json = ? WHERE security_scope = ?",
            (broken_metadata, "public"),
        )
    assert await service.search("Atlas", ["public"]) == []
    # Corruption in one scope cannot prevent authorised retrieval from an intact scope.
    assert await service.search("annual budget", ["finance"])


async def test_invalid_top_k_rejected(settings):
    service = RetrievalService(settings)
    for limit in (0, -1, 101, True):
        with pytest.raises(ValueError, match="top_k"):
            await service.search("Atlas", ["public"], limit)
