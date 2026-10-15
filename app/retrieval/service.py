"""Manifest-controlled corpus ingestion and scope-filtered vector retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from app.config.settings import Settings
from app.retrieval.embeddings import HashEmbeddings

_METADATA_FIELDS = (
    "document_id",
    "document_name",
    "chunk_id",
    "source",
    "created_at",
    "security_scope",
)
_COLLECTION = "enterprise_documents"


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split bounded character windows, preferring whitespace at their end."""
    text = text.replace("\r\n", "\n").strip()
    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            split = text.rfind(" ", start + size // 2, end)
            if split > start + overlap:
                end = split
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result


class RetrievalService:
    """Ingest a trusted manifest, then search only vectors in authorised scopes.

    SQLite is a persistent local baseline. Qdrant executes its scope filter within the
    vector query; it must never retrieve unrestricted top-k and filter afterwards.
    """

    def __init__(self, settings: Settings, *, qdrant_client: QdrantClient | None = None):
        self.settings = settings
        self.embeddings = HashEmbeddings()
        self.corpus_path = settings.corpus_path.resolve()
        self.corpus_id = hashlib.sha256(str(self.corpus_path).encode()).hexdigest()
        self._qdrant = qdrant_client
        if settings.vector_backend == "qdrant" and self._qdrant is None:
            self._qdrant = QdrantClient(url=settings.qdrant_url, timeout=30)
        if settings.vector_backend == "sqlite":
            settings.database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS retrieval_chunks (
                        corpus_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        security_scope TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_version TEXT NOT NULL,
                        PRIMARY KEY (corpus_id, chunk_id)
                    )
                """)
                connection.execute("""
                    CREATE INDEX IF NOT EXISTS retrieval_scope
                    ON retrieval_chunks (corpus_id, security_scope)
                """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.settings.database_path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _manifest_documents(self) -> list[tuple[Path, dict[str, str]]]:
        manifest_path = self.corpus_path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("A valid corpus manifest.json is required for ingestion") from exc
        if not isinstance(manifest, dict) or set(manifest) != {"documents"}:
            raise ValueError("Corpus manifest must contain a documents array")
        entries = manifest["documents"]
        if not isinstance(entries, list):
            raise ValueError("Corpus manifest documents must be an array")
        documents = []
        paths: set[Path] = set()
        document_ids: set[str] = set()
        required = {"file", "document_id", "document_name", "created_at", "security_scope"}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != required:
                raise ValueError("Every manifest document must declare its identity and scope")
            if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
                raise ValueError("Manifest document values must be nonblank strings")
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Manifest document path must stay inside the corpus")
            path = (self.corpus_path / relative).resolve()
            if not path.is_relative_to(self.corpus_path) or path.suffix.lower() not in {
                ".txt",
                ".md",
            }:
                raise ValueError("Manifest document path must identify a corpus .md or .txt file")
            if not path.is_file():
                raise ValueError("A document declared in the manifest is missing")
            if path in paths or entry["document_id"] in document_ids:
                raise ValueError("Corpus manifest contains duplicate files or document IDs")
            try:
                timestamp = datetime.fromisoformat(entry["created_at"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timezone required")
            except ValueError as exc:
                raise ValueError(
                    "Manifest created_at must be an ISO timestamp with a timezone"
                ) from exc
            paths.add(path)
            document_ids.add(entry["document_id"])
            documents.append(
                (
                    path,
                    {key: entry[key] for key in required if key != "file"}
                    | {"source": relative.as_posix()},
                )
            )
        discovered = {
            path.resolve()
            for path in self.corpus_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        }
        if discovered != paths:
            raise ValueError("Every corpus .md and .txt file must be authorised by the manifest")
        return documents

    def ingest(self) -> dict[str, int]:
        """Validate the complete corpus before changing its index, removing stale chunks."""
        documents = self._manifest_documents()
        chunks = []
        for path, metadata in documents:
            text = path.read_text(encoding="utf-8")
            for index, content in enumerate(
                _chunks(text, self.settings.chunk_size, self.settings.chunk_overlap)
            ):
                chunk_metadata = metadata | {"chunk_id": f"{metadata['document_id']}:{index:04d}"}
                chunks.append((content, chunk_metadata, self.embeddings.embed(content)))
        if self.settings.vector_backend == "sqlite":
            with self._connect() as connection:
                # A transaction makes replacement, changed scopes and deletions atomic.
                connection.execute(
                    "DELETE FROM retrieval_chunks WHERE corpus_id = ?", (self.corpus_id,)
                )
                connection.executemany(
                    """
                    INSERT INTO retrieval_chunks VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            self.corpus_id,
                            metadata["chunk_id"],
                            metadata["security_scope"],
                            content,
                            json.dumps(metadata, sort_keys=True),
                            vector.tobytes(),
                            self.embeddings.version,
                        )
                        for content, metadata, vector in chunks
                    ],
                )
        else:
            self._ingest_qdrant(chunks)
        return {"documents": len(documents), "chunks": len(chunks)}

    def _ingest_qdrant(self, chunks: list[tuple[str, dict[str, str], np.ndarray]]) -> None:
        client = self._qdrant
        if not client.collection_exists(_COLLECTION):
            client.create_collection(
                _COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.embeddings.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{self.corpus_id}:{metadata['chunk_id']}")),
                vector=vector.tolist(),
                payload=metadata
                | {
                    "content": content,
                    "corpus_id": self.corpus_id,
                    "embedding_version": self.embeddings.version,
                },
            )
            for content, metadata, vector in chunks
        ]
        corpus_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="corpus_id",
                    match=models.MatchValue(value=self.corpus_id),
                )
            ]
        )
        # Qdrant reconciliation is not transactional: prefer an availability gap to
        # keeping old broader-scope/deleted chunks while replacement points arrive.
        # Complete deletion before publishing replacements; a failed upsert leaves
        # the corpus unavailable rather than exposing its previous permissions.
        client.delete(_COLLECTION, points_selector=corpus_filter, wait=True)
        if points:
            client.upsert(_COLLECTION, points=points, wait=True)

    async def search(
        self,
        query: str,
        scopes: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        limit = self.settings.retrieval_top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("top_k must be between 1 and 100")
        allowed = sorted({scope for scope in scopes if isinstance(scope, str) and scope.strip()})
        if not allowed or not query.strip():
            return []
        vector = self.embeddings.embed(query)
        if not np.any(vector):
            return []
        if self.settings.vector_backend == "sqlite":
            results = await asyncio.to_thread(self._search_sqlite, vector, allowed, limit)
        else:
            results = await asyncio.to_thread(self._search_qdrant, vector, allowed, limit)
        # Defence in depth against damaged rows or a misbehaving external vector store.
        return [
            result
            for result in results
            if (
                result["metadata"].get("security_scope") in allowed
                and all(isinstance(result["metadata"].get(key), str) for key in _METADATA_FIELDS)
                and isinstance(result["content"], str)
                and math.isfinite(result["score"])
            )
        ]

    def _search_sqlite(
        self,
        vector: np.ndarray,
        scopes: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in scopes)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT content, metadata_json, embedding FROM retrieval_chunks
                WHERE corpus_id = ? AND security_scope IN ({placeholders})
                AND embedding_version = ?
            """,
                [self.corpus_id, *scopes, self.embeddings.version],
            ).fetchall()
        results = []
        for content, metadata_json, embedding in rows:
            try:
                metadata = json.loads(metadata_json)
                stored = np.frombuffer(embedding, dtype=np.float32)
            except (ValueError, TypeError):
                # A damaged vector or JSON row cannot become trusted evidence.
                continue
            if not isinstance(metadata, dict) or metadata.get("security_scope") not in scopes:
                continue
            if not all(isinstance(metadata.get(key), str) for key in _METADATA_FIELDS):
                continue
            if stored.shape != vector.shape:
                continue
            score = float(np.dot(vector, stored))
            if score > 0:
                results.append({"content": content, "metadata": metadata, "score": score})
        results.sort(key=lambda item: (-item["score"], item["metadata"]["chunk_id"]))
        return results[:limit]

    def _search_qdrant(
        self,
        vector: np.ndarray,
        scopes: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        response = self._qdrant.query_points(
            collection_name=_COLLECTION,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
            score_threshold=0.000001,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="security_scope", match=models.MatchAny(any=scopes)),
                    models.FieldCondition(
                        key="corpus_id", match=models.MatchValue(value=self.corpus_id)
                    ),
                    models.FieldCondition(
                        key="embedding_version",
                        match=models.MatchValue(
                            value=self.embeddings.version,
                        ),
                    ),
                ]
            ),
        )
        return [
            {
                "content": (point.payload or {}).get("content"),
                "metadata": {key: (point.payload or {}).get(key) for key in _METADATA_FIELDS},
                "score": float(point.score),
            }
            for point in response.points
        ]

    async def aclose(self) -> None:
        if self._qdrant is not None:
            await asyncio.to_thread(self._qdrant.close)

    async def health(self) -> None:
        """Probe the configured index without returning document content."""
        if self._qdrant is not None:
            await asyncio.to_thread(self._qdrant.get_collection, _COLLECTION)
        else:

            def check():
                with self._connect() as connection:
                    connection.execute("SELECT 1 FROM retrieval_chunks LIMIT 1").fetchone()

            await asyncio.to_thread(check)
