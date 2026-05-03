"""Local persistent vector store backed by ChromaDB.

No network, no API keys. Embeddings live on disk under `chroma_dir/`.
Each session gets its own logical "namespace" implemented as a Chroma
collection name suffix.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunker import Chunk


_NAME_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_collection_name(base: str, namespace: str) -> str:
    """Chroma requires 3-63 chars, [A-Za-z0-9._-], starting/ending alnum."""
    raw = f"{base}--{namespace}"
    cleaned = _NAME_SAFE.sub("-", raw)
    cleaned = cleaned.strip("-_.") or "docqa"
    if len(cleaned) < 3:
        cleaned = (cleaned + "xxx")[:3]
    return cleaned[:63]


class LocalVectorStore:
    def __init__(self, persist_dir: str, collection_base: str = "doc-qa"):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_base = collection_base
        self.client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

    def _collection(self, namespace: str):
        # `get_or_create_collection` is idempotent and namespaced.
        # We pre-normalize embeddings, so cosine distance ≈ 1 - cosine similarity.
        return self.client.get_or_create_collection(
            name=_safe_collection_name(self._collection_base, namespace),
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self, chunks: List[Chunk], vectors: List[List[float]], namespace: str
    ) -> None:
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        if not chunks:
            return
        col = self._collection(namespace)
        ids = [
            f"{c.doc_name}-{c.page_number}-{c.chunk_index}-{uuid.uuid4().hex[:8]}"
            for c in chunks
        ]
        metadatas = [c.to_metadata() for c in chunks]
        documents = [c.text for c in chunks]
        col.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)

    def query(
        self, vector: List[float], top_k: int, namespace: str
    ) -> List[Tuple[float, dict]]:
        col = self._collection(namespace)
        if col.count() == 0:
            return []
        res = col.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        documents = res.get("documents", [[]])[0]
        out: List[Tuple[float, dict]] = []
        for md, dist, doc in zip(metadatas, distances, documents):
            md = dict(md or {})
            # Ensure metadata always carries the snippet text (RAG layer expects "text").
            md.setdefault("text", doc or "")
            # Convert cosine distance -> similarity in [0, 1]-ish range.
            similarity = 1.0 - float(dist)
            out.append((similarity, md))
        return out

    def delete_namespace(self, namespace: str) -> None:
        try:
            self.client.delete_collection(
                name=_safe_collection_name(self._collection_base, namespace)
            )
        except Exception:
            pass
