"""Pinecone serverless wrapper: auto-creates the index and exposes upsert/query.

Includes resilience against Streamlit hot-reload + cached-resource issues where
the underlying HTTP client gets closed between runs ("Cannot send a request, as
the client has been closed.").
"""
from __future__ import annotations

import time
import uuid
from typing import List, Tuple

from pinecone import Pinecone, ServerlessSpec

from .chunker import Chunk


_CLIENT_CLOSED_MARKERS = (
    "client has been closed",
    "client is closed",
    "event loop is closed",
)


def _looks_like_closed_client(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _CLIENT_CLOSED_MARKERS)


class PineconeStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self._api_key = api_key
        self.index_name = index_name
        self._dimension = dimension
        self._cloud = cloud
        self._region = region
        self._connect()

    def _connect(self) -> None:
        """(Re)build a fresh Pinecone client + index handle."""
        self.pc = Pinecone(api_key=self._api_key)
        self._ensure_index(self._dimension, self._cloud, self._region)
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self, dimension: int, cloud: str, region: str) -> None:
        existing = {idx["name"] for idx in self.pc.list_indexes()}
        if self.index_name in existing:
            return
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not self.pc.describe_index(self.index_name).status["ready"]:
            time.sleep(1)

    def _call_with_retry(self, fn, *args, **kwargs):
        """Run a Pinecone call; on a 'client closed' error, reconnect once and retry."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if _looks_like_closed_client(exc):
                self._connect()
                return fn(*args, **kwargs)
            raise

    def upsert_chunks(
        self, chunks: List[Chunk], vectors: List[List[float]], namespace: str
    ) -> None:
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        payload = []
        for chunk, vec in zip(chunks, vectors):
            payload.append(
                {
                    "id": f"{chunk.doc_name}-{chunk.page_number}-{chunk.chunk_index}-{uuid.uuid4().hex[:8]}",
                    "values": vec,
                    "metadata": chunk.to_metadata(),
                }
            )
        for i in range(0, len(payload), 100):
            batch = payload[i : i + 100]
            self._call_with_retry(self.index.upsert, vectors=batch, namespace=namespace)

    def query(
        self, vector: List[float], top_k: int, namespace: str
    ) -> List[Tuple[float, dict]]:
        res = self._call_with_retry(
            self.index.query,
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [(m["score"], m["metadata"]) for m in res.get("matches", [])]

    def delete_namespace(self, namespace: str) -> None:
        try:
            self._call_with_retry(self.index.delete, delete_all=True, namespace=namespace)
        except Exception:
            pass
