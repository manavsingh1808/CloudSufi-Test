"""Pinecone serverless wrapper: auto-creates the index and exposes upsert/query."""
from __future__ import annotations

import time
import uuid
from typing import List, Tuple

from pinecone import Pinecone, ServerlessSpec

from .chunker import Chunk


class PineconeStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self._ensure_index(dimension, cloud, region)
        self.index = self.pc.Index(index_name)

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
        # Wait for the index to become ready.
        while not self.pc.describe_index(self.index_name).status["ready"]:
            time.sleep(1)

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
        # Upsert in batches of 100 (Pinecone soft limit).
        for i in range(0, len(payload), 100):
            self.index.upsert(vectors=payload[i : i + 100], namespace=namespace)

    def query(
        self, vector: List[float], top_k: int, namespace: str
    ) -> List[Tuple[float, dict]]:
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [(m["score"], m["metadata"]) for m in res.get("matches", [])]

    def delete_namespace(self, namespace: str) -> None:
        try:
            self.index.delete(delete_all=True, namespace=namespace)
        except Exception:
            # Namespace may not exist yet; ignore.
            pass
