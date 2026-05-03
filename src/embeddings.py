"""Thin wrapper around Voyage AI embeddings."""
from __future__ import annotations

from typing import List

import voyageai


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str = "voyage-3"):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Voyage caps batch size; chunk to be safe.
        batch_size = 64
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = self.client.embed(batch, model=self.model, input_type="document")
            out.extend(res.embeddings)
        return out

    def embed_query(self, text: str) -> List[float]:
        res = self.client.embed([text], model=self.model, input_type="query")
        return res.embeddings[0]
