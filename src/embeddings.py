"""Local sentence-transformers embeddings (no API key required)."""
from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Wraps a sentence-transformers model. The first instantiation downloads
    the model weights (~80 MB for `all-MiniLM-L6-v2`) and caches them locally.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vec.tolist()
