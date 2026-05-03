"""Centralised configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill in your keys."
        )
    return value


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    voyage_api_key: str
    pinecone_api_key: str

    pinecone_index: str = os.getenv("PINECONE_INDEX", "doc-qa-rag")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Voyage AI's `voyage-3` produces 1024-dim embeddings.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "voyage-3")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))

    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))


def get_settings() -> Settings:
    return Settings(
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        voyage_api_key=_require("VOYAGE_API_KEY"),
        pinecone_api_key=_require("PINECONE_API_KEY"),
    )
