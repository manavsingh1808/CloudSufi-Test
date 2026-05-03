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
    openrouter_api_key: str = ""
    pinecone_api_key: str = ''

    pinecone_index: str = os.getenv("PINECONE_INDEX", "doc-qa-rag")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Local sentence-transformers model. all-MiniLM-L6-v2 -> 384-dim.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    # OpenRouter model slug. Default is a strong free model.
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

    # Optional: shown to OpenRouter for analytics / per-app rate limits.
    site_url: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
    site_title: str = os.getenv("OPENROUTER_SITE_TITLE", "Doc Q&A RAG")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))


def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=_require("OPENROUTER_API_KEY"),
        pinecone_api_key=_require("PINECONE_API_KEY"),
    )
