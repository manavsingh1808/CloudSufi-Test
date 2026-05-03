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
    openrouter_api_key: str

    # Where Chroma persists its SQLite + vectors on disk.
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_data")
    collection_base: str = os.getenv("CHROMA_COLLECTION", "doc-qa")

    # Local sentence-transformers model. all-MiniLM-L6-v2 -> 384-dim.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # OpenRouter model slug. Default is a strong free model.
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

    # Optional: shown to OpenRouter for analytics / per-app rate limits.
    site_url: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
    site_title: str = os.getenv("OPENROUTER_SITE_TITLE", "Doc Q&A RAG")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))


def get_settings() -> Settings:
    return Settings(openrouter_api_key=_require("OPENROUTER_API_KEY"))
