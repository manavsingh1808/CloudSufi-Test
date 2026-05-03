"""Split per-page text into overlapping chunks suitable for embedding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import PageText


@dataclass
class Chunk:
    doc_name: str
    page_number: int
    chunk_index: int
    text: str

    def to_metadata(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            # Keep a snippet in metadata so we can show citations without a second store.
            "text": self.text,
        }


def chunk_pages(
    pages: List[PageText], chunk_size: int = 900, chunk_overlap: int = 150
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    counter = 0
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_index=counter,
                    text=piece,
                )
            )
            counter += 1
    return chunks
