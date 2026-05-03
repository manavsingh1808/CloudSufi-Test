"""Extract text from PDF files, preserving page numbers for citations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, List

import pypdf


@dataclass
class PageText:
    doc_name: str
    page_number: int  # 1-indexed
    text: str


def load_pdf(file: BinaryIO, doc_name: str) -> List[PageText]:
    """Read a PDF file-like object and return per-page text blocks."""
    reader = pypdf.PdfReader(file)
    pages: List[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(PageText(doc_name=doc_name, page_number=idx, text=text))
    return pages
