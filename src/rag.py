"""RAG orchestration: retrieve relevant chunks then ask Claude with citations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from anthropic import Anthropic

from .embeddings import VoyageEmbedder
from .vector_store import PineconeStore


SYSTEM_PROMPT = """You are a careful research assistant answering questions strictly using the provided document excerpts.

Rules:
1. Only use information present in the excerpts. If the answer is not in them, say you don't know.
2. After every factual statement, cite the source inline using the format [#] where # is the excerpt number.
3. Prefer concise, direct answers. Quote short snippets when helpful.
4. Never invent page numbers, document names, or facts.
"""


@dataclass
class Citation:
    index: int
    doc_name: str
    page_number: int
    score: float
    snippet: str


@dataclass
class RAGAnswer:
    answer: str
    citations: List[Citation]


class RAGPipeline:
    def __init__(
        self,
        embedder: VoyageEmbedder,
        store: PineconeStore,
        anthropic_api_key: str,
        claude_model: str,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.store = store
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = claude_model
        self.top_k = top_k

    def ask(self, question: str, namespace: str) -> RAGAnswer:
        query_vec = self.embedder.embed_query(question)
        matches = self.store.query(query_vec, top_k=self.top_k, namespace=namespace)

        if not matches:
            return RAGAnswer(
                answer="I couldn't find anything relevant in the uploaded documents.",
                citations=[],
            )

        citations: List[Citation] = []
        context_blocks: List[str] = []
        for i, (score, md) in enumerate(matches, start=1):
            snippet = md.get("text", "")
            citations.append(
                Citation(
                    index=i,
                    doc_name=md.get("doc_name", "unknown"),
                    page_number=int(md.get("page_number", 0)),
                    score=float(score),
                    snippet=snippet,
                )
            )
            context_blocks.append(
                f"[{i}] (source: {md.get('doc_name')}, page {md.get('page_number')})\n{snippet}"
            )

        context = "\n\n---\n\n".join(context_blocks)
        user_message = (
            f"Excerpts from the user's documents:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the excerpts above and cite sources inline like [1], [2]."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Anthropic returns a list of content blocks; concatenate any text blocks.
        answer_text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

        return RAGAnswer(answer=answer_text, citations=citations)
