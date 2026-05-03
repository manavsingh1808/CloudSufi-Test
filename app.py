"""Streamlit UI for the Document Q&A RAG app."""
from __future__ import annotations

import io
import uuid

import streamlit as st

from src.chunker import chunk_pages
from src.config import get_settings
from src.embeddings import VoyageEmbedder
from src.pdf_loader import load_pdf
from src.rag import RAGPipeline
from src.vector_store import PineconeStore


st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def init_backend():
    settings = get_settings()
    embedder = VoyageEmbedder(api_key=settings.voyage_api_key, model=settings.embedding_model)
    store = PineconeStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index,
        dimension=settings.embedding_dim,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    pipeline = RAGPipeline(
        embedder=embedder,
        store=store,
        anthropic_api_key=settings.anthropic_api_key,
        claude_model=settings.claude_model,
        top_k=settings.top_k,
    )
    return settings, pipeline, store


def reset_session():
    """Drop the current Pinecone namespace and clear UI state."""
    if "namespace" in st.session_state:
        try:
            _, _, store = init_backend()
            store.delete_namespace(st.session_state["namespace"])
        except Exception as exc:
            st.warning(f"Could not clear previous index data: {exc}")
    st.session_state.clear()


def main() -> None:
    st.title("📄 Document Q&A")
    st.caption("RAG over your PDFs — powered by Claude, Voyage AI, and Pinecone.")

    try:
        settings, pipeline, _ = init_backend()
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    if "namespace" not in st.session_state:
        # One Pinecone namespace per user session keeps documents isolated.
        st.session_state["namespace"] = f"session-{uuid.uuid4().hex[:12]}"
    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.sidebar:
        st.subheader("Settings")
        st.text(f"Model: {settings.claude_model}")
        st.text(f"Embeddings: {settings.embedding_model}")
        st.text(f"Index: {settings.pinecone_index}")
        st.text(f"Top-K: {settings.top_k}")
        st.divider()
        if st.button("🧹 Clear session & documents", use_container_width=True):
            reset_session()
            st.rerun()

    st.subheader("1. Upload PDFs (1–3)")
    uploaded = st.file_uploader(
        "Drop up to 3 PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        if len(uploaded) > 3:
            st.error("Please upload at most 3 PDFs.")
        else:
            new_files = [f for f in uploaded if f.name not in st.session_state["ingested_docs"]]
            if new_files and st.button(f"Ingest {len(new_files)} new document(s)"):
                with st.spinner("Reading, chunking, embedding, and indexing…"):
                    for f in new_files:
                        pages = load_pdf(io.BytesIO(f.getvalue()), doc_name=f.name)
                        if not pages:
                            st.warning(f"No extractable text in {f.name}; skipping.")
                            continue
                        chunks = chunk_pages(
                            pages,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap,
                        )
                        vectors = pipeline.embedder.embed_documents([c.text for c in chunks])
                        pipeline.store.upsert_chunks(
                            chunks, vectors, namespace=st.session_state["namespace"]
                        )
                        st.session_state["ingested_docs"].append(f.name)
                        st.success(f"Indexed {f.name} — {len(chunks)} chunks across {len(pages)} pages.")

    if st.session_state["ingested_docs"]:
        st.markdown("**Indexed documents:** " + ", ".join(st.session_state["ingested_docs"]))

    st.divider()
    st.subheader("2. Ask a question")

    question = st.text_input("Your question", placeholder="What does the document say about…?")
    ask = st.button("Ask", type="primary", disabled=not (question and st.session_state["ingested_docs"]))

    if ask:
        with st.spinner("Thinking…"):
            result = pipeline.ask(question, namespace=st.session_state["namespace"])
        st.session_state["history"].insert(0, {"q": question, "result": result})

    for item in st.session_state["history"]:
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(item["result"].answer)
        if item["result"].citations:
            with st.expander("Sources"):
                for c in item["result"].citations:
                    st.markdown(
                        f"**[{c.index}] {c.doc_name} — page {c.page_number}** "
                        f"_(similarity: {c.score:.3f})_"
                    )
                    st.caption(c.snippet[:600] + ("…" if len(c.snippet) > 600 else ""))
        st.divider()


if __name__ == "__main__":
    main()
