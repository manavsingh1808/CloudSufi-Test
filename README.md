# 📄 Document Q&A — RAG over your PDFs

A small Retrieval-Augmented Generation (RAG) app that lets you upload **1–3 PDFs** and ask
natural-language questions about them. Answers come back with **inline citations** showing
which document and page each fact came from.

Built with:

| Layer        | Choice                                                       |
| ------------ | ------------------------------------------------------------ |
| LLM          | **OpenRouter** (default `deepseek/deepseek-chat-v3.1:free`)  |
| Embeddings   | **sentence-transformers** `all-MiniLM-L6-v2` (local, 384-dim)|
| Vector store | **ChromaDB** (local, persistent SQLite + DuckDB on disk)     |
| UI           | **Streamlit**                                                |
| PDF parsing  | `pypdf` + LangChain `RecursiveCharacterTextSplitter`         |

> **No external infra required.** Everything except the LLM call runs on your machine.
> Only one API key is needed (OpenRouter — has a free tier).

---

## 🚀 Quick start (single command)

```bash
# 1. Clone and enter the project
git clone <your-repo-url> doc-qa-rag
cd doc-qa-rag

# 2. Create a virtual environment and install deps
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and paste your OpenRouter key

# 4. Run!
streamlit run app.py
```

The app opens at <http://localhost:8501>. Upload up to 3 PDFs, click **Ingest**, then ask away.

> **First-run notes:**
> - The embedding model (~80 MB) downloads into your HuggingFace cache on the first ingest.
> - ChromaDB creates a local folder `./chroma_data/` (SQLite + index files). It's safe to
>   delete that folder anytime to start fresh.

---

## 🔑 What you need

Just **one** API key — has a free tier:

- **OpenRouter** — <https://openrouter.ai/keys> (free models available, no card needed)

Embeddings and the vector store run **locally** — no Pinecone, no embedding API key.

### Picking a different OpenRouter model

Any [OpenRouter model](https://openrouter.ai/models) works. Just change `OPENROUTER_MODEL`
in `.env`. Some good free options:

- `deepseek/deepseek-chat-v3.1:free` (default, strong general reasoning)
- `meta-llama/llama-3.3-70b-instruct:free`
- `google/gemini-2.0-flash-exp:free`

For better quality (paid):

- `anthropic/claude-sonnet-4.5`
- `openai/gpt-4o-mini`

---

## 🧠 How it works (architecture)

```
   ┌──────────────┐    ┌──────────────┐   ┌────────────────────┐   ┌──────────────┐
   │  PDF upload  │───▶│  Chunk text  │──▶│ MiniLM embed (CPU) │──▶│   ChromaDB   │
   └──────────────┘    │ (page-aware) │   │   (384-dim, local) │   │ (local disk) │
                       └──────────────┘   └────────────────────┘   └──────┬───────┘
                                                                          │
   ┌──────────────┐    ┌──────────────┐   ┌────────────────────┐          │
   │  User asks   │───▶│ MiniLM embed │──▶│  Chroma top-K      │◀─────────┘
   │  question    │    │  (query)     │   │  (cosine)          │
   └──────────────┘    └──────────────┘   └─────────┬──────────┘
                                                    │
                                                    ▼
                                       ┌────────────────────────┐
                                       │ OpenRouter LLM         │
                                       │ (DeepSeek free, etc.)  │
                                       │ + system prompt +      │
                                       │ numbered excerpts      │
                                       └─────────┬──────────────┘
                                                 │
                                                 ▼
                                  Answer with inline [1] [2] citations
                                  rendered alongside source snippets.
```

### Pipeline details

1. **PDF parsing** (`src/pdf_loader.py`) — extract text per page so we can cite an exact
   page number. Empty pages are skipped.
2. **Chunking** (`src/chunker.py`) — `RecursiveCharacterTextSplitter` with `chunk_size=900`
   and `overlap=150`. Each chunk keeps `doc_name`, `page_number`, and a snippet in metadata.
3. **Embeddings** (`src/embeddings.py`) — local `sentence-transformers/all-MiniLM-L6-v2`,
   normalized to unit length so cosine similarity behaves well.
   No API key needed; the model is cached after the first run.
4. **Vector store** (`src/vector_store.py`) — `chromadb.PersistentClient` writes vectors,
   metadata, and snippets into `./chroma_data/`. Each browser session gets its own
   **collection** (`doc-qa--session-<uuid>`) so different sessions stay isolated.
5. **Retrieval + generation** (`src/rag.py`) — embed the question, fetch top-K chunks
   from Chroma, build a numbered-context prompt, and call **OpenRouter** through the
   OpenAI SDK (`base_url="https://openrouter.ai/api/v1"`) with a strict
   "only use these excerpts and cite inline" system prompt.
6. **UI** (`app.py`) — Streamlit page for upload → ingest → chat, with an expandable
   **Sources** panel showing the document name, page, similarity score, and snippet
   for every retrieved chunk.

---

## 📁 Project layout

```
doc-qa-rag/
├── app.py                  # Streamlit entrypoint
├── src/
│   ├── config.py           # env-var loading + defaults
│   ├── pdf_loader.py       # PDF → per-page text
│   ├── chunker.py          # text → overlapping chunks
│   ├── embeddings.py       # local sentence-transformers wrapper
│   ├── vector_store.py     # local ChromaDB wrapper
│   └── rag.py              # retrieval + OpenRouter prompting
├── chroma_data/            # auto-created on first run; safe to delete
├── .streamlit/config.toml  # UI theme / server defaults
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚠️ Known limitations

- **Scanned PDFs aren't OCR'd.** `pypdf` only extracts text from PDFs with a real text
  layer. Image-only scans will yield no chunks (the app warns you).
- **Section-level citations are coarse.** I cite by **document + page**, not by
  heading/section. The expandable Sources panel shows the exact retrieved snippet,
  which usually makes the section obvious.
- **MiniLM is small.** `all-MiniLM-L6-v2` is fast and free but less semantically rich
  than larger embedding models. For higher-quality retrieval, swap in
  `BAAI/bge-base-en-v1.5` (~440 MB, 768-dim) by setting `EMBEDDING_MODEL` in `.env`.
  If you change the embedding model, **delete `chroma_data/`** first so the new model's
  vectors don't collide with the old ones.
- **Free OpenRouter models have rate limits** — typically a few requests per minute.
  If you hit a 429, wait a moment and retry, or switch to a paid model.
- **Local store is single-process.** ChromaDB's `PersistentClient` is fine for
  Streamlit's single-process dev server, but isn't designed for multi-process or
  multi-user concurrent writes. For that, point Chroma at its server mode or move to
  Qdrant/Pinecone/etc.
- **Re-uploading the same file re-embeds it.** No content hashing yet.

---

## 🛠️ Improvements I'd make with more time

- **OCR fallback** (e.g. `pytesseract` or `unstructured`) for scanned PDFs.
- **Hybrid search** — combine BM25/keyword with dense retrieval for short factual
  questions where embeddings underperform.
- **Reranking** with a cross-encoder (e.g. `BAAI/bge-reranker-base`) before sending
  the top chunk to the LLM.
- **Per-document hashing** so re-uploading the same file is a no-op instead of a re-embed.
- **Section detection** — parse PDF outline/bookmarks (and fall back to heading
  heuristics) so citations can read "Section 3.2" instead of just "page 7".
- **Streaming responses** from OpenRouter for snappier UX.
- **Eval harness** — a small set of (question, expected-source) pairs and a script that
  measures retrieval recall@k and answer faithfulness.
- **Dockerfile + `docker compose up`** for an even simpler one-command run.

---

## 🧪 Tips for trying it out

- Start with the PDF you trust most and ask it a question whose answer you already know
  — it's the fastest way to sanity-check retrieval quality.
- If an answer looks off, open the **Sources** expander: 9 times out of 10, the wrong
  chunk got retrieved (a chunking/embedding issue), not an LLM hallucination.
- Tweak `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` in `.env` to trade off precision vs.
  context. Smaller chunks + higher `top_k` works better for specific facts; larger
  chunks help for summarisation-style questions.
- To completely reset everything: stop Streamlit, delete the `chroma_data/` folder,
  restart.
