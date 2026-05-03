# 📄 Document Q&A — RAG over your PDFs

A small Retrieval-Augmented Generation (RAG) app that lets you upload **1–3 PDFs** and ask
natural-language questions about them. Answers come back with **inline citations** showing
which document and page each fact came from.

Built with:

| Layer        | Choice                                  |
| ------------ | --------------------------------------- |
| LLM          | **Anthropic Claude** (`claude-sonnet-4-5` by default) |
| Embeddings   | **Voyage AI** (`voyage-3`, 1024-dim)    |
| Vector store | **Pinecone** (serverless, auto-created) |
| UI           | **Streamlit**                           |
| PDF parsing  | `pypdf` + LangChain `RecursiveCharacterTextSplitter` |

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

# 3. Add your API keys
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and paste your keys

# 4. Run!
streamlit run app.py
```

The app opens at <http://localhost:8501>. Upload up to 3 PDFs, click **Ingest**, then ask away.

---

## 🔑 What you need

You'll need three free-tier-friendly API keys:

1. **Anthropic** — <https://console.anthropic.com/>
2. **Voyage AI** — <https://dash.voyageai.com/> (Anthropic's recommended embeddings partner; 50M free tokens)
3. **Pinecone** — <https://app.pinecone.io/> (free serverless tier is plenty)

Drop them into `.env` (see `.env.example`). The Pinecone index is **created automatically**
on first run — no manual setup needed.

---

## 🧠 How it works (architecture)

```
   ┌──────────────┐    ┌──────────────┐   ┌────────────────┐   ┌──────────────┐
   │  PDF upload  │───▶│  Chunk text  │──▶│  Voyage embed  │──▶│  Pinecone    │
   └──────────────┘    │ (page-aware) │   │  (1024-dim)    │   │  (upsert)    │
                       └──────────────┘   └────────────────┘   └──────┬───────┘
                                                                      │
   ┌──────────────┐    ┌──────────────┐   ┌────────────────┐          │
   │  User asks   │───▶│  Voyage embed│──▶│   Pinecone     │◀─────────┘
   │  question    │    │  (query)     │   │   top-K query  │
   └──────────────┘    └──────────────┘   └───────┬────────┘
                                                  │
                                                  ▼
                                       ┌────────────────────┐
                                       │  Claude Sonnet 4.5 │
                                       │  + system prompt + │
                                       │  numbered excerpts │
                                       └─────────┬──────────┘
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
3. **Embeddings** (`src/embeddings.py`) — Voyage `voyage-3`, batched in groups of 64.
   Anthropic doesn't ship its own embeddings model and officially recommends Voyage.
4. **Vector store** (`src/vector_store.py`) — Pinecone serverless, cosine similarity.
   Each browser session gets its own **namespace** (`session-<uuid>`) so different users
   don't see each other's documents.
5. **Retrieval + generation** (`src/rag.py`) — embed the question, fetch top-K chunks
   from Pinecone, build a numbered-context prompt, and ask Claude with a strict
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
│   ├── embeddings.py       # Voyage AI client wrapper
│   ├── vector_store.py     # Pinecone wrapper (auto-create index)
│   └── rag.py              # retrieval + Claude prompting
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
  heading/section, because reliable section detection across arbitrary PDFs is its own
  project. The expandable Sources panel shows the exact snippet retrieved, which usually
  makes the section obvious.
- **Pinecone serverless cold start.** First query after a long idle period can take a
  couple of seconds while the index spins up.
- **No persistence between sessions.** Each browser session uses its own namespace and
  the **Clear** button wipes it. Re-uploading the same PDF will re-embed it (extra cost).
- **No rerun-safe deduping.** If you click Ingest twice on the same file inside one
  session, you'll get duplicate vectors. The UI guards against this for *new* uploads
  but doesn't dedupe across full app restarts.
- **Token/cost guardrails are minimal.** `max_tokens=1024` for the answer and `top_k=5`
  for retrieval. Big PDFs will burn embedding tokens on first ingest.

---

## 🛠️ Improvements I'd make with more time

- **OCR fallback** (e.g. `pytesseract` or `unstructured`) for scanned PDFs.
- **Hybrid search** — combine BM25/keyword with dense retrieval for short factual
  questions where embeddings underperform.
- **Reranking** with `voyage-rerank-2` or Cohere Rerank to push the best chunk to the top
  before sending to Claude.
- **Per-document hashing** so re-uploading the same file is a no-op instead of a re-embed.
- **Section detection** — parse PDF outline/bookmarks (and fall back to heading
  heuristics) so citations can read "Section 3.2" instead of just "page 7".
- **Streaming responses** from Claude for snappier UX.
- **Eval harness** — a small set of (question, expected-source) pairs and a script that
  measures retrieval recall@k and answer faithfulness.
- **Auth + multi-user persistence** — today, namespaces are per-session and ephemeral.
  A real deployment would tie namespaces to authenticated users.
- **Dockerfile + `docker compose up`** for an even simpler one-command run.

---

## 🧪 Tips for trying it out

- Start with the PDF you trust most and ask it a question whose answer you already know
  — it's the fastest way to sanity-check retrieval quality.
- If an answer looks off, open the **Sources** expander: 9 times out of 10, the wrong
  chunk got retrieved (a chunking/embedding issue), not a Claude hallucination.
- Tweak `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` in `.env` to trade off precision vs.
  context. Smaller chunks + higher `top_k` works better for specific facts; larger
  chunks help for summarisation-style questions.
