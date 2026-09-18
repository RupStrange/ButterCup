# ButterCup — YouTube AI Analyst with Corrective RAG

Turn any YouTube video into a clean summary and an ask-anything chat
interface. Answers are grounded in the video's own transcript first, graded
for relevance by an LLM, and automatically backed by a live web search
whenever the video itself doesn't cover what was asked — a technique known
as **Corrective RAG (CRAG)**.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Design Notes](#design-notes)
- [Roadmap](#roadmap)

## Features

- **Paste a link, get a summary** — fetches the transcript, translates/cleans
  it, and generates a structured, fact-checked summary.
- **Chat with the video** — ask follow-up questions and get answers grounded
  in what was actually said.
- **Self-correcting retrieval (CRAG)** — every retrieved chunk is graded for
  relevance before it's allowed to inform an answer.
- **Automatic web fallback** — if the video doesn't cover the question, the
  app rewrites the query and pulls in a live Tavily web search instead of
  hallucinating or refusing to answer.
- **Source transparency** — each answer is tagged so you can see where it
  came from: 🟢 video only, 🌐 web only, or 🟡 both.
- **Conversation memory** — follow-up questions are answered with awareness
  of the last several turns, not in isolation.

## How It Works

```
User question
     │
     ▼
┌─────────────┐      ┌───────────────────┐
│  retrieve   │ ───▶ │  eval_each_doc     │  LLM grades every chunk 0–1
└─────────────┘      └─────────┬─────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼          ▼          ▼
                 CORRECT   AMBIGUOUS   INCORRECT
                     │          │          │
                     │          ▼          ▼
                     │   rewrite query → Tavily web search
                     │          │          │
                     └────┬─────┴──────────┘
                          ▼
                    ┌─────────────┐
                    │   refine    │  keep only relevant sentences
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  generate   │  history-aware final answer
                    └─────────────┘
```

1. **`retrieve`** — pulls the top-k chunks from the video's FAISS index
   (MMR search for diversity, not just raw similarity).
2. **`eval_each_doc`** — an LLM independently scores each chunk's relevance
   to the question.
3. **Verdict** — based on two thresholds (`upper` / `lower`), the graph
   decides whether the retrieved context is `CORRECT` (trust it), `AMBIGUOUS`
   (use it *and* supplement with web search), or `INCORRECT` (discard it and
   search the web instead).
4. **`refine`** — trusted context is decomposed into individual sentences,
   and an LLM keeps only the ones actually relevant to the question, so the
   final prompt isn't diluted with tangential transcript text.
5. **`generate`** — the answer is produced from the refined context, aware
   of prior turns in the conversation.

## Architecture

The project started as a single 460-line Streamlit script that mixed UI
code, prompts, LLM calls, and progress bars together. It's now split by
responsibility so the retrieval/grading/generation logic has no dependency
on Streamlit and can be tested, reused in a CLI, or wrapped in an API
without touching the UI layer.

Only `app.py` imports `streamlit`; every other module is plain Python.

## Tech Stack

| Layer            | Choice                                              |
|-------------------|------------------------------------------------------|
| UI                | Streamlit                                            |
| LLM               | Groq (`langchain-groq`)                              |
| Orchestration     | LangGraph (compiled `StateGraph` for the CRAG flow)  |
| Vector store      | FAISS                                                |
| Embeddings        | `sentence-transformers` (`BAAI/bge-small-en-v1.5`)   |
| Web search        | Tavily                                               |
| Transcript source | `youtube-transcript-api`                             |
| Structured output | Pydantic v2 (`with_structured_output`)               |

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com)
- A [Tavily API key](https://tavily.com)

### Installation

```bash
git clone https://github.com/<your-username>/ButterCup.git
cd ButterCup
pip install -r requirements.txt
```

### Configure secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
TAVILY_API_KEY = "your-tavily-api-key-here"
```

> `.streamlit/secrets.toml` and `buttercup/secrets.toml` are already listed
> in `.gitignore` — never commit real keys.

### Run it

```bash
streamlit run app.py
```

## Configuration

All tunables live in `buttercup/config.py`, with the CRAG thresholds
overridable via environment variables:

| Setting                 | Default | Description                                                        |
|--------------------------|---------|----------------------------------------------------------------------|
| `CRAG_UPPER_THRESHOLD`  | `0.7`   | A chunk scoring above this alone justifies a `CORRECT` verdict.     |
| `CRAG_LOWER_THRESHOLD`  | `0.3`   | Chunks scoring below this are treated as irrelevant.                |
| `retriever_k`           | `4`     | Number of chunks retrieved per question.                            |
| `memory_max_messages`   | `12`    | Number of past chat messages kept as conversational context.        |

## Project Structure

```
buttercup_project/
├── app.py                      # Streamlit UI only — no prompts/LLM logic
├── requirements.txt
├── .streamlit/secrets.toml.example
└── buttercup/
    ├── config.py                # every tunable constant + secret lookup
    ├── schemas.py                # pydantic structured-output models + CRAGState
    ├── youtube_service.py        # URL parsing, video metadata, transcript fetch
    ├── transcript_processor.py   # translation + cleanup
    ├── summarizer.py             # fact extraction + summary generation
    ├── vectorstore_builder.py    # FAISS retriever construction
    ├── memory_store.py           # conversation memory wrapper
    └── crag/                     # the Corrective RAG graph, one node per file
        ├── grading.py             # retrieve + relevance-eval nodes
        ├── refine.py              # sentence decomposition + filtering
        ├── web_search.py          # query rewrite + Tavily search nodes
        ├── generate.py            # history-aware answer generation
        └── graph.py               # wires nodes into a compiled StateGraph
```

## Design Notes

- **Latency trade-off**: CRAG grades every retrieved chunk with a separate
  LLM call, then filters every refined sentence with another. With `k=4`
  that's easily 6–10+ LLM calls per question — fine for a demo, but a
  production deployment would benefit from batching grading calls
  (`chain.batch(...)`) or grading concatenated chunks in one call.
- **Structured output**: `with_structured_output` requires a Groq model
  that supports tool/function calling.
- **Tavily**: `TavilySearchResults` is community-maintained and slated for
  deprecation upstream in favor of `langchain-tavily`; swapping is a
  one-line change in `buttercup/crag/web_search.py`.

## Roadmap

- [ ] Batch/parallelize CRAG grading calls to cut per-question latency
- [ ] Swap `TavilySearchResults` for `langchain-tavily`
- [ ] Add automated tests for the grading and refine nodes
- [ ] Support multi-video sessions (chat across more than one video at once)

## License

Add a license of your choice (MIT is a common default for portfolio
projects) before making the repository public.
