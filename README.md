# ButterCup — YouTube AI Analyst (with Corrective RAG chat)

Turns a YouTube video into a summary + a chat interface. The chat tab is now
backed by **Corrective RAG (CRAG)**: every answer is graded against the
retrieved transcript context, and the graph automatically falls back to a
live web search when the video doesn't actually cover what was asked.

## What changed from the original script

The original `utube_chatbot.py` was one 460-line file mixing Streamlit
widgets, prompts, LLM calls, and progress bars together, and its chat tab
was a plain `retriever -> prompt -> LLM` chain with no self-correction.

This version:

1. **Adds Corrective RAG**, adapted from your `6_ambiguous.ipynb` notebook,
   as the chat engine:
   - `retrieve` → pull chunks from the video's FAISS index
   - `eval_each_doc` → an LLM grades every chunk's relevance (0–1)
   - verdict: `CORRECT` (skip web) / `INCORRECT` / `AMBIGUOUS` (rewrite query → Tavily web search)
   - `refine` → decompose the trusted context into sentences, keep only the ones an LLM judges relevant
   - `generate` → answer from the refined context, aware of prior chat turns
   - The chat UI shows a badge under each answer: 🟢 from the video, 🌐 from
     the web, or 🟡 both — so it's visible when CRAG kicks in.
   - Swapped the notebook's `ChatOpenAI` for `ChatGroq`, to match the app's
     existing model, and reused the app's own FAISS retriever as CRAG's
     internal knowledge source instead of PDFs.

2. **Splits the single script into a package** organized by responsibility:

```
buttercup_project/
├── app.py                       # Streamlit UI only — no prompts/LLM logic
├── requirements.txt
├── .streamlit/secrets.toml.example
└── buttercup/
    ├── config.py                 # every tunable constant + secret lookup, in one place
    ├── schemas.py                 # pydantic structured-output models + CRAGState
    ├── youtube_service.py         # URL parsing, video metadata, transcript fetch (no Streamlit)
    ├── transcript_processor.py    # translation + cleanup (progress via callback, not st.progress)
    ├── summarizer.py               # fact extraction + summary generation
    ├── vectorstore_builder.py      # FAISS retriever construction
    ├── memory_store.py             # conversation memory wrapper
    └── crag/                       # the Corrective RAG graph, one node per file
        ├── grading.py               # retrieve + relevance-eval nodes
        ├── refine.py                 # sentence decomposition + filtering
        ├── web_search.py             # query rewrite + Tavily search nodes
        ├── generate.py                # history-aware answer generation
        └── graph.py                   # wires nodes into a compiled StateGraph
```

Every non-`app.py` module is plain Python with no Streamlit import (except
`config.py`'s optional `st.secrets` read), so the retrieval, grading, and
generation logic can be unit tested or reused in a CLI/API without spinning
up a UI. `app.py` is the only file that touches `st.*` widgets.

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in GROQ_API_KEY and TAVILY_API_KEY
streamlit run app.py
```

## Tuning CRAG

Grading thresholds and all other knobs live in `buttercup/config.py` (or set
via env vars `CRAG_UPPER_THRESHOLD` / `CRAG_LOWER_THRESHOLD`):

- `crag_upper_threshold` (default `0.7`) — a chunk above this alone justifies `CORRECT`
- `crag_lower_threshold` (default `0.3`) — chunks below this are treated as irrelevant

## Notes / things worth knowing before you deploy this

- **Latency**: CRAG grades every retrieved chunk with a separate LLM call,
  then filters every refined sentence with another. With `k=4` that's easily
  6–10+ LLM calls per question. Fine for a demo; for production you may want
  to batch the grading calls (`chain.batch(...)`) or grade chunks concatenated
  in one call instead of one-by-one.
- **Structured output**: `with_structured_output` needs a Groq model that
  supports tool/function calling. `meta-llama/llama-4-scout-17b-16e-instruct`
  (the app's existing model) supports this; if you swap models, confirm the
  new one does too.
- **Tavily**: `TavilySearchResults` is community-maintained and slated for
  deprecation upstream in favor of `langchain-tavily`. It's kept here to
  match your notebook 1:1; swapping is a one-line change in
  `buttercup/crag/web_search.py`.
