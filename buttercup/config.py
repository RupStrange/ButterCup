"""
Centralized configuration for ButterCup.

All tunables (models, chunk sizes, CRAG thresholds, retriever params) live
here so the rest of the codebase never hardcodes a magic number. Secrets are
read from Streamlit's `st.secrets` when available (deployed app) and fall
back to plain environment variables (local scripts / tests / notebooks).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_secret(key: str, default: str = "") -> str:
    """Read a config value from Streamlit secrets first, else env vars."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # --- LLM -----------------------------------------------------------
    groq_api_key: str = field(default_factory=lambda: _get_secret("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: os.environ.get(
            "GROQ_MODEL", "openai/gpt-oss-safeguard-20b"
        )
    )

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # --- Web search (Corrective RAG fallback) ---------------------------
    tavily_api_key: str = field(default_factory=lambda: _get_secret("TAVILY_API_KEY"))
    web_search_max_results: int = field(
        default_factory=lambda: int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "5"))
    )

    # --- Text splitting ---------------------------------------------------
    transcript_chunk_size: int = 1000
    transcript_chunk_overlap: int = 200
    translation_chunk_size: int = 1000
    translation_chunk_overlap: int = 200
    facts_chunk_size: int = 3500
    facts_chunk_overlap: int = 150

    # --- Retrieval ---------------------------------------------------------
    retriever_search_type: str = "mmr"
    retriever_k: int = 4
    retriever_fetch_k: int = 20
    retriever_lambda_mult: float = 0.5

    # --- Corrective RAG grading thresholds --------------------------------
    # score > upper  -> at least one chunk is trustworthy enough on its own (CORRECT)
    # score < lower  -> chunk is irrelevant
    # all(<lower)    -> nothing usable was retrieved (INCORRECT) -> go to web
    # otherwise      -> mixed signal (AMBIGUOUS) -> use retrieved + web
    crag_upper_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CRAG_UPPER_THRESHOLD", "0.7"))
    )
    crag_lower_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CRAG_LOWER_THRESHOLD", "0.3"))
    )
    min_sentence_len: int = 20  # sentences shorter than this are dropped during decomposition

    # --- Conversation memory ------------------------------------------------
    # Number of past messages (human + AI, so /2 for turns) kept as chat history.
    memory_max_messages: int = 12


settings = Settings()
