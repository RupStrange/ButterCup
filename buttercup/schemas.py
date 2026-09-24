"""Structured-output models and the shared LangGraph state definition."""
from __future__ import annotations

from typing import Any, List, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class DocEvalScore(BaseModel):
    """Structured relevance grade the evaluator LLM assigns to one chunk."""

    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score in [0, 1].")
    reason: str = Field(..., description="Short justification for the score.")


class KeepOrDrop(BaseModel):
    """Structured keep/drop decision made while refining context sentence-by-sentence."""

    keep: bool


class WebQuery(BaseModel):
    """Structured web-search query rewritten from the user's raw question."""

    query: str


class NeedsRetrieval(BaseModel):
    """Structured router decision: does this question need transcript/web retrieval at all?"""

    needs_retrieval: bool = Field(
        ...,
        description=(
            "False for greetings, chit-chat, thanks, or questions about the assistant "
            "itself. True for anything that needs the video's content or outside facts."
        ),
    )


class CRAGState(TypedDict, total=False):
    """State threaded through every node of the Corrective RAG graph."""

    question: str
    history: List[Any]  # list[BaseMessage] - prior chat turns, for conversational context

    docs: List[Document]
    good_docs: List[Document]

    verdict: str  # "DIRECT" | "CORRECT" | "INCORRECT" | "AMBIGUOUS"
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    web_docs: List[Document]

    answer: str
