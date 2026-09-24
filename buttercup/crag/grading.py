"""Retrieve node + the per-chunk relevance evaluator that drives CRAG's verdict."""
from __future__ import annotations

from typing import Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from ..schemas import CRAGState, DocEvalScore
from .json_llm import call_for_json

_DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Score how relevant the chunk is in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Respond with ONLY a JSON object with exactly two keys:\n"
            '  "score": a number between 0.0 and 1.0\n'
            '  "reason": a short string justifying the score\n'
            "No other text, no markdown fences.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)


def make_retrieve_node(retriever: VectorStoreRetriever) -> Callable[[CRAGState], dict]:
    """Bind a video-specific retriever into a graph node."""

    def retrieve_node(state: CRAGState) -> dict:
        return {"docs": retriever.invoke(state["question"])}

    return retrieve_node


def make_eval_each_doc_node(
    llm: BaseChatModel,
    upper_threshold: float,
    lower_threshold: float,
) -> Callable[[CRAGState], dict]:
    """
    Build the node that scores every retrieved chunk and assigns a verdict:

      CORRECT   - at least one chunk scored above `upper_threshold`
      INCORRECT - every chunk scored below `lower_threshold`
      AMBIGUOUS - anything in between
    """
    # See json_llm.py: llm.with_structured_output (any method) routes
    # through Groq's tool-calling machinery, and gpt-oss reasoning models
    # intermittently hallucinate a tool call even when tool_choice="none",
    # which Groq then rejects with a tool_use_failed 400. call_for_json
    # asks for plain-text JSON and parses it ourselves instead.

    def eval_each_doc_node(state: CRAGState) -> dict:
        question = state["question"]
        scores: List[float] = []
        good_docs = []

        for doc in state["docs"]:
            result: DocEvalScore = call_for_json(
                llm, _DOC_EVAL_PROMPT, {"question": question, "chunk": doc.page_content}, DocEvalScore
            )
            scores.append(result.score)
            if result.score > lower_threshold:
                good_docs.append(doc)

        if not scores:
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": "No chunks were retrieved.",
            }

        if any(s > upper_threshold for s in scores):
            return {
                "good_docs": good_docs,
                "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {upper_threshold}.",
            }

        if all(s < lower_threshold for s in scores):
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {lower_threshold}.",
            }

        return {
            "good_docs": good_docs,
            "verdict": "AMBIGUOUS",
            "reason": f"No chunk scored > {upper_threshold}, but not all were < {lower_threshold}.",
        }

    return eval_each_doc_node
