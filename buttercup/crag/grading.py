"""Retrieve node + the per-chunk relevance evaluator that drives CRAG's verdict."""
from __future__ import annotations

from typing import Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from ..schemas import CRAGState, DocEvalBatch
from .json_llm import call_for_json

_DOC_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given a question and a numbered list of retrieved chunks.\n"
            "Score EACH chunk's relevance independently, in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Respond with ONLY a JSON object with exactly one key, \"evaluations\", whose "
            "value is a JSON array with one entry per chunk, in the same order, each with "
            "exactly three keys:\n"
            '  "index": the chunk\'s 0-based position in the list\n'
            '  "score": a number between 0.0 and 1.0\n'
            '  "reason": a short string justifying the score\n'
            "No other text, no markdown fences.",
        ),
        ("human", "Question: {question}\n\nChunks:\n{chunks}"),
    ]
)


def _format_chunks(docs: List) -> str:
    return "\n\n".join(f"[{i}]\n{doc.page_content}" for i, doc in enumerate(docs))


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
        docs = state["docs"]

        if not docs:
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": "No chunks were retrieved.",
            }

        # One call grades every chunk at once instead of one call per chunk.
        batch: DocEvalBatch = call_for_json(
            llm, _DOC_EVAL_PROMPT, {"question": question, "chunks": _format_chunks(docs)}, DocEvalBatch
        )

        # Map back by index; default any chunk the model skipped to 0.0 so a
        # partial/malformed batch degrades gracefully instead of crashing.
        score_by_index = {item.index: item.score for item in batch.evaluations if 0 <= item.index < len(docs)}
        scores: List[float] = [score_by_index.get(i, 0.0) for i in range(len(docs))]
        good_docs = [doc for doc, score in zip(docs, scores) if score > lower_threshold]

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
