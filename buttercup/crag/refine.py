"""
Sentence-level "strip" refinement.

After the graph decides which document pool to trust (internal, web, or
both), we don't hand the whole thing to the generator - we decompose it into
sentences and let an LLM judge keep only the ones that actually help answer
the question. This is the "knowledge refinement" step from the CRAG paper.
"""
from __future__ import annotations

import re
from typing import Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..config import settings
from ..schemas import CRAGState, KeepOrDrop

_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)


def decompose_to_sentences(text: str) -> List[str]:
    """Split text into sentences, dropping fragments that are too short to be useful."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > settings.min_sentence_len]


def make_refine_node(llm: BaseChatModel) -> Callable[[CRAGState], dict]:
    """
    Build the refine node.

    Which documents get decomposed depends on the verdict set by grading:
      CORRECT   -> good_docs (internal) only
      INCORRECT -> web_docs only
      AMBIGUOUS -> good_docs + web_docs
    """
    filter_chain = _FILTER_PROMPT | llm.with_structured_output(KeepOrDrop)

    def refine_node(state: CRAGState) -> dict:
        question = state["question"]
        verdict = state.get("verdict")

        if verdict == "CORRECT":
            docs_to_use = state["good_docs"]
        elif verdict == "INCORRECT":
            docs_to_use = state["web_docs"]
        else:  # AMBIGUOUS
            docs_to_use = state["good_docs"] + state["web_docs"]

        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        strips = decompose_to_sentences(context)

        kept: List[str] = []
        for sentence in strips:
            decision: KeepOrDrop = filter_chain.invoke({"question": question, "sentence": sentence})
            if decision.keep:
                kept.append(sentence)

        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": "\n".join(kept).strip(),
        }

    return refine_node
