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
from ..schemas import CRAGState, SentenceFilterBatch
from .json_llm import call_for_json

_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "You will be given a question and a numbered list of candidate sentences.\n"
            "For EACH sentence independently, decide whether it directly helps answer "
            "the question. Judge each sentence using ONLY that sentence.\n"
            "Respond with ONLY a JSON object with exactly one key, \"decisions\", whose "
            "value is a JSON array with one entry per sentence, in the same order, each "
            "with exactly two keys:\n"
            '  "index": the sentence\'s 0-based position in the list\n'
            '  "keep": true or false\n'
            "No other text, no markdown fences.",
        ),
        ("human", "Question: {question}\n\nSentences:\n{sentences}"),
    ]
)

# Cap on sentences sent to the filter in one call. Keeps the prompt (and the
# JSON reply) a manageable size for long transcripts/web results instead of
# growing unbounded; batches beyond this are filtered in successive calls.
_MAX_SENTENCES_PER_CALL = 40


def _format_sentences(sentences: List[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def _filter_batch(llm: BaseChatModel, question: str, sentences: List[str]) -> List[str]:
    """Ask which of these sentences to keep, in as few calls as possible."""
    kept: List[str] = []
    for start in range(0, len(sentences), _MAX_SENTENCES_PER_CALL):
        chunk = sentences[start : start + _MAX_SENTENCES_PER_CALL]
        batch: SentenceFilterBatch = call_for_json(
            llm, _FILTER_PROMPT, {"question": question, "sentences": _format_sentences(chunk)}, SentenceFilterBatch
        )
        keep_by_index = {d.index: d.keep for d in batch.decisions}
        # Default any sentence the model skipped to "keep" so a partial/malformed
        # batch loses information rather than silently dropping context.
        kept.extend(s for i, s in enumerate(chunk) if keep_by_index.get(i, True))
    return kept


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
    # See json_llm.py for why we don't use llm.with_structured_output on Groq.

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

        # One call filters up to _MAX_SENTENCES_PER_CALL sentences at once,
        # instead of one call per sentence.
        kept: List[str] = _filter_batch(llm, question, strips) if strips else []

        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": "\n".join(kept).strip(),
        }

    return refine_node
