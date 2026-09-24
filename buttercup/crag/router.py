"""
Cheap up-front router.

Not every message needs the full retrieve -> grade -> refine pipeline -
"hi", "thanks!", or "what can you do?" don't touch the transcript or the
web at all. Running those through CRAG anyway wastes calls and, on Groq's
reasoning models, is exactly the kind of off-topic grading call that
tends to trigger flaky tool-call behavior. This node makes one cheap LLM
call up front to decide whether retrieval is needed; if not, the graph
skips straight to generation with an empty context.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..schemas import CRAGState, NeedsRetrieval
from .json_llm import call_for_json

_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether answering this question requires looking at the video's "
            "transcript or a live web search.\n"
            "Say it does NOT need retrieval for greetings, small talk, thanks, or "
            "questions about the assistant itself (e.g. \"hi\", \"thanks!\", \"what can "
            "you do?\").\n"
            "Say it DOES need retrieval for anything asking about the video's content, "
            "facts, opinions, or outside information.\n"
            "Respond with ONLY a JSON object with exactly one key:\n"
            '  "needs_retrieval": true or false\n'
            "No other text, no markdown fences.",
        ),
        ("human", "Question: {question}"),
    ]
)


def make_route_node(llm: BaseChatModel) -> Callable[[CRAGState], dict]:
    # See json_llm.py for why we don't use llm.with_structured_output on Groq.
    def route_node(state: CRAGState) -> dict:
        result: NeedsRetrieval = call_for_json(
            llm, _ROUTE_PROMPT, {"question": state["question"]}, NeedsRetrieval
        )
        return {"verdict": "" if result.needs_retrieval else "DIRECT"}

    return route_node


def route_decision(state: CRAGState) -> str:
    return "generate" if state.get("verdict") == "DIRECT" else "retrieve"
