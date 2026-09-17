"""Final answer generation from the refined context, aware of prior chat turns."""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..schemas import CRAGState

_SYSTEM_PROMPT = """You are ButterCup, an expert analyst assistant for YouTube video transcripts.

Your behavior:
- Answer using the provided context below, which may come from the video transcript,
  a live web search, or both, depending on how well the video covered the topic.
- If the context is empty or insufficient, say: "I don't know."
- Never give one-word answers; always be thorough and helpful.
- Remember prior conversation turns and refer to them when relevant.
- Greet warmly if greeted; introduce yourself as ButterCup.

Context:
{context}
"""

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

_parser = StrOutputParser()


def make_generate_node(llm: BaseChatModel) -> Callable[[CRAGState], dict]:
    chain = _ANSWER_PROMPT | llm | _parser

    def generate_node(state: CRAGState) -> dict:
        answer = chain.invoke(
            {
                "question": state["question"],
                "context": state.get("refined_context", ""),
                "history": state.get("history", []),
            }
        )
        return {"answer": answer}

    return generate_node
