"""Fact-extraction + summary generation for a cleaned transcript."""
from __future__ import annotations

from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

ProgressCallback = Optional[Callable[[int, int], None]]

_parser = StrOutputParser()

_FACT_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Extract only explicit facts as short bullet points. No interpretation.\n\n{text}\n\nFacts:",
)

_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["facts"],
    template=(
        "Write a concise summary (max 500 words) using ONLY these facts. "
        "Chronological order, no new info.\n\n{facts}\n\nFinal Summary:"
    ),
)


def extract_facts(text: str, llm: BaseChatModel, on_progress: ProgressCallback = None) -> str:
    """Chunk the transcript and pull out explicit facts, chunk by chunk."""
    docs = RecursiveCharacterTextSplitter(
        chunk_size=settings.facts_chunk_size,
        chunk_overlap=settings.facts_chunk_overlap,
    ).create_documents([text])

    chain = _FACT_PROMPT | llm | _parser
    facts = []
    for i, doc in enumerate(docs):
        facts.append(chain.invoke(doc.page_content))
        if on_progress:
            on_progress(i + 1, len(docs))

    return "\n".join(facts)


def generate_summary(text: str, llm: BaseChatModel, on_progress: ProgressCallback = None) -> str:
    """Extract facts, then compress them into a single chronological summary."""
    facts = extract_facts(text, llm, on_progress)
    chain = _SUMMARY_PROMPT | llm | _parser
    return chain.invoke({"facts": facts})
