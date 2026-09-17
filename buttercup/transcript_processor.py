"""
Transcript translation + cleanup.

These functions used to call `st.progress(...)` directly, which made them
untestable outside a running Streamlit app. Instead they accept an optional
`on_progress(done, total)` callback - the UI layer decides what to do with
that (render a progress bar, log it, ignore it, etc.).
"""
from __future__ import annotations
import time
from groq import RateLimitError
import re
from typing import Callable, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

ProgressCallback = Optional[Callable[[int, int], None]]

_parser = StrOutputParser()

_TRANSLATE_PROMPT = PromptTemplate(
    template=(
        "You are an expert translator. Translate from {language_code} to English.\n"
        "Output ONLY the translated text.\n"
        "text: {snippet}"
    ),
    input_variables=["language_code", "snippet"],
)


def translate_transcript(
    transcript_snippets: List,
    language_code: str,
    llm: BaseChatModel,
    on_progress: ProgressCallback = None,
) -> str:
    """Join transcript snippets into text, translating to English if needed."""

    full_text = " ".join(snippet.text for snippet in transcript_snippets)

    if language_code == "en":
        return full_text
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.translation_chunk_size,
        chunk_overlap=settings.translation_chunk_overlap,
    ).split_text(full_text)

    chain = _TRANSLATE_PROMPT | llm | _parser

    translated_chunks = []

    for i, chunk in enumerate(chunks):
        while True:
            try:
                translated_text = chain.invoke(
                    {
                        "language_code": language_code,
                        "snippet": chunk,
                    }
                )

                translated_chunks.append(translated_text)
                break

            except RateLimitError:
                print("Groq rate limit reached. Waiting 2 seconds...")
                time.sleep(2)

        if on_progress:
            on_progress(i + 1, len(chunks))

    return " ".join(translated_chunks)

def clean_transcript(
    transcript_snippets: List,
    language_code: str,
    llm: BaseChatModel,
    on_progress: ProgressCallback = None,
) -> str:
    """Translate (if needed) and normalize whitespace on a transcript."""
    text = translate_transcript(transcript_snippets, language_code, llm, on_progress)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
