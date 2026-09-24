"""
Manual JSON calling, to work around a Groq + gpt-oss bug.

This started as a workaround for with_structured_output routing through
Groq's tool-calling machinery. It turned out that's not the whole story:
openai/gpt-oss models on Groq intermittently emit a tool-call-shaped
response even on a bare plain-text chat call with zero tools bound - the
misbehavior is in Groq's serving stack for gpt-oss, not in anything the
client sends. Groq then rejects the request with a 400 "Tool choice is
none, but model called a tool" / tool_use_failed error. There is no
client-side setting that prevents it; it's a known, ongoing bug (see
Groq's community forum and langchain-groq's GitHub issues). The
recommended real fix is to not use a gpt-oss model on Groq for anything
that needs reliable JSON output (see config.py - default model is now
llama-3.3-70b-versatile).

Since the bug is intermittent, this also retries the whole call (not
just JSON parsing) a few times, which is a reasonable mitigation for
whichever model is configured but does not guarantee success on gpt-oss.
"""
from __future__ import annotations

import json
import re

from groq import BadRequestError as GroqBadRequestError
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

_MAX_ATTEMPTS = 3


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def call_for_json(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    variables: dict,
    schema: type[BaseModel],
) -> BaseModel:
    """Call the llm as plain text, parse the reply as JSON, validate against schema.

    Plain-text call - no tools bound, no response_format set - so this
    never asks Groq for tool calling. Retries a few times both on a
    Groq tool_use_failed error and on the model's output not being
    valid JSON.
    """
    chain = prompt | llm | StrOutputParser()

    last_error: Exception | None = None
    attempt = 0
    while attempt < _MAX_ATTEMPTS:
        attempt = attempt + 1
        try:
            raw_text = chain.invoke(variables)
            cleaned_text = _strip_code_fences(raw_text)
            data = json.loads(cleaned_text)
            parsed = schema.model_validate(data)
            return parsed
        except GroqBadRequestError as error:
            last_error = error
            continue
        except Exception as error:
            last_error = error
            continue

    raise ValueError(
        f"Could not get valid JSON from model after {_MAX_ATTEMPTS} attempts: {last_error}"
    )
