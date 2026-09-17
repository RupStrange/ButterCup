"""
Lightweight conversation memory.

`langchain.memory.ConversationSummaryBufferMemory` was deprecated in
langchain 0.3.1 and has since been removed from current `langchain`
releases (import now fails with
`ModuleNotFoundError: No module named 'langchain.memory'`). Rather than pin
this project to an old langchain version, we keep our own minimal history:
a capped list of `langchain_core` messages, which is a stable, non-deprecated
module. This is intentionally simple (no LLM-generated running summary) -
the CRAG generate node only needs recent turns for conversational context,
not a compressed summary of the whole conversation.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import settings


class ChatMemory:
    """Keeps the last `max_messages` chat messages (human + AI turns)."""

    def __init__(self, max_messages: Optional[int] = None):
        self.max_messages = max_messages or settings.memory_max_messages
        self._messages: List[BaseMessage] = []

    def get_history(self) -> List[BaseMessage]:
        return self._messages[-self.max_messages :]

    def save_turn(self, question: str, answer: str) -> None:
        self._messages.append(HumanMessage(content=question))
        self._messages.append(AIMessage(content=answer))

    def clear(self) -> None:
        self._messages = []


def create_memory(max_messages: Optional[int] = None) -> ChatMemory:
    return ChatMemory(max_messages=max_messages)


def get_history(memory: Optional[ChatMemory]) -> List[BaseMessage]:
    if memory is None:
        return []
    return memory.get_history()


def save_turn(memory: Optional[ChatMemory], question: str, answer: str) -> None:
    if memory:
        memory.save_turn(question, answer)


def clear(memory: Optional[ChatMemory]) -> None:
    if memory:
        memory.clear()
