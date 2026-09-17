"""Web-search fallback used when the video's own transcript can't answer the question."""
from __future__ import annotations

from typing import Callable, List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..schemas import CRAGState, WebQuery

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), "
            "add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)


def make_rewrite_query_node(llm: BaseChatModel) -> Callable[[CRAGState], dict]:
    rewrite_chain = _REWRITE_PROMPT | llm.with_structured_output(WebQuery)

    def rewrite_query_node(state: CRAGState) -> dict:
        result: WebQuery = rewrite_chain.invoke({"question": state["question"]})
        return {"web_query": result.query}

    return rewrite_query_node


def make_web_search_node(tavily_api_key: str, max_results: int) -> Callable[[CRAGState], dict]:
    tavily = TavilySearchResults(max_results=max_results, tavily_api_key=tavily_api_key)

    def web_search_node(state: CRAGState) -> dict:
        query = state.get("web_query") or state["question"]
        results = tavily.invoke({"query": query})

        web_docs: List[Document] = []
        for r in results or []:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

        return {"web_docs": web_docs}

    return web_search_node
