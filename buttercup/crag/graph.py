"""Wires the CRAG nodes into a compiled LangGraph StateGraph for one video's retriever."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..config import settings
from ..schemas import CRAGState
from .generate import make_generate_node
from .grading import make_eval_each_doc_node, make_retrieve_node
from .refine import make_refine_node
from .web_search import make_rewrite_query_node, make_web_search_node


def _route_after_eval(state: CRAGState) -> str:
    """CORRECT skips web search entirely; everything else needs a query rewrite first."""
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"


def build_crag_graph(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    tavily_api_key: str | None = None,
    upper_threshold: float | None = None,
    lower_threshold: float | None = None,
    web_search_max_results: int | None = None,
) -> CompiledStateGraph:
    """
    Build and compile the Corrective RAG graph for a specific video's retriever.

    retrieve -> eval_each_doc -> [CORRECT] -> refine -> generate
                              -> [INCORRECT/AMBIGUOUS] -> rewrite_query -> web_search -> refine -> generate
    """
    tavily_api_key = tavily_api_key or settings.tavily_api_key
    upper_threshold = upper_threshold if upper_threshold is not None else settings.crag_upper_threshold
    lower_threshold = lower_threshold if lower_threshold is not None else settings.crag_lower_threshold
    web_search_max_results = web_search_max_results or settings.web_search_max_results

    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("eval_each_doc", make_eval_each_doc_node(llm, upper_threshold, lower_threshold))
    graph.add_node("rewrite_query", make_rewrite_query_node(llm))
    graph.add_node("web_search", make_web_search_node(tavily_api_key, web_search_max_results))
    graph.add_node("refine", make_refine_node(llm))
    graph.add_node("generate", make_generate_node(llm))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "eval_each_doc")
    graph.add_conditional_edges(
        "eval_each_doc",
        _route_after_eval,
        {"refine": "refine", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "refine")
    graph.add_edge("refine", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def empty_state(question: str, history: list | None = None) -> CRAGState:
    """Convenience constructor for a fresh invocation's input state."""
    return CRAGState(
        question=question,
        history=history or [],
        docs=[],
        good_docs=[],
        verdict="",
        reason="",
        strips=[],
        kept_strips=[],
        refined_context="",
        web_query="",
        web_docs=[],
        answer="",
    )
