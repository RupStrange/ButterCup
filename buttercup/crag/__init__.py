"""
Corrective RAG (CRAG) graph.

Pipeline: retrieve -> grade each doc -> route on verdict
  CORRECT   -> refine(internal docs only) -> generate
  INCORRECT -> rewrite_query -> web_search -> refine(web docs only) -> generate
  AMBIGUOUS -> rewrite_query -> web_search -> refine(internal + web docs) -> generate

Each node lives in its own module; `graph.build_crag_graph(...)` wires them
into a compiled LangGraph `StateGraph` for a specific video's retriever.
"""
from .graph import build_crag_graph

__all__ = ["build_crag_graph"]
