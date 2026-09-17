"""Builds the per-video FAISS retriever used as the CRAG graph's internal knowledge source."""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def build_retriever(cleaned_transcript: str, embeddings: Embeddings) -> VectorStoreRetriever:
    """Chunk a cleaned transcript, embed it, and return an MMR retriever over it."""
    docs = RecursiveCharacterTextSplitter(
        chunk_size=settings.transcript_chunk_size,
        chunk_overlap=settings.transcript_chunk_overlap,
    ).create_documents([cleaned_transcript])

    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vector_store.as_retriever(
        search_type=settings.retriever_search_type,
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
            "lambda_mult": settings.retriever_lambda_mult,
        },
    )
