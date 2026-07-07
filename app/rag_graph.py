from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.config import RagSettings, get_rag_settings
from app.generator import generate_answer
from app.rag_types import RagDocument
from app.schemas import SourceChunk
from app.vector_store import similarity_search


NO_RELEVANT_INFORMATION_ANSWER = "No relevant indexed information was found for this question."


class RagGraphState(TypedDict, total=False):
    question: str
    top_k: int
    retrieval_mode: Literal["vector", "hybrid"]
    filters: dict[str, str]
    documents: list[RagDocument]
    sources: list[SourceChunk]
    answer: str
    retrieved_chunk_count: int
    filters_applied: dict[str, str]
    error: str | None
    settings: RagSettings


def run_rag_graph(
    question: str,
    top_k: int,
    retrieval_mode: Literal["vector", "hybrid"],
    filters: dict[str, str] | None = None,
    settings: RagSettings | None = None,
) -> RagGraphState:
    graph = _get_graph()
    initial_state: RagGraphState = {
        "question": question,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "filters": filters or {},
        "documents": [],
        "sources": [],
        "answer": "",
        "retrieved_chunk_count": 0,
        "filters_applied": filters or {},
        "error": None,
        "settings": settings or get_rag_settings(),
    }
    return graph.invoke(initial_state)


def build_rag_graph():
    graph = StateGraph(RagGraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def retrieve_node(state: RagGraphState) -> RagGraphState:
    filters = state.get("filters") or {}
    documents = similarity_search(
        query=state["question"],
        top_k=state["top_k"],
        filters=filters,
        retrieval_mode=state["retrieval_mode"],
    )
    return {
        **state,
        "documents": documents,
        "sources": [_document_to_source_chunk(document) for document in documents],
        "retrieved_chunk_count": len(documents),
        "filters_applied": filters,
    }


def decide_node(state: RagGraphState) -> RagGraphState:
    return state


def route_after_decide(state: RagGraphState) -> str:
    return "generate" if state.get("documents") else "fallback"


def generate_node(state: RagGraphState) -> RagGraphState:
    answer = generate_answer(
        question=state["question"],
        documents=state.get("documents") or [],
        settings=state.get("settings"),
    )
    return {**state, "answer": answer}


def fallback_node(state: RagGraphState) -> RagGraphState:
    return {
        **state,
        "answer": NO_RELEVANT_INFORMATION_ANSWER,
        "sources": [],
        "retrieved_chunk_count": 0,
    }


def _get_graph():
    global _COMPILED_RAG_GRAPH
    if _COMPILED_RAG_GRAPH is None:
        _COMPILED_RAG_GRAPH = build_rag_graph()
    return _COMPILED_RAG_GRAPH


def _document_to_source_chunk(document: RagDocument) -> SourceChunk:
    metadata: dict[str, Any] = document.metadata
    return SourceChunk(
        source_id=metadata.get("source_id"),
        filename=metadata.get("filename"),
        doc_type=metadata.get("doc_type"),
        chunk_index=metadata.get("chunk_index"),
        page_number=metadata.get("page_number"),
        text=document.page_content,
        distance=metadata.get("distance"),
        content_hash=metadata.get("content_hash"),
        document_hash=metadata.get("document_hash"),
        stored_file_path=metadata.get("stored_file_path"),
        retrieval_score=metadata.get("retrieval_score"),
        section_title=metadata.get("section_title"),
        row_number=metadata.get("row_number"),
        chunk_char_count=metadata.get("chunk_char_count"),
    )


_COMPILED_RAG_GRAPH = None
