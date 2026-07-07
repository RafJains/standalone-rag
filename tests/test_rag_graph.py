import pytest

from app import rag_graph
from app.rag_graph import NO_RELEVANT_INFORMATION_ANSWER, build_rag_graph, run_rag_graph
from app.rag_types import RagDocument


def reset_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_graph, "_COMPILED_RAG_GRAPH", None)


def test_graph_returns_fallback_without_calling_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_graph(monkeypatch)
    monkeypatch.setattr(rag_graph, "similarity_search", lambda **kwargs: [])

    def fail_generate_answer(**kwargs):
        raise AssertionError("generator should not be called")

    monkeypatch.setattr(rag_graph, "generate_answer", fail_generate_answer)

    result = run_rag_graph(
        question="No match?",
        top_k=3,
        retrieval_mode="vector",
        filters={"project_id": "alpha"},
    )

    assert result["answer"] == NO_RELEVANT_INFORMATION_ANSWER
    assert result["sources"] == []
    assert result["retrieved_chunk_count"] == 0
    assert result["filters_applied"] == {"project_id": "alpha"}


def test_graph_generates_answer_and_source_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_graph(monkeypatch)
    document = RagDocument(
        "Alpha context",
        {
            "project_id": "alpha",
            "source_id": "source-1",
            "filename": "alpha.txt",
            "doc_type": "policy",
            "chunk_index": 0,
        },
    )
    monkeypatch.setattr(rag_graph, "similarity_search", lambda **kwargs: [document])
    monkeypatch.setattr(rag_graph, "generate_answer", lambda **kwargs: "mocked answer")

    result = run_rag_graph(
        question="What?",
        top_k=1,
        retrieval_mode="hybrid",
        filters={"project_id": "alpha"},
    )

    assert result["answer"] == "mocked answer"
    assert result["retrieved_chunk_count"] == 1
    assert result["sources"][0].project_id == "alpha"
    assert result["sources"][0].filename == "alpha.txt"
    assert result["sources"][0].doc_type == "policy"


def test_graph_nodes_are_preserved() -> None:
    compiled = build_rag_graph()
    graph = compiled.get_graph()

    assert {"retrieve", "decide", "generate", "fallback"}.issubset(set(graph.nodes))
