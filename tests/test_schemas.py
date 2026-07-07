import pytest
from pydantic import ValidationError

from app.schemas import (
    DeleteDocumentResponse,
    DocumentSummary,
    IngestTextRequest,
    QueryRequest,
    RagStatusResponse,
    SourceChunk,
)


def test_ingest_text_request_accepts_project_id_and_existing_fields() -> None:
    request = IngestTextRequest(
        text="hello",
        project_id="alpha",
        source_id="source-1",
        filename="note.txt",
        doc_type="policy",
    )

    assert request.project_id == "alpha"
    assert request.source_id == "source-1"
    assert request.filename == "note.txt"
    assert request.doc_type == "policy"


def test_query_request_project_id_and_retrieval_modes() -> None:
    vector = QueryRequest(question="What?", project_id="alpha", retrieval_mode="vector")
    hybrid = QueryRequest(question="What?", retrieval_mode="hybrid")

    assert vector.project_id == "alpha"
    assert hybrid.retrieval_mode == "hybrid"
    with pytest.raises(ValidationError):
        QueryRequest(question="What?", retrieval_mode="keyword")


def test_response_schemas_include_project_id_and_existing_fields() -> None:
    source = SourceChunk(
        project_id="alpha",
        source_id="source-1",
        filename="note.txt",
        doc_type="policy",
        chunk_index=0,
        text="context",
    )
    summary = DocumentSummary(
        project_id="alpha",
        source_id="source-1",
        filename="note.txt",
        chunk_count=1,
    )
    deleted = DeleteDocumentResponse(
        project_id="alpha",
        source_id="source-1",
        deleted_count=1,
        message="deleted",
    )

    assert source.project_id == "alpha"
    assert source.text == "context"
    assert summary.project_id == "alpha"
    assert summary.chunk_count == 1
    assert deleted.project_id == "alpha"
    assert deleted.deleted_files == []


def test_status_schema_exposes_auth_and_default_project_fields() -> None:
    status = RagStatusResponse(
        service="rohan-rag-api",
        rag_enabled=True,
        workflow="langgraph",
        vector_db="weaviate",
        embedding_model="BAAI/bge-m3",
        weaviate_url="http://weaviate:8080",
        weaviate_collection="KnowledgeBase",
        weaviate_reachable=True,
        auth_required=True,
        default_project_id="default",
        message="ok",
    )

    assert status.workflow == "langgraph"
    assert status.auth_required is True
    assert status.default_project_id == "default"
