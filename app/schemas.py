from typing import Literal

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    source_id: str | None = None
    filename: str | None = None
    doc_type: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    text: str | None = None
    distance: float | None = None
    content_hash: str | None = None
    document_hash: str | None = None
    stored_file_path: str | None = None
    retrieval_score: float | None = None


class IngestTextRequest(BaseModel):
    text: str
    source_id: str | None = None
    filename: str | None = None
    doc_type: str = "general"


class IngestResponse(BaseModel):
    source_id: str
    chunks_indexed: int
    chunks_skipped: int = 0
    duplicate: bool = False
    stored_file_path: str | None = None
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    doc_type: str | None = None
    filename: str | None = None
    source_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    filters_applied: dict[str, str] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    source_id: str
    filename: str | None = None
    original_filename: str | None = None
    doc_type: str | None = None
    chunk_count: int = 0
    page_numbers: list[int] = Field(default_factory=list)
    preview: str | None = None
    stored_file_path: str | None = None
    original_file_available: bool = False
    document_hash: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    source_id: str
    deleted_count: int
    deleted_files: list[str] = Field(default_factory=list)
    message: str


class RagStatusResponse(BaseModel):
    service: str
    rag_enabled: bool
    workflow: str
    vector_db: str
    embedding_model: str
    weaviate_url: str
    weaviate_collection: str
    weaviate_reachable: bool
    message: str
