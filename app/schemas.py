from typing import Literal

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    project_id: str | None = None
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
    section_title: str | None = None
    row_number: int | None = None
    chunk_char_count: int | None = None


class IngestTextRequest(BaseModel):
    text: str
    project_id: str | None = None
    source_id: str | None = None
    filename: str | None = None
    doc_type: str = "general"


class IngestResponse(BaseModel):
    project_id: str
    source_id: str
    chunks_indexed: int
    chunks_skipped: int = 0
    duplicate: bool = False
    stored_file_path: str | None = None
    parser_used: str | None = None
    warnings: list[str] = Field(default_factory=list)
    original_file_size_bytes: int | None = None
    detected_extension: str | None = None
    message: str


class QueryRequest(BaseModel):
    question: str
    project_id: str | None = None
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
    project_id: str
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
    parser_used: str | None = None
    warnings: list[str] = Field(default_factory=list)
    original_file_size_bytes: int | None = None
    detected_extension: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    project_id: str
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
    auth_required: bool = False
    default_project_id: str = "default"
    message: str
