from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    source_id: str | None = None
    filename: str | None = None
    doc_type: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    text: str | None = None
    distance: float | None = None


class IngestTextRequest(BaseModel):
    text: str
    source_id: str | None = None
    filename: str | None = None
    doc_type: str = "general"


class IngestResponse(BaseModel):
    source_id: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    retrieved_chunk_count: int = 0


class DocumentSummary(BaseModel):
    source_id: str
    filename: str | None = None
    doc_type: str | None = None
    chunk_count: int = 0
    page_numbers: list[int] = Field(default_factory=list)
    preview: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary] = Field(default_factory=list)


class DeleteDocumentResponse(BaseModel):
    source_id: str
    deleted_count: int
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
