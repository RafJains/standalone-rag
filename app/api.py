import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import get_rag_settings
from app.document_loader import load_file_as_documents, load_text_as_documents
from app.schemas import (
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RagStatusResponse,
    SourceChunk,
)
from app.vector_store import add_documents, check_weaviate_ready, similarity_search


app = FastAPI(title="Rohan Standalone RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    settings = get_rag_settings()
    return {
        "status": "ok",
        "service": "rohan-rag-api",
        "rag_enabled": settings.rag_enabled,
        "workflow": settings.rag_workflow,
        "vector_db": settings.vector_db,
        "weaviate_collection": settings.weaviate_collection,
    }


@app.get("/rag/status", response_model=RagStatusResponse)
def rag_status() -> RagStatusResponse:
    settings = get_rag_settings()
    reachable, message = check_weaviate_ready()
    return RagStatusResponse(
        service="rohan-rag-api",
        rag_enabled=settings.rag_enabled,
        workflow=settings.rag_workflow,
        vector_db=settings.vector_db,
        embedding_model=settings.embedding_model,
        weaviate_url=settings.weaviate_url,
        weaviate_collection=settings.weaviate_collection,
        weaviate_reachable=reachable,
        message=message,
    )


@app.post("/rag/ingest/text", response_model=IngestResponse)
def ingest_text(request: IngestTextRequest) -> IngestResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    source_id = request.source_id or str(uuid4())
    documents = load_text_as_documents(
        text=request.text,
        source_id=source_id,
        filename=request.filename,
        doc_type=request.doc_type,
    )
    if not documents:
        raise HTTPException(status_code=400, detail="text did not produce any chunks")

    chunks_indexed = _index_documents(documents)
    return IngestResponse(
        source_id=source_id,
        chunks_indexed=chunks_indexed,
        message=f"Indexed {chunks_indexed} chunk(s) into Weaviate.",
    )


@app.post("/rag/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    source_id: str | None = Form(None),
) -> IngestResponse:
    filename = file.filename or "upload"
    resolved_source_id = source_id or str(uuid4())
    suffix = Path(filename).suffix

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        try:
            shutil.copyfileobj(file.file, temp_file)
        finally:
            await file.close()

    try:
        documents = load_file_as_documents(
            file_path=temp_path,
            source_id=resolved_source_id,
            filename=filename,
            doc_type=doc_type,
        )
        if not documents:
            raise HTTPException(status_code=400, detail="file did not produce any chunks")

        chunks_indexed = _index_documents(documents)
        return IngestResponse(
            source_id=resolved_source_id,
            chunks_indexed=chunks_indexed,
            message=f"Indexed {chunks_indexed} chunk(s) from {filename} into Weaviate.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.post("/rag/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    settings = get_rag_settings()
    top_k = request.top_k or settings.rag_top_k
    documents = _retrieve_documents(request.question, top_k)
    return QueryResponse(
        answer="Phase 2 retrieval-only response. Generator LLM will be added in Phase 3.",
        sources=[_document_to_source_chunk(document) for document in documents],
        retrieved_chunk_count=len(documents),
    )


def _index_documents(documents: list) -> int:
    try:
        return add_documents(documents)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _retrieve_documents(question: str, top_k: int) -> list:
    try:
        return similarity_search(question, top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _document_to_source_chunk(document) -> SourceChunk:
    metadata = document.metadata
    return SourceChunk(
        source_id=metadata.get("source_id"),
        filename=metadata.get("filename"),
        doc_type=metadata.get("doc_type"),
        chunk_index=metadata.get("chunk_index"),
        page_number=metadata.get("page_number"),
        text=document.page_content,
        distance=metadata.get("distance"),
    )
