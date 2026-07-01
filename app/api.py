from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_rag_settings
from app.document_loader import load_file_as_documents, load_text_as_documents
from app.rag_chain import GeneratorUnavailableError, generate_answer
from app.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummary,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RagStatusResponse,
    SourceChunk,
)
from app.storage import (
    content_hash_from_text,
    delete_source_folder_if_empty,
    delete_stored_file,
    document_hash_from_bytes,
    document_hash_from_text,
    ensure_data_directories,
    relative_storage_path,
    resolve_stored_path,
    safe_filename,
    save_text,
    save_upload_bytes,
)
from app.vector_store import (
    add_documents,
    check_weaviate_ready,
    delete_documents_by_source_id,
    get_document_by_hash,
    get_document_storage,
    list_documents,
    similarity_search,
)


app = FastAPI(title="Rohan Standalone RAG API", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    ensure_data_directories()


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.head("/", include_in_schema=False)
def frontend_head() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
        "generator_model": settings.rag_generator_model,
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


@app.get("/rag/documents", response_model=DocumentListResponse)
def documents() -> DocumentListResponse:
    try:
        summaries = list_documents()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    summaries = [_with_file_availability(summary) for summary in summaries]
    return DocumentListResponse(
        documents=[DocumentSummary(**summary) for summary in summaries]
    )


@app.head("/rag/documents/{source_id}/download")
@app.get("/rag/documents/{source_id}/download")
def download_document(source_id: str) -> FileResponse:
    if not source_id.strip():
        raise HTTPException(status_code=400, detail="source_id cannot be empty")

    try:
        storage = get_document_storage(source_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not storage["exists"]:
        raise HTTPException(status_code=404, detail=f"No indexed document found for source_id {source_id}.")

    for stored_file_path in storage["stored_file_paths"]:
        path = resolve_stored_path(stored_file_path)
        if path and path.exists() and path.is_file():
            return FileResponse(path, filename=path.name)

    raise HTTPException(
        status_code=404,
        detail="Original file is not available for this indexed document.",
    )


@app.delete("/rag/documents/{source_id}", response_model=DeleteDocumentResponse)
def delete_document(source_id: str) -> DeleteDocumentResponse:
    if not source_id.strip():
        raise HTTPException(status_code=400, detail="source_id cannot be empty")

    try:
        storage = get_document_storage(source_id)
        deleted_count = delete_documents_by_source_id(source_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    deleted_files = []
    for stored_file_path in storage["stored_file_paths"]:
        deleted_file = delete_stored_file(stored_file_path)
        if deleted_file:
            deleted_files.append(deleted_file)
    delete_source_folder_if_empty(source_id)

    if deleted_count:
        if deleted_files:
            message = (
                f"Deleted {deleted_count} indexed chunk(s) and {len(deleted_files)} "
                f"stored file(s) for source_id {source_id}."
            )
        else:
            message = (
                f"Deleted {deleted_count} indexed chunk(s) for source_id {source_id}; "
                "no stored original file was available."
            )
    else:
        message = f"No indexed chunks found for source_id {source_id}."

    return DeleteDocumentResponse(
        source_id=source_id,
        deleted_count=deleted_count,
        deleted_files=deleted_files,
        message=message,
    )


@app.post("/rag/ingest/text", response_model=IngestResponse)
def ingest_text(request: IngestTextRequest) -> IngestResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    source_id = request.source_id or str(uuid4())
    filename = safe_filename(request.filename, "manual-note.txt")
    doc_type = (request.doc_type or "general").strip() or "general"
    document_hash = document_hash_from_text(request.text, filename, doc_type)

    try:
        existing = get_document_by_hash(document_hash)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if existing:
        return _duplicate_ingest_response(existing)

    stored_path = save_text(source_id, filename, request.text)
    stored_file_path = relative_storage_path(stored_path)
    saved_text = stored_path.read_text(encoding="utf-8")
    documents = load_text_as_documents(
        text=saved_text,
        source_id=source_id,
        filename=filename,
        doc_type=doc_type,
    )
    if not documents:
        raise HTTPException(status_code=400, detail="text did not produce any chunks")

    _enrich_documents(
        documents=documents,
        stored_file_path=stored_file_path,
        original_filename=request.filename or filename,
        document_hash=document_hash,
    )
    chunks_indexed, chunks_skipped = _index_documents(documents)
    return IngestResponse(
        source_id=source_id,
        chunks_indexed=chunks_indexed,
        chunks_skipped=chunks_skipped,
        duplicate=False,
        stored_file_path=stored_file_path,
        message=f"Indexed {chunks_indexed} chunk(s) into Weaviate.",
    )


@app.post("/rag/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    source_id: str | None = Form(None),
) -> IngestResponse:
    resolved_source_id = source_id or str(uuid4())
    original_filename = file.filename or "upload"
    filename = safe_filename(original_filename, "upload")
    doc_type = (doc_type or "general").strip() or "general"

    try:
        content = await file.read()
    finally:
        await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="file cannot be empty")

    document_hash = document_hash_from_bytes(content, doc_type)
    try:
        existing = get_document_by_hash(document_hash)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if existing:
        return _duplicate_ingest_response(existing)

    stored_path = save_upload_bytes(resolved_source_id, filename, content)
    stored_file_path = relative_storage_path(stored_path)

    try:
        documents = load_file_as_documents(
            file_path=str(stored_path),
            source_id=resolved_source_id,
            filename=filename,
            doc_type=doc_type,
        )
        if not documents:
            raise HTTPException(status_code=400, detail="file did not produce any chunks")

        _enrich_documents(
            documents=documents,
            stored_file_path=stored_file_path,
            original_filename=original_filename,
            document_hash=document_hash,
        )
        chunks_indexed, chunks_skipped = _index_documents(documents)
        return IngestResponse(
            source_id=resolved_source_id,
            chunks_indexed=chunks_indexed,
            chunks_skipped=chunks_skipped,
            duplicate=False,
            stored_file_path=stored_file_path,
            message=f"Indexed {chunks_indexed} chunk(s) from {filename} into Weaviate.",
        )
    except ValueError as exc:
        delete_stored_file(stored_file_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rag/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    settings = get_rag_settings()
    top_k = request.top_k or settings.rag_top_k
    documents = _retrieve_documents(request.question, top_k)
    sources = [_document_to_source_chunk(document) for document in documents]

    if not documents:
        return QueryResponse(
            answer="No relevant indexed information was found for this question.",
            sources=sources,
            retrieved_chunk_count=0,
        )

    try:
        answer = generate_answer(request.question, documents, settings)
    except GeneratorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieved_chunk_count=len(documents),
    )


def _index_documents(documents: list) -> tuple[int, int]:
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


def _enrich_documents(
    documents: list,
    stored_file_path: str,
    original_filename: str,
    document_hash: str,
) -> None:
    for document in documents:
        metadata = document.metadata
        filename = metadata.get("filename") or original_filename
        doc_type = metadata.get("doc_type") or "general"
        chunk_index = metadata.get("chunk_index") or 0
        metadata["stored_file_path"] = stored_file_path
        metadata["original_filename"] = original_filename
        metadata["document_hash"] = document_hash
        metadata["content_hash"] = content_hash_from_text(
            document.page_content,
            str(filename),
            str(doc_type),
            int(chunk_index),
        )


def _duplicate_ingest_response(existing: dict) -> IngestResponse:
    skipped = existing.get("chunk_count") or 1
    source_id = str(existing["source_id"])
    return IngestResponse(
        source_id=source_id,
        chunks_indexed=0,
        chunks_skipped=int(skipped),
        duplicate=True,
        stored_file_path=existing.get("stored_file_path"),
        message=(
            "Document already exists in Weaviate; no duplicate file was saved "
            f"and no duplicate chunks were indexed. Existing source_id: {source_id}."
        ),
    )


def _with_file_availability(summary: dict) -> dict:
    stored_file_path = summary.get("stored_file_path")
    path = resolve_stored_path(stored_file_path)
    summary["original_file_available"] = bool(path and path.exists() and path.is_file())
    return summary
