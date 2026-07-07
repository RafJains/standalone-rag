from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_rag_settings
from app.document_loader import ALLOWED_FILE_EXTENSIONS, load_file_as_documents, load_text_as_documents
from app.generator import GeneratorUnavailableError
from app.rag_graph import run_rag_graph
from app.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentSummary,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RagStatusResponse,
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
    HybridRetrievalUnavailableError,
    add_documents,
    check_weaviate_ready,
    delete_documents_by_source_id,
    get_document_by_hash,
    get_document_storage,
    list_documents,
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
        "max_upload_mb": settings.rag_max_upload_mb,
        "allowed_upload_extensions": sorted(ALLOWED_FILE_EXTENSIONS),
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
    text_size_bytes = len(request.text.encode("utf-8"))
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
        parser_used="plain_text",
        warnings=[],
        original_file_size_bytes=text_size_bytes,
        detected_extension=Path(filename).suffix.lower() or ".txt",
    )
    chunks_indexed, chunks_skipped = _index_documents(documents)
    return IngestResponse(
        source_id=source_id,
        chunks_indexed=chunks_indexed,
        chunks_skipped=chunks_skipped,
        duplicate=False,
        stored_file_path=stored_file_path,
        parser_used="plain_text",
        warnings=[],
        original_file_size_bytes=text_size_bytes,
        detected_extension=Path(filename).suffix.lower() or ".txt",
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
    detected_extension = Path(filename).suffix.lower()
    _validate_upload_extension(detected_extension)

    try:
        content = await file.read()
    finally:
        await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="file cannot be empty")
    _validate_upload_size(len(content))

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
        processing = load_file_as_documents(
            file_path=str(stored_path),
            source_id=resolved_source_id,
            filename=filename,
            doc_type=doc_type,
        )
        documents = processing.documents
        if not documents:
            raise HTTPException(status_code=400, detail="file did not produce any chunks")

        _enrich_documents(
            documents=documents,
            stored_file_path=stored_file_path,
            original_filename=original_filename,
            document_hash=document_hash,
            parser_used=processing.parser_used,
            warnings=processing.warnings,
            original_file_size_bytes=len(content),
            detected_extension=processing.detected_extension or detected_extension,
        )
        chunks_indexed, chunks_skipped = _index_documents(documents)
        return IngestResponse(
            source_id=resolved_source_id,
            chunks_indexed=chunks_indexed,
            chunks_skipped=chunks_skipped,
            duplicate=False,
            stored_file_path=stored_file_path,
            parser_used=processing.parser_used,
            warnings=processing.warnings,
            original_file_size_bytes=len(content),
            detected_extension=processing.detected_extension or detected_extension,
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
    filters = _query_filters(request)
    try:
        result = run_rag_graph(
            question=request.question,
            top_k=top_k,
            retrieval_mode=request.retrieval_mode,
            filters=filters,
            settings=settings,
        )
    except HybridRetrievalUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeneratorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        retrieved_chunk_count=result.get("retrieved_chunk_count", 0),
        retrieval_mode=request.retrieval_mode,
        filters_applied=result.get("filters_applied", filters),
    )


def _index_documents(documents: list) -> tuple[int, int]:
    try:
        return add_documents(documents)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _query_filters(request: QueryRequest) -> dict[str, str]:
    filters = {}
    for field in ("doc_type", "filename", "source_id"):
        value = getattr(request, field)
        if value is not None and value.strip():
            filters[field] = value.strip()
    return filters
def _enrich_documents(
    documents: list,
    stored_file_path: str,
    original_filename: str,
    document_hash: str,
    parser_used: str,
    warnings: list[str],
    original_file_size_bytes: int,
    detected_extension: str,
) -> None:
    for document in documents:
        metadata = document.metadata
        filename = metadata.get("filename") or original_filename
        doc_type = metadata.get("doc_type") or "general"
        chunk_index = metadata.get("chunk_index") or 0
        metadata["stored_file_path"] = stored_file_path
        metadata["original_filename"] = original_filename
        metadata["document_hash"] = document_hash
        metadata["parser_used"] = metadata.get("parser_used") or parser_used
        metadata["warnings"] = warnings
        metadata["original_file_size_bytes"] = original_file_size_bytes
        metadata["detected_extension"] = detected_extension
        metadata["chunk_char_count"] = metadata.get("chunk_char_count") or len(document.page_content)
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
        parser_used=existing.get("parser_used"),
        warnings=existing.get("warnings") or [],
        original_file_size_bytes=existing.get("original_file_size_bytes"),
        detected_extension=existing.get("detected_extension"),
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


def _validate_upload_extension(extension: str) -> None:
    if extension not in ALLOWED_FILE_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_FILE_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or 'unknown'}'. Supported formats: {supported}.",
        )


def _validate_upload_size(size_bytes: int) -> None:
    settings = get_rag_settings()
    max_bytes = settings.rag_max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum upload size is {settings.rag_max_upload_mb} MB.",
        )
