from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pypdf import PdfReader


FALLBACK_CHUNK_SIZE = 1000
FALLBACK_CHUNK_OVERLAP = 150


def load_text_as_documents(
    text: str,
    source_id: str,
    filename: str | None,
    doc_type: str,
) -> list[Document]:
    return _documents_from_text(
        text=text,
        source_id=source_id,
        filename=filename,
        doc_type=doc_type,
        page_number=None,
    )


def load_file_as_documents(
    file_path: str,
    source_id: str,
    filename: str,
    doc_type: str,
) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = _read_text_file(path)
        return _documents_from_text(text, source_id, filename, doc_type, page_number=None)

    docling_error: Exception | None = None
    try:
        documents = _load_with_docling(path, source_id, filename, doc_type)
        if documents:
            return documents
    except Exception as exc:
        docling_error = exc

    if suffix == ".pdf":
        try:
            return _load_pdf_fallback(path, source_id, filename, doc_type)
        except Exception as exc:
            raise ValueError(
                f"Could not parse PDF with Docling or pypdf fallback: {exc}"
            ) from exc

    supported = ".txt and .pdf"
    if docling_error is not None:
        raise ValueError(
            f"Could not parse {filename} with Docling. Supported fallback formats are {supported}. "
            f"Docling error: {docling_error}"
        ) from docling_error

    raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Supported formats: {supported}.")


def _load_with_docling(
    path: Path,
    source_id: str,
    filename: str,
    doc_type: str,
) -> list[Document]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    docling_document = getattr(result, "document", None)
    if docling_document is None:
        raise ValueError("Docling did not return a document object.")

    chunks = _chunk_docling_document(docling_document)
    if chunks:
        documents: list[Document] = []
        for index, chunk in enumerate(chunks):
            text = _extract_chunk_text(chunk)
            if not text:
                continue
            metadata = _base_metadata(source_id, filename, doc_type, index)
            page_number = _extract_page_number(chunk)
            if page_number is not None:
                metadata["page_number"] = page_number
            documents.append(Document(page_content=text, metadata=metadata))
        if documents:
            return documents

    exported_text = _export_docling_text(docling_document)
    return _documents_from_text(exported_text, source_id, filename, doc_type, page_number=None)


def _chunk_docling_document(docling_document: Any) -> list[Any]:
    try:
        from docling.chunking import HybridChunker
    except Exception:
        try:
            from docling.chunking.hybrid_chunker import HybridChunker
        except Exception:
            return []

    chunker = HybridChunker()
    try:
        return list(chunker.chunk(dl_doc=docling_document))
    except TypeError:
        return list(chunker.chunk(docling_document))


def _extract_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk.strip()
    for attr in ("text", "page_content"):
        value = getattr(chunk, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    export_text = getattr(chunk, "export_text", None)
    if callable(export_text):
        value = export_text()
        if isinstance(value, str):
            return value.strip()
    return str(chunk).strip()


def _extract_page_number(chunk: Any) -> int | None:
    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("page_number", "page", "page_no"):
            value = metadata.get(key)
            if isinstance(value, int):
                return value
    return None


def _export_docling_text(docling_document: Any) -> str:
    for method_name in ("export_to_markdown", "export_to_text"):
        method = getattr(docling_document, method_name, None)
        if callable(method):
            text = method()
            if isinstance(text, str) and text.strip():
                return text
    raise ValueError("Docling parsed the document but could not export text.")


def _load_pdf_fallback(
    path: Path,
    source_id: str,
    filename: str,
    doc_type: str,
) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []
    chunk_index = 0
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in _split_text(text):
            metadata = _base_metadata(source_id, filename, doc_type, chunk_index)
            metadata["page_number"] = page_index
            documents.append(Document(page_content=chunk, metadata=metadata))
            chunk_index += 1
    if not documents:
        raise ValueError("No text could be extracted from the PDF.")
    return documents


def _documents_from_text(
    text: str,
    source_id: str,
    filename: str | None,
    doc_type: str,
    page_number: int | None,
) -> list[Document]:
    documents: list[Document] = []
    for index, chunk in enumerate(_split_text(text)):
        metadata = _base_metadata(source_id, filename, doc_type, index)
        if page_number is not None:
            metadata["page_number"] = page_number
        documents.append(Document(page_content=chunk, metadata=metadata))
    return documents


def _split_text(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + FALLBACK_CHUNK_SIZE, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - FALLBACK_CHUNK_OVERLAP, start + 1)
    return chunks


def _base_metadata(
    source_id: str,
    filename: str | None,
    doc_type: str,
    chunk_index: int,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "filename": filename,
        "doc_type": doc_type,
        "chunk_index": chunk_index,
    }


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
