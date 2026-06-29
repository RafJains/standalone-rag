from typing import Any
from urllib.parse import urlparse

import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery
from weaviate.exceptions import WeaviateBaseError

from app.config import get_rag_settings
from app.embeddings import get_embeddings


def get_weaviate_client() -> weaviate.WeaviateClient:
    settings = get_rag_settings()
    parsed = urlparse(settings.weaviate_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid WEAVIATE_URL: {settings.weaviate_url}")

    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    grpc_port = 50051

    try:
        client = weaviate.connect_to_custom(
            http_host=parsed.hostname,
            http_port=http_port,
            http_secure=parsed.scheme == "https",
            grpc_host=parsed.hostname,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
        if not client.is_ready():
            raise ConnectionError(f"Weaviate is not ready at {settings.weaviate_url}")
        return client
    except Exception as exc:
        raise ConnectionError(f"Weaviate is unavailable at {settings.weaviate_url}: {exc}") from exc


def ensure_collection(client: weaviate.WeaviateClient | None = None) -> None:
    owns_client = client is None
    active_client = client or get_weaviate_client()
    settings = get_rag_settings()

    try:
        if active_client.collections.exists(settings.weaviate_collection):
            return
        active_client.collections.create(
            name=settings.weaviate_collection,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source_id", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="page_number", data_type=DataType.INT),
            ],
        )
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not ensure Weaviate collection: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()


def add_documents(documents: list[Document]) -> int:
    non_empty_documents = [doc for doc in documents if doc.page_content.strip()]
    if not non_empty_documents:
        return 0

    embeddings = get_embeddings()
    texts = [doc.page_content for doc in non_empty_documents]
    vectors = embeddings.embed_documents(texts)

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        with collection.batch.dynamic() as batch:
            for document, vector in zip(non_empty_documents, vectors, strict=True):
                properties = _document_properties(document)
                batch.add_object(properties=properties, vector=vector)

        failed_count = len(getattr(collection.batch, "failed_objects", []) or [])
        if failed_count:
            raise RuntimeError(f"Weaviate failed to index {failed_count} chunk(s).")
        return len(non_empty_documents)
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not index documents in Weaviate: {exc}") from exc
    finally:
        client.close()


def similarity_search(query: str, top_k: int) -> list[Document]:
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [_object_to_document(item) for item in response.objects]
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not retrieve documents from Weaviate: {exc}") from exc
    finally:
        client.close()


def check_weaviate_ready() -> tuple[bool, str]:
    try:
        client = get_weaviate_client()
    except Exception as exc:
        return False, str(exc)

    try:
        ensure_collection(client)
        return True, "Weaviate is reachable and collection is available."
    except Exception as exc:
        return False, str(exc)
    finally:
        client.close()


def _document_properties(document: Document) -> dict[str, Any]:
    metadata = document.metadata
    return {
        "text": document.page_content,
        "source_id": _string_or_none(metadata.get("source_id")),
        "filename": _string_or_none(metadata.get("filename")),
        "doc_type": _string_or_none(metadata.get("doc_type")),
        "chunk_index": _int_or_none(metadata.get("chunk_index")),
        "page_number": _int_or_none(metadata.get("page_number")),
    }


def _object_to_document(item: Any) -> Document:
    properties = item.properties or {}
    metadata = {
        "source_id": properties.get("source_id"),
        "filename": properties.get("filename"),
        "doc_type": properties.get("doc_type"),
        "chunk_index": properties.get("chunk_index"),
        "page_number": properties.get("page_number"),
    }
    distance = getattr(getattr(item, "metadata", None), "distance", None)
    if distance is not None:
        metadata["distance"] = distance
    return Document(page_content=properties.get("text") or "", metadata=metadata)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
