from typing import Any
from urllib.parse import urlparse

import weaviate
from langchain_core.documents import Document
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import WeaviateBaseError

from app.config import get_rag_settings
from app.embeddings import get_embeddings


COLLECTION_PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="source_id", data_type=DataType.TEXT),
    Property(name="filename", data_type=DataType.TEXT),
    Property(name="original_filename", data_type=DataType.TEXT),
    Property(name="doc_type", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="page_number", data_type=DataType.INT),
    Property(name="stored_file_path", data_type=DataType.TEXT),
    Property(name="document_hash", data_type=DataType.TEXT),
    Property(name="content_hash", data_type=DataType.TEXT),
]

DOCUMENT_RETURN_PROPERTIES = [prop.name for prop in COLLECTION_PROPERTIES]


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
            _ensure_collection_properties(active_client)
            return
        active_client.collections.create(
            name=settings.weaviate_collection,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=COLLECTION_PROPERTIES,
        )
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not ensure Weaviate collection: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()


def add_documents(documents: list[Document]) -> tuple[int, int]:
    non_empty_documents = [doc for doc in documents if doc.page_content.strip()]
    if not non_empty_documents:
        return 0, len(documents)

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        documents_to_index = [
            document
            for document in non_empty_documents
            if not _content_hash_exists(collection, document.metadata.get("content_hash"))
        ]
        skipped_count = len(documents) - len(documents_to_index)
        if not documents_to_index:
            return 0, skipped_count

        embeddings = get_embeddings()
        texts = [doc.page_content for doc in documents_to_index]
        vectors = embeddings.embed_documents(texts)

        with collection.batch.dynamic() as batch:
            for document, vector in zip(documents_to_index, vectors, strict=True):
                properties = _document_properties(document)
                batch.add_object(properties=properties, vector=vector)

        failed_count = len(getattr(collection.batch, "failed_objects", []) or [])
        if failed_count:
            raise RuntimeError(f"Weaviate failed to index {failed_count} chunk(s).")
        return len(documents_to_index), skipped_count
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


def list_documents() -> list[dict[str, Any]]:
    client = get_weaviate_client()
    documents_by_source: dict[str, dict[str, Any]] = {}

    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        for item in collection.iterator(
            return_properties=DOCUMENT_RETURN_PROPERTIES,
        ):
            properties = item.properties or {}
            source_id = properties.get("source_id")
            if not source_id:
                continue

            source_key = str(source_id)
            summary = documents_by_source.setdefault(
                source_key,
                {
                    "source_id": source_key,
                    "filename": properties.get("filename"),
                    "original_filename": properties.get("original_filename"),
                    "doc_type": properties.get("doc_type"),
                    "chunk_count": 0,
                    "page_numbers": set(),
                    "preview": None,
                    "stored_file_path": properties.get("stored_file_path"),
                    "original_file_available": False,
                    "document_hash": properties.get("document_hash"),
                    "_first_chunk_index": None,
                },
            )
            summary["chunk_count"] += 1
            if not summary.get("filename") and properties.get("filename"):
                summary["filename"] = properties.get("filename")
            if not summary.get("original_filename") and properties.get("original_filename"):
                summary["original_filename"] = properties.get("original_filename")
            if not summary.get("doc_type") and properties.get("doc_type"):
                summary["doc_type"] = properties.get("doc_type")
            if not summary.get("stored_file_path") and properties.get("stored_file_path"):
                summary["stored_file_path"] = properties.get("stored_file_path")
            if not summary.get("document_hash") and properties.get("document_hash"):
                summary["document_hash"] = properties.get("document_hash")

            page_number = _int_or_none(properties.get("page_number"))
            if page_number is not None:
                summary["page_numbers"].add(page_number)

            chunk_index = _int_or_none(properties.get("chunk_index"))
            text = (properties.get("text") or "").strip()
            first_chunk_index = summary.get("_first_chunk_index")
            if text and (first_chunk_index is None or chunk_index is None or chunk_index < first_chunk_index):
                summary["preview"] = _preview_text(text)
                summary["_first_chunk_index"] = chunk_index

        summaries = []
        for summary in documents_by_source.values():
            summary["page_numbers"] = sorted(summary["page_numbers"])
            summary["original_file_available"] = bool(summary.get("stored_file_path"))
            summary.pop("_first_chunk_index", None)
            summaries.append(summary)
        return sorted(summaries, key=lambda item: (item.get("filename") or "", item["source_id"]))
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not list documents from Weaviate: {exc}") from exc
    finally:
        client.close()


def get_document_by_hash(document_hash: str) -> dict[str, Any] | None:
    if not document_hash:
        return None

    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        response = collection.query.fetch_objects(
            filters=Filter.by_property("document_hash").equal(document_hash),
            limit=1,
            return_properties=DOCUMENT_RETURN_PROPERTIES,
        )
        if not response.objects:
            return None

        properties = response.objects[0].properties or {}
        source_id = properties.get("source_id")
        if not source_id:
            return None
        return {
            "source_id": str(source_id),
            "filename": properties.get("filename"),
            "stored_file_path": properties.get("stored_file_path"),
            "document_hash": document_hash,
            "chunk_count": count_documents_by_source_id(str(source_id)),
        }
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not check document hash in Weaviate: {exc}") from exc
    finally:
        client.close()


def get_document_storage(source_id: str) -> dict[str, Any]:
    client = get_weaviate_client()
    stored_paths: set[str] = set()
    exists = False
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        for item in collection.iterator(return_properties=DOCUMENT_RETURN_PROPERTIES):
            properties = item.properties or {}
            if str(properties.get("source_id") or "") != source_id:
                continue
            exists = True
            stored_file_path = properties.get("stored_file_path")
            if stored_file_path:
                stored_paths.add(str(stored_file_path))
        return {"exists": exists, "stored_file_paths": sorted(stored_paths)}
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not inspect document storage in Weaviate: {exc}") from exc
    finally:
        client.close()


def count_documents_by_source_id(source_id: str) -> int:
    client = get_weaviate_client()
    count = 0
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        for item in collection.iterator(return_properties=["source_id"]):
            properties = item.properties or {}
            if str(properties.get("source_id") or "") == source_id:
                count += 1
        return count
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not count documents in Weaviate: {exc}") from exc
    finally:
        client.close()


def delete_documents_by_source_id(source_id: str) -> int:
    client = get_weaviate_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(get_rag_settings().weaviate_collection)
        result = collection.data.delete_many(where=Filter.by_property("source_id").equal(source_id))
        return _deleted_count(result)
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Could not delete documents from Weaviate: {exc}") from exc
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
        "original_filename": _string_or_none(metadata.get("original_filename")),
        "doc_type": _string_or_none(metadata.get("doc_type")),
        "chunk_index": _int_or_none(metadata.get("chunk_index")),
        "page_number": _int_or_none(metadata.get("page_number")),
        "stored_file_path": _string_or_none(metadata.get("stored_file_path")),
        "document_hash": _string_or_none(metadata.get("document_hash")),
        "content_hash": _string_or_none(metadata.get("content_hash")),
    }


def _object_to_document(item: Any) -> Document:
    properties = item.properties or {}
    metadata = {
        "source_id": properties.get("source_id"),
        "filename": properties.get("filename"),
        "original_filename": properties.get("original_filename"),
        "doc_type": properties.get("doc_type"),
        "chunk_index": properties.get("chunk_index"),
        "page_number": properties.get("page_number"),
        "stored_file_path": properties.get("stored_file_path"),
        "document_hash": properties.get("document_hash"),
        "content_hash": properties.get("content_hash"),
    }
    distance = getattr(getattr(item, "metadata", None), "distance", None)
    if distance is not None:
        metadata["distance"] = distance
    return Document(page_content=properties.get("text") or "", metadata=metadata)


def _ensure_collection_properties(client: weaviate.WeaviateClient) -> None:
    collection = client.collections.get(get_rag_settings().weaviate_collection)
    existing_names = _collection_property_names(collection)
    for prop in COLLECTION_PROPERTIES:
        if prop.name in existing_names:
            continue
        try:
            collection.config.add_property(prop)
        except WeaviateBaseError as exc:
            if "already" not in str(exc).lower():
                raise


def _collection_property_names(collection: Any) -> set[str]:
    try:
        config = collection.config.get()
    except Exception:
        return set()

    properties = getattr(config, "properties", None)
    if isinstance(properties, dict):
        return {str(name) for name in properties}
    if isinstance(properties, list):
        names: set[str] = set()
        for prop in properties:
            name = getattr(prop, "name", None)
            if name:
                names.add(str(name))
        return names
    return set()


def _content_hash_exists(collection: Any, content_hash: Any) -> bool:
    if not content_hash:
        return False
    response = collection.query.fetch_objects(
        filters=Filter.by_property("content_hash").equal(str(content_hash)),
        limit=1,
        return_properties=["content_hash"],
    )
    return bool(response.objects)


def _deleted_count(result: Any) -> int:
    for attribute in ("successful", "matches"):
        value = getattr(result, attribute, None)
        if isinstance(value, int):
            return value

    if isinstance(result, dict):
        for key in ("successful", "matches"):
            value = result.get(key)
            if isinstance(value, int):
                return value

    return 0


def _preview_text(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


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
