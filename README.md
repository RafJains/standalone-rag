# Rohan Standalone RAG

This is a standalone, domain-neutral RAG service for ingesting, indexing,
retrieving, and generating answers from private documents and general knowledge
sources. It has its own Dockerfile, Docker Compose file, requirements,
environment example, API, and documentation.

Rohan stack:

- Document parsing: Docling
- Embeddings: `BAAI/bge-m3`
- Vector search: Weaviate
- Workflow: LangChain
- Generator: local OpenAI-compatible Ministral or Mistral endpoint

Phase 4.5 adds saved original files, duplicate prevention, clean delete,
download support, and a simple FastAPI-served browser console. The existing
ingestion, status, and query endpoints remain available.

## Start

```bash
cd RAG
docker compose up -d --build
```

API host port: `8090`

Weaviate host port: `8082`

## Frontend

Open:

```text
http://localhost:8090
```

The frontend is served by FastAPI from `app/static` and uses the same backend
origin. It calls:

- `GET /health` and `GET /rag/status` for status
- `POST /rag/ingest/file` for PDF/TXT uploads
- `POST /rag/ingest/text` for pasted text
- `GET /rag/documents` to list indexed documents
- `GET /rag/documents/{source_id}/download` to download a saved original file
- `DELETE /rag/documents/{source_id}` to delete all chunks for a document
- `POST /rag/query` for question answering

To upload a document, choose a PDF or TXT file, keep or change the `doc_type`,
and click Upload. To ingest text, paste text, set a filename and `doc_type`, and
click Ingest Text. To ask questions, enter a question and `top_k`; answers show
the retrieved sources. To delete a document, refresh the document list and click
Delete for the matching `source_id`. Documents with a saved original file also
show a Download button.

## File Storage

Uploaded files and pasted text are saved under:

```text
data/uploads/<source_id>/<safe_filename>
```

The saved original file is separate from the indexed chunks. Original files live
on disk in `data/uploads`; parsed chunks, embeddings, and metadata live in the
generic Weaviate `KnowledgeBase` collection. Each new chunk includes metadata
such as `source_id`, `filename`, `original_filename`, `stored_file_path`,
`doc_type`, `chunk_index`, optional `page_number`, `document_hash`, and
`content_hash`.

`data/demo_documents` is reserved for local sample material. Move any sample PDF
or TXT files there manually, then upload them through the UI or API when you want
to index them.

User documents in `data/uploads` and `data/demo_documents` are ignored by Git
except for `.gitkeep` files.

## Health

```bash
curl http://localhost:8090/health
```

## Status

```bash
curl http://localhost:8090/rag/status
```

## Text Ingestion

```bash
curl -X POST "http://localhost:8090/rag/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{"text":"The refund policy allows cancellation within 7 days of purchase. Support is available from 9 AM to 6 PM.","filename":"test-policy.txt","doc_type":"general"}'
```

The response includes `source_id`, `chunks_indexed`, `chunks_skipped`,
`duplicate`, `stored_file_path`, and `message`.

## File Ingestion

```bash
curl -X POST "http://localhost:8090/rag/ingest/file" \
  -F "file=@/path/to/sample-document.pdf" \
  -F "doc_type=general"
```

## Query

```bash
curl -X POST "http://localhost:8090/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","top_k":5}'
```

## Documents

List indexed documents grouped by `source_id`:

```bash
curl http://localhost:8090/rag/documents
```

Delete all chunks for a document:

```bash
curl -X DELETE "http://localhost:8090/rag/documents/<SOURCE_ID>"
```

Download the saved original file when available:

```bash
curl -I "http://localhost:8090/rag/documents/<SOURCE_ID>/download"
```

Delete removes all Weaviate chunks for the `source_id`, then removes any saved
original file recorded in chunk metadata. Older indexed documents without
`stored_file_path` can still be listed and deleted, but download returns 404 with
a clear message.

## Duplicate Prevention

File ingestion computes `document_hash` from file bytes plus `doc_type`. Text
ingestion computes `document_hash` from normalized text plus filename and
`doc_type`. If the hash already exists in Weaviate, the API skips saving another
file and returns `duplicate: true`.

Each chunk also gets a `content_hash` from normalized chunk text, filename,
`doc_type`, and `chunk_index`. If an individual chunk hash already exists, that
chunk is skipped and counted in `chunks_skipped`.

## Generator Configuration

```bash
RAG_GENERATOR_BASE_URL=http://host.docker.internal:8001/v1
RAG_GENERATOR_API_KEY=not-needed
RAG_GENERATOR_MODEL=Ministral-3-8B-Instruct
RAG_GENERATOR_TEMPERATURE=0.1
RAG_GENERATOR_MAX_TOKENS=512
RAG_GENERATOR_TIMEOUT_SECONDS=60
```

Production generator example:

```bash
RAG_GENERATOR_BASE_URL=http://vllm:8000/v1
RAG_GENERATOR_MODEL=Mistral-Small-4-Open
```

## Validate Stack

```bash
python scripts/validate_stack.py
```
