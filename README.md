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

Phase 4 adds a simple FastAPI-served browser console and document-management
endpoints. The existing ingestion, status, and query endpoints remain unchanged.

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
- `DELETE /rag/documents/{source_id}` to delete all chunks for a document
- `POST /rag/query` for question answering

To upload a document, choose a PDF or TXT file, keep or change the `doc_type`,
and click Upload. To ingest text, paste text, set a filename and `doc_type`, and
click Ingest Text. To ask questions, enter a question and `top_k`; answers show
the retrieved sources. To delete a document, refresh the document list and click
Delete for the matching `source_id`.

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
