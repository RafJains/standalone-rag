# Rohan Standalone RAG

This is a standalone, domain-neutral RAG service for ingesting, indexing,
retrieving, and later generating answers from private documents and general
knowledge sources. It has its own Dockerfile, Docker Compose file, requirements,
environment example, API, and documentation.

Rohan stack:

- Document parsing: Docling
- Embeddings: `BAAI/bge-m3`
- Vector search: Weaviate
- Workflow: LangChain
- Generator later: Ministral 3 8B/14B initially, then Mistral Small 4 Open
  through vLLM for production

Phase 2 is retrieval-only. `/rag/query` retrieves chunks from Weaviate and
returns sources. It does not call a generator LLM yet.

## Start

```bash
cd RAG
docker compose up -d --build
```

API host port: `8090`

Weaviate host port: `8082`

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

## Validate Stack

```bash
python scripts/validate_stack.py
```
