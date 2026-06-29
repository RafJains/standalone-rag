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

Phase 3 adds generator answering. `/rag/query` retrieves chunks from Weaviate,
builds a grounded prompt, calls the configured local generator endpoint, and
returns the generated answer with sources. If no relevant chunks are found, the
API returns a clear no-information answer without calling the generator. If the
generator server is not running or cannot be reached, `/rag/query` returns HTTP
503 with a clear local generator unavailable message.

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
