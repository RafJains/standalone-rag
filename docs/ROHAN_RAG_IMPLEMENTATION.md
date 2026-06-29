# Rohan Standalone RAG Implementation

## Phase 2 Architecture

This RAG service is a separate project under `RAG/`. It is not wired into the
any external application codebase, prompts, tools, Dockerfile, requirements, compose file, or environment file.

Phase 2 flow:

1. Text or files are accepted by `app.api`.
2. `app.document_loader` parses documents with Docling when possible.
3. Docling `HybridChunker` is used when available. A conservative text chunking
   fallback is used for `.txt`, PDFs via `pypdf`, and Docling API differences.
4. `app.embeddings` embeds chunks with `BAAI/bge-m3`.
5. `app.vector_store` writes external vectors and metadata into Weaviate.
6. Query currently performs retrieval only. Generator LLM integration is Phase 3.

## Rohan Stack

- Document parsing: Docling
- Embeddings: `BAAI/bge-m3`
- Vector search: Weaviate
- Workflow: LangChain
- Initial generator later: Ministral 3 8B/14B
- Production generator later: Mistral Small 4 Open through vLLM

## Startup

```bash
cd RAG
docker compose up -d --build
```

This starts only `rohan-rag-weaviate` and `rohan-rag-api`.

## Health Check

```bash
curl http://localhost:8090/health
```

`/health` is lightweight and should return even if Weaviate is unavailable.

## Status Check

```bash
curl http://localhost:8090/rag/status
```

`/rag/status` reports the standalone RAG config summary and whether Weaviate is
reachable. It does not load the embedding model.

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

## Retrieval Query

```bash
curl -X POST "http://localhost:8090/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","top_k":5}'
```

The response is retrieval-only in Phase 2. It includes matching chunks in
`sources`, but it does not call a generator LLM.

## Troubleshooting

### Weaviate Not Reachable

Run:

```bash
cd RAG
docker compose ps
curl http://localhost:8090/rag/status
```

Inside Docker, the service uses `WEAVIATE_URL=http://weaviate:8080`. The host
port is `8082`.

### First Embedding Model Download Is Slow

The first ingestion or query downloads `BAAI/bge-m3` through
`sentence-transformers`. This can take time and requires enough disk space.

### Docker Build Takes Time

`docling`, `sentence-transformers`, and their transitive dependencies are heavy.
The first build can be slow.

### PDF Parsing Fallback

Docling is tried first for file parsing. If Docling cannot parse a PDF, the API
falls back to `pypdf` text extraction. Scanned image PDFs may produce no text.
