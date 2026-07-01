# Rohan Standalone RAG Implementation

## Phase 4 Architecture

This RAG service is a separate project under `RAG/`. It is not wired into the
any external application codebase, prompts, tools, Dockerfile, requirements,
compose file, or environment file.

Phase 4 keeps the Phase 3 RAG flow and adds a small FastAPI-served frontend plus
document listing and deletion APIs.

RAG flow:

1. Text or files are accepted by `app.api`.
2. `app.document_loader` parses documents with Docling when possible.
3. Docling `HybridChunker` is used when available. A conservative text chunking
   fallback is used for `.txt`, PDFs via `pypdf`, and Docling API differences.
4. `app.embeddings` embeds chunks with `BAAI/bge-m3`.
5. `app.vector_store` writes external vectors and metadata into Weaviate.
6. `app.vector_store.similarity_search` retrieves the most relevant chunks.
7. `app.rag_chain` builds a grounded prompt and calls the configured local
   OpenAI-compatible generator endpoint through LangChain.
8. `/rag/query` returns the generated answer, sources, and retrieved chunk count.

Document-management flow:

1. `GET /rag/documents` reads the generic `KnowledgeBase` collection and groups
   chunks by `source_id`.
2. Each summary includes filename, document type, chunk count, available page
   numbers, and a short preview when text is available.
3. `DELETE /rag/documents/{source_id}` deletes all matching chunks from
   Weaviate and returns the number deleted. Missing source IDs return
   `deleted_count: 0`.

Frontend flow:

1. `GET /` serves `app/static/index.html`.
2. `/static/styles.css` and `/static/app.js` provide the vanilla browser UI.
3. The browser calls the same local FastAPI endpoints; there is no separate
   frontend service, package manager, CDN, or build step.

## Rohan Stack

- Document parsing: Docling
- Embeddings: `BAAI/bge-m3`
- Vector search: Weaviate
- Workflow: LangChain
- Initial generator: Ministral 3 8B or 14B
- Production generator: Mistral Small 4 Open through vLLM

## Startup

```bash
cd RAG
docker compose up -d --build
```

This starts only `rohan-rag-weaviate` and `rohan-rag-api`.

## Frontend Console

Open:

```text
http://localhost:8090
```

The console has panels for service status, file upload, text ingestion, indexed
documents, and question answering.

To upload documents, use the Upload Document panel with a PDF or TXT file and a
`doc_type` such as `general`.

To ingest pasted text, use the Ingest Text panel with a filename and `doc_type`.

To ask questions, use the Ask a Question panel. The response shows the answer
and source chunks with filename, document type, chunk index, page number,
distance, and text preview.

To delete documents, refresh the Documents panel and click Delete on the target
document. The UI confirms before calling `DELETE /rag/documents/{source_id}`.

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

When the configured local generator server is running, `/rag/query` returns a
generated answer grounded in the retrieved chunks. The response includes
`sources` and `retrieved_chunk_count`.

If no relevant indexed information is found, `/rag/query` returns HTTP 200 with
a concise no-information answer and no source chunks.

If the generator server is not running or cannot be reached, `/rag/query`
returns HTTP 503 with a clear message that the local generator LLM is
unavailable. The API should not crash or restart for this condition.

## Document Management

List indexed documents:

```bash
curl http://localhost:8090/rag/documents
```

Example response:

```json
{
  "documents": [
    {
      "source_id": "example-source-id",
      "filename": "sample-policy.txt",
      "doc_type": "general",
      "chunk_count": 1,
      "page_numbers": [],
      "preview": "The refund policy allows cancellation within 7 days..."
    }
  ]
}
```

Delete all chunks for a source:

```bash
curl -X DELETE "http://localhost:8090/rag/documents/<SOURCE_ID>"
```

## Generator Configuration

Default local generator settings:

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
