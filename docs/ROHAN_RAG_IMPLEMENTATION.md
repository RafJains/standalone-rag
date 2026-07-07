# Rohan Standalone RAG Implementation

## Phase 5 Architecture

This RAG service is a separate project under `RAG/`. It is not wired into the
any external application codebase, prompts, tools, Dockerfile, requirements,
compose file, or environment file.

Phase 6 keeps the RAG flow and adds saved originals, duplicate prevention,
download support, clean delete, metadata-filtered retrieval, selectable vector
or hybrid retrieval mode, richer source metadata, document processing upgrades,
upload validation, processing diagnostics, retrieval evaluation, and a small
FastAPI-served frontend. Phase 6.5 migrates query orchestration to LangGraph
while preserving the existing API and frontend behavior. Phase 7 removes the
retired chain framework from project code and dependencies, keeps LangGraph as
the workflow layer, keeps Weaviate retrieval, keeps `BAAI/bge-m3` embeddings,
and keeps the local OpenAI-compatible Ministral generator endpoint. Production
Backend Hardening is now Phase 8, adding API key authentication and project
isolation without adding services or changing the generic collection name. Phase
9 adds automated tests and a repeatable reliability check runner.

RAG flow:

1. Text or files are accepted by `app.api`.
2. Protected endpoints optionally require `X-RAG-API-Key` when
   `RAG_REQUIRE_API_KEY=true`.
3. The API validates `project_id`, using `RAG_DEFAULT_PROJECT_ID` when omitted.
4. The API creates a `source_id`, computes a document hash, and checks Weaviate
   for an existing document in the same project before saving a new copy.
5. Original uploads and pasted text are saved under
   `data/uploads/<source_id>/<safe_filename>`.
6. `app.document_loader` validates and parses supported documents: PDF, TXT,
   Markdown, DOCX, and CSV.
7. Docling `HybridChunker` is used when available. A conservative text chunking
   fallback is used for `.txt`, PDFs via `pypdf`, and Docling API differences.
8. Each chunk receives `project_id`, storage metadata, parser metadata,
   `document_hash`, and `content_hash`.
9. `app.embeddings` embeds new chunks with `BAAI/bge-m3`.
10. `app.vector_store` writes external vectors and metadata into Weaviate.
11. `app.vector_store.similarity_search` retrieves the most relevant chunks
    using vector search by default, mandatory project filtering, optional
    metadata filters, and optional hybrid search when supported by Weaviate.
12. `app.rag_graph` runs a LangGraph workflow with `retrieve`, `decide`,
    `generate`, and `fallback` nodes.
13. `app.generator` keeps the grounded prompt and calls the configured local
    OpenAI-compatible generator endpoint directly.
14. `/rag/query` returns the generated answer, sources, retrieved chunk count,
    retrieval mode, and applied filters.

Storage layout:

```text
data/
  uploads/
    <source_id>/
      <safe_filename>
  demo_documents/
```

Original files are normal files in `data/uploads`. Indexed chunks, vectors, and
metadata are stored in Weaviate in the generic `KnowledgeBase` collection. The
two are intentionally separate: deleting a stored original file alone would not
remove indexed chunks, and deleting chunks alone would not remove a saved file.

Document-management flow:

1. `GET /rag/documents` reads the generic `KnowledgeBase` collection, filters by
   project, and groups chunks by `source_id`.
2. Each summary includes project ID, filename, original filename, document type,
   chunk count, available page numbers, preview, stored path, original-file
   availability, and document hash when available.
3. `GET /rag/documents/{source_id}/download` returns the saved original file when
   it still exists and the source belongs to the requested project.
4. `DELETE /rag/documents/{source_id}` deletes matching chunks from the requested
   project in Weaviate, removes saved originals listed in metadata, removes an empty
   `source_id` folder when possible, and returns deleted chunk and file counts.
   Missing source IDs return `deleted_count: 0`.

Frontend flow:

1. `GET /` serves `app/static/index.html`.
2. `/static/styles.css` and `/static/app.js` provide the vanilla browser UI.
3. The browser calls the same local FastAPI endpoints; there is no separate
   frontend service, package manager, CDN, or build step.

## Rohan Stack

- Document parsing: Docling
- Embeddings: `BAAI/bge-m3`
- Vector search: Weaviate
- Workflow: LangGraph
- Initial generator: Ministral 3 8B or 14B
- Later generator option: Mistral Small 4 Open through vLLM

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

The console has panels for service status, settings, file upload, text
ingestion, indexed documents, and question answering. It is vanilla HTML, CSS,
and JavaScript in `app/static`. Settings are stored in localStorage as
`rag_api_key` and `rag_project_id`.

To upload documents, use the Upload Document panel with a supported file and a
`doc_type` such as `general`. Supported extensions are `.pdf`, `.txt`, `.md`,
`.docx`, and `.csv`. The panel displays the configured maximum upload size.
Requests include the current project ID and the API key header when one is saved.

To ingest pasted text, use the Ingest Text panel with a filename and `doc_type`.

To ask questions, use the Ask a Question panel. Set `top_k`, choose `vector` or
`hybrid`, and optionally enter `doc_type`, `filename`, or `source_id` filters.
The response shows the answer, retrieval mode, applied filters, and source
chunks with filename, document type, source ID, chunk index, page number,
distance or retrieval score, and a contained text preview.

To manage documents, refresh the Documents panel. Cards show whether the
original file is available, parser used, detected extension, original size,
warning count, and stored path when present. Large documents show only the first
12 page numbers in the card with a remainder count, while the API response still
keeps the full `page_numbers` list. Click Download to open
`/rag/documents/{source_id}/download`, or click Delete to remove indexed chunks
and any saved original file for that source.

## Health Check

```bash
curl http://localhost:8090/health
```

`/health` is lightweight and should return even if Weaviate is unavailable.

## Status Check

```bash
curl http://localhost:8090/rag/status
```

`/rag/status` reports the standalone RAG config summary, whether Weaviate is
reachable, whether API key auth is required, and the default project ID. It does
not load the embedding model or expose the API key.

## Authentication

Default local settings:

```bash
RAG_REQUIRE_API_KEY=false
RAG_API_KEY=dev-rag-key
RAG_DEFAULT_PROJECT_ID=default
```

When `RAG_REQUIRE_API_KEY=true`, the protected endpoints require
`X-RAG-API-Key`. Missing or wrong keys return HTTP 401, and comparison uses a
constant-time check. Public endpoints remain `GET /`, `HEAD /`, static assets,
`GET /health`, and `GET /rag/status`.

Auth-enabled local check:

```bash
RAG_REQUIRE_API_KEY=true RAG_API_KEY=dev-rag-key docker compose up -d --build
curl -i "http://localhost:8090/rag/documents"
curl -i "http://localhost:8090/rag/documents" -H "X-RAG-API-Key: wrong-key"
curl -i "http://localhost:8090/rag/documents" -H "X-RAG-API-Key: dev-rag-key"
docker compose up -d --build
```

The first two protected requests should return HTTP 401. The correct key should
return HTTP 200. The final command restores default non-required auth mode.

## Project Isolation

Project IDs are validated with a simple letters, numbers, dash, and underscore
pattern. Omitted values use `RAG_DEFAULT_PROJECT_ID`.

All new chunks store `project_id` in Weaviate. Query, list, delete, and download
lookups are scoped to that project. Duplicate document and chunk checks are also
project-aware, so the same document can exist independently in separate projects.

## Text Ingestion

```bash
curl -X POST "http://localhost:8090/rag/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{"text":"The refund policy allows cancellation within 7 days of purchase. Support is available from 9 AM to 6 PM.","filename":"test-policy.txt","doc_type":"general","project_id":"default"}'
```

Pasted text is saved as a `.txt` file under `data/uploads/<source_id>/`. If no
filename is supplied, the API uses `manual-note.txt`.

## File Ingestion

```bash
curl -X POST "http://localhost:8090/rag/ingest/file" \
  -F "file=@/path/to/sample-document.pdf" \
  -F "doc_type=general" \
  -F "project_id=default"
```

File uploads are validated before saving. Allowed extensions are `.pdf`, `.txt`,
`.md`, `.docx`, and `.csv`. `RAG_MAX_UPLOAD_MB` controls maximum upload size and
defaults to `25`. Empty files, unsupported extensions, and oversized files return
clear HTTP errors.

Valid uploads are saved under `data/uploads/<source_id>/` using a sanitized
filename, then parsed from that saved path.

Parser behavior:

- PDF uses Docling first and keeps the existing PDF text fallback.
- TXT uses safe text decoding.
- Markdown uses safe text decoding and preserves headings as `section_title`
  metadata when possible.
- DOCX is parsed as text without adding any external service.
- CSV is converted into readable row text. Empty rows are skipped and reported in
  `warnings`.

OCR is not implemented yet. Scanned image PDFs are future work unless the
current parser path can already extract text.

Ingestion responses keep the existing fields and add processing diagnostics:

```json
{
  "parser_used": "markdown_text",
  "warnings": [],
  "original_file_size_bytes": 2048,
  "detected_extension": ".md"
}
```

Chunks may include `section_title`, CSV `row_number`, `page_number`, and
`chunk_char_count`. Existing Weaviate collections are updated safely with these
additional generic metadata properties.

## Duplicate Prevention

File ingestion computes `document_hash` from file bytes plus `doc_type`. Text
ingestion computes `document_hash` from normalized text plus filename and
`doc_type`. If that hash already exists in Weaviate for the same project, the API returns the
existing `source_id`, `chunks_indexed: 0`, `duplicate: true`, and skips saving
another original file.

Chunk-level prevention uses `content_hash`, computed from normalized chunk text,
filename, `doc_type`, and `chunk_index`. Existing chunk hashes are skipped and
reported as `chunks_skipped` within the same project.

## Retrieval Query

```bash
curl -X POST "http://localhost:8090/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","top_k":5,"retrieval_mode":"vector","project_id":"default"}'
```

When the configured local generator server is running, `/rag/query` returns a
generated answer grounded in the retrieved chunks. The response includes
`sources`, `retrieved_chunk_count`, `retrieval_mode`, and `filters_applied`.

The query path is orchestrated by LangGraph. The `retrieve` node calls Weaviate
vector or hybrid search with filters, `decide` routes based on whether chunks
were found, `generate` calls the local OpenAI-compatible Ministral endpoint, and
`fallback` returns the no-information answer without calling the generator.

`top_k` controls the maximum number of chunks retrieved before answer
generation. `retrieval_mode` defaults to `vector`, which embeds the question and
uses Weaviate vector similarity search. The `hybrid` mode attempts Weaviate
hybrid search by combining the query text with the query vector. If hybrid
search is not safely available with the current Weaviate client or collection,
the API returns HTTP 400 with `Hybrid retrieval is not available in the current
configuration.`

Optional filters:

```json
{
  "question": "What is the refund policy?",
  "top_k": 3,
  "retrieval_mode": "vector",
  "project_id": "default",
  "doc_type": "general",
  "filename": "policy.txt",
  "source_id": "example-source-id"
}
```

Multiple filters use AND semantics and are pushed into the Weaviate retrieval
query. The project filter is always included. If filters produce no matching
chunks, `/rag/query` returns HTTP 200 with the no-information answer,
`retrieved_chunk_count: 0`, and `sources: []`; the generator is not called.

Source entries preserve project ID, filename, document type, source ID, chunk
index, page number, text, and vector distance. They also include `content_hash`,
`document_hash`, `stored_file_path`, and `retrieval_score` when available.
Smaller vector distances are closer matches. Hybrid retrieval scores are
returned from Weaviate when the client provides them.

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
      "original_filename": "sample-policy.txt",
      "doc_type": "general",
      "chunk_count": 1,
      "page_numbers": [],
      "preview": "The refund policy allows cancellation within 7 days...",
      "stored_file_path": "data/uploads/example-source-id/sample-policy.txt",
      "original_file_available": true,
      "document_hash": "example-hash"
    }
  ]
}
```

Delete all chunks for a source:

```bash
curl -X DELETE "http://localhost:8090/rag/documents/<SOURCE_ID>"
```

Download a saved original file:

```bash
curl -I "http://localhost:8090/rag/documents/<SOURCE_ID>/download"
```

Older indexed chunks without `stored_file_path` remain compatible. They list
with `original_file_available: false`, delete still removes their Weaviate
chunks, and download returns 404 with a clear message.

## Retrieval Evaluation

## Automated Tests And Quality Checks

Fast local checks do not require live Weaviate or the local generator:

```bash
python3 -m pytest -q
python3 scripts/run_quality_checks.py
```

The pytest suite covers auth, project ID validation, Pydantic schemas, generator
request and error handling, document loading, LangGraph routing, and static stack
contracts. The quality runner executes stack validation, Python compilation,
pytest, and the Compose service contract.

## Retrieval Evaluation

With the stack and generator running, execute:

```bash
python3 scripts/evaluate_retrieval.py
```

The script uses `http://localhost:8090`, ingests small generic text documents,
accepts duplicate ingest responses, runs vector retrieval tests with `top_k=3`,
verifies `doc_type` metadata filtering, verifies the no-match filter response,
and tests hybrid mode when supported. A clear hybrid HTTP 400 is reported as
unavailable and is acceptable. The script exits non-zero only when required
vector or filter tests fail, or when hybrid fails in a way other than the
documented unavailable response.

## Processing Evaluation

With the stack and generator running, execute:

```bash
python3 scripts/evaluate_processing.py
```

The script uses `http://localhost:8090`, creates temporary TXT, MD, CSV, and
DOCX samples, uploads them through `/rag/ingest/file`, verifies that processing
diagnostics are present, accepts duplicate upload responses, and runs a hybrid
query with a filename filter.

## Auth And Project Evaluation

With the stack and generator running, execute:

```bash
python3 scripts/evaluate_auth_projects.py
```

The script uses `http://localhost:8090`, ingests one text document into project
`alpha` and one into project `beta`, verifies query isolation, verifies document
list isolation, deletes the alpha source under project `alpha`, and confirms the
beta document still exists.

The live evaluation scripts exercise the API, Weaviate, embeddings, and the
configured generator. They complement the pytest suite, which is intentionally
fast and isolated from live services.

## Demo Documents

`data/demo_documents` is for local sample PDFs or TXT files. Move sample files
there manually, then upload through the UI or `POST /rag/ingest/file` when you
want them indexed. Files in this directory are ignored by Git except for
`.gitkeep`.

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
