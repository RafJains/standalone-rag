# Rohan Standalone RAG

This is a standalone, domain-neutral RAG service for ingesting, indexing,
retrieving, and generating answers from private documents and general knowledge
sources. It has its own Dockerfile, Docker Compose file, requirements,
environment example, API, and documentation.

Rohan stack:

- Document parsing: Docling
- Embeddings: `BAAI/bge-m3`
- Vector search: Weaviate
- Workflow: LangGraph
- Generator: local OpenAI-compatible Ministral or Mistral endpoint

Phase 6 adds broader document processing, upload validation, parser diagnostics,
and processing evaluation. Phase 6.5 uses LangGraph for RAG workflow
orchestration. The existing ingestion, status, document, retrieval, and query
endpoints remain available.

The query workflow is a LangGraph graph with `retrieve`, `decide`, `generate`,
and `fallback` nodes. Retrieval remains in Weaviate, embeddings remain
`BAAI/bge-m3`, and generation still uses the configured local OpenAI-compatible
Ministral endpoint.

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

To upload a document, choose a supported file, keep or change the `doc_type`,
and click Upload. Supported extensions are `.pdf`, `.txt`, `.md`, `.docx`, and
`.csv`; the default maximum upload size is 25 MB. To ingest text, paste text,
set a filename and `doc_type`, and click Ingest Text. Upload and text-ingest
responses include parser diagnostics such as `parser_used`, `warnings`,
`original_file_size_bytes`, and `detected_extension`. To ask questions, enter a
question, set `top_k`, choose
`retrieval_mode`, and optionally add `doc_type`, `filename`, or `source_id`
filters. Empty filter fields are not sent. Answers show the retrieval mode,
applied filters, and retrieved sources. To delete a document, refresh the
document list and click Delete for the matching `source_id`. Documents with a
saved original file also show a Download button.

Document cards keep the full page list in the API response, but the frontend
shows only the first 12 page numbers and a remainder count for large documents.

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
`content_hash`. Phase 6 also stores processing metadata such as `parser_used`,
`detected_extension`, `original_file_size_bytes`, `warnings`, `section_title`,
CSV `row_number`, and `chunk_char_count` when available.

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

Allowed upload extensions are `.pdf`, `.txt`, `.md`, `.docx`, and `.csv`.
`RAG_MAX_UPLOAD_MB` controls the maximum accepted file size and defaults to
`25`. Empty files, unsupported extensions, and oversized files return clear HTTP
errors before indexing.

Parser behavior:

- PDF: parsed through Docling first, with the existing PDF text fallback.
- TXT: safely decoded as plain text.
- MD: decoded as text and Markdown headings are preserved as section metadata.
- DOCX: parsed as document text without requiring a separate service.
- CSV: parsed into readable row text; empty rows are skipped with a warning.

OCR is not implemented in Phase 6. Scanned image PDFs remain future work unless
text extraction is already available through the current parser path.

Ingestion responses include:

```json
{
  "parser_used": "csv",
  "warnings": ["CSV contained empty rows that were skipped."],
  "original_file_size_bytes": 128,
  "detected_extension": ".csv"
}
```

## Query

```bash
curl -X POST "http://localhost:8090/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","top_k":5,"retrieval_mode":"vector"}'
```

`top_k` controls how many chunks are retrieved before generation. The default
retrieval mode is `vector`, which embeds the query and uses Weaviate vector
similarity search. `hybrid` attempts Weaviate hybrid search using the query text
and query vector. If hybrid search is not available with the current Weaviate
client or collection configuration, `/rag/query` returns HTTP 400 with:

```text
Hybrid retrieval is not available in the current configuration.
```

Optional metadata filters are supported:

```json
{
  "question": "What is the refund policy?",
  "top_k": 3,
  "retrieval_mode": "vector",
  "doc_type": "general",
  "filename": "policy.txt",
  "source_id": "example-source-id"
}
```

When more than one filter is supplied, all filters must match. If no matching
chunks are retrieved, the API returns HTTP 200 with no sources and does not call
the generator.

`/rag/query` is orchestrated by LangGraph. The graph retrieves chunks, decides
whether any context exists, calls the generator only when documents were found,
and otherwise returns the no-information fallback answer.

Responses keep `answer`, `sources`, and `retrieved_chunk_count`, and also return
`retrieval_mode` and `filters_applied`. Source entries include metadata such as
filename, document type, source ID, chunk index, page number, text, vector
distance, stored file path, document hash, content hash, and retrieval score
when Weaviate provides one. Smaller vector distances are closer matches; hybrid
scores are Weaviate-provided relevance scores.

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

## Retrieval Evaluation

With the stack running on `http://localhost:8090`, run:

```bash
python3 scripts/evaluate_retrieval.py
```

The script ingests small generic policy documents, accepts duplicate ingest
responses, runs vector retrieval tests with `top_k=3`, checks `doc_type`
filtering, checks the no-match filter response, and tests hybrid mode when
available. A clear hybrid HTTP 400 is reported as unavailable and is acceptable.

## Processing Evaluation

With the stack running on `http://localhost:8090`, run:

```bash
python3 scripts/evaluate_processing.py
```

The script creates temporary TXT, MD, CSV, and DOCX samples, uploads them through
`/rag/ingest/file`, verifies processing diagnostics, accepts duplicate uploads,
and runs a hybrid query with a filename filter.
