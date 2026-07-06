from __future__ import annotations

import json
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


BASE_URL = "http://localhost:8090"
REQUIRED_DIAGNOSTICS = {
    "parser_used",
    "warnings",
    "original_file_size_bytes",
    "detected_extension",
}


class ApiError(RuntimeError):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


def request_json(path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = body_text
        raise ApiError(exc.code, body) from exc


def upload_file(path: Path, doc_type: str) -> dict[str, Any]:
    boundary = f"----rag-boundary-{uuid.uuid4().hex}"
    body = _multipart_body(boundary, path, doc_type)
    request = urllib.request.Request(
        f"{BASE_URL}/rag/ingest/file",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = body_text
        raise ApiError(exc.code, body) from exc


def _multipart_body(boundary: str, path: Path, doc_type: str) -> bytes:
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="doc_type"\r\n\r\n'
            f"{doc_type}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {_content_type(path)}\r\n\r\n"
        ).encode("utf-8")
        + path.read_bytes()
        + b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts)


def _content_type(path: Path) -> str:
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(path.suffix.lower(), "application/octet-stream")


def create_samples(directory: Path) -> list[Path]:
    samples = []

    txt = directory / "phase6-processing-refund.txt"
    txt.write_text(
        "Processing TXT sample: refund processing requires validating the request "
        "within three business days.",
        encoding="utf-8",
    )
    samples.append(txt)

    md = directory / "phase6-processing-guide.md"
    md.write_text(
        "# Processing Guide\n\nMarkdown files preserve section headings for source metadata.\n\n"
        "## Escalation\n\nEscalation notes should remain searchable.",
        encoding="utf-8",
    )
    samples.append(md)

    csv_file = directory / "phase6-processing-table.csv"
    csv_file.write_text(
        "policy,window,owner\n"
        "refund,3 business days,operations\n"
        "\n"
        "warranty,24 months,support\n",
        encoding="utf-8",
    )
    samples.append(csv_file)

    docx = directory / "phase6-processing-docx.docx"
    create_minimal_docx(
        docx,
        [
            "Processing DOCX sample",
            "DOCX documents are parsed into searchable text without a separate service.",
        ],
    )
    samples.append(docx)

    return samples


def create_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    paragraph_xml = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraph_xml}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def assert_diagnostics(filename: str, response: dict[str, Any]) -> bool:
    missing = sorted(field for field in REQUIRED_DIAGNOSTICS if field not in response)
    warnings_value = response.get("warnings")
    passed = (
        not missing
        and bool(response.get("parser_used"))
        and isinstance(warnings_value, list)
        and isinstance(response.get("original_file_size_bytes"), int)
        and bool(response.get("detected_extension"))
    )
    print(f"upload: {filename}")
    print(f"parser_used: {response.get('parser_used')}")
    print(f"detected_extension: {response.get('detected_extension')}")
    print(f"original_file_size_bytes: {response.get('original_file_size_bytes')}")
    print(f"warnings: {warnings_value}")
    print(f"diagnostics: {'PASS' if passed else 'FAIL'}")
    if missing:
        print(f"missing diagnostics: {', '.join(missing)}")
    print()
    return passed


def run_query(filename: str) -> bool:
    try:
        response = request_json(
            "/rag/query",
            {
                "question": "How quickly is refund processing validated?",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "filename": filename,
            },
        )
    except ApiError as exc:
        print(f"hybrid filename-filtered query: FAIL ({exc})")
        return False

    filenames = [
        source.get("filename")
        for source in response.get("sources", [])
        if source.get("filename")
    ]
    passed = filename in filenames
    print("hybrid filename-filtered query")
    print(f"expected filename: {filename}")
    print(f"retrieved filenames: {', '.join(filenames) if filenames else '(none)'}")
    print(f"result: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rag-processing-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        samples = create_samples(temp_dir)
        diagnostics_passed = True
        for sample in samples:
            try:
                response = upload_file(sample, "processing-eval")
            except ApiError as exc:
                print(f"upload: {sample.name} -> FAIL ({exc})")
                diagnostics_passed = False
                continue

            if response.get("duplicate"):
                print(f"upload: {sample.name} -> duplicate accepted")
            diagnostics_passed = assert_diagnostics(sample.name, response) and diagnostics_passed

        query_passed = run_query("phase6-processing-refund.txt")

    if diagnostics_passed and query_passed:
        print("processing evaluation passed")
        return 0

    print("processing evaluation failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
