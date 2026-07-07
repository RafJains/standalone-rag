from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


BASE_URL = os.environ.get("RAG_EVAL_BASE_URL", "http://localhost:8090")
API_KEY = os.environ.get("RAG_API_KEY")


class ApiError(RuntimeError):
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


def request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    api_key: str | None = API_KEY,
) -> Any:
    data = None
    headers = {}
    resolved_method = method or "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        resolved_method = method or "POST"
    if api_key:
        headers["X-RAG-API-Key"] = api_key

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=resolved_method,
    )
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


def query_string(project_id: str) -> str:
    return urllib.parse.urlencode({"project_id": project_id})


def filenames(response: dict[str, Any]) -> set[str]:
    return {
        str(source.get("filename"))
        for source in response.get("sources", [])
        if source.get("filename")
    }


def document_filenames(response: dict[str, Any]) -> set[str]:
    return {
        str(document.get("filename"))
        for document in response.get("documents", [])
        if document.get("filename")
    }


def source_id_for(response: dict[str, Any], filename: str) -> str:
    for document in response.get("documents", []):
        if document.get("filename") == filename and document.get("source_id"):
            return str(document["source_id"])
    raise AssertionError(f"source_id not found for {filename}")


def expect(name: str, passed: bool, detail: str = "") -> bool:
    print(f"{'PASS' if passed else 'FAIL'}: {name}{f' - {detail}' if detail else ''}")
    return passed


def ingest() -> bool:
    alpha = request_json(
        "/rag/ingest/text",
        {
            "project_id": "alpha",
            "filename": "alpha-refund-policy.txt",
            "doc_type": "policy",
            "text": "Alpha project refund window is 11 days.",
        },
    )
    beta = request_json(
        "/rag/ingest/text",
        {
            "project_id": "beta",
            "filename": "beta-refund-policy.txt",
            "doc_type": "policy",
            "text": "Beta project refund window is 22 days.",
        },
    )
    return expect("ingest alpha", alpha.get("project_id") == "alpha") and expect(
        "ingest beta", beta.get("project_id") == "beta"
    )


def query_projects() -> bool:
    alpha = request_json(
        "/rag/query",
        {
            "question": "What is the refund window?",
            "top_k": 3,
            "retrieval_mode": "hybrid",
            "project_id": "alpha",
        },
    )
    beta = request_json(
        "/rag/query",
        {
            "question": "What is the refund window?",
            "top_k": 3,
            "retrieval_mode": "hybrid",
            "project_id": "beta",
        },
    )
    alpha_files = filenames(alpha)
    beta_files = filenames(beta)
    return expect(
        "alpha query isolation",
        "alpha-refund-policy.txt" in alpha_files and "beta-refund-policy.txt" not in alpha_files,
        ", ".join(sorted(alpha_files)) or "(none)",
    ) and expect(
        "beta query isolation",
        "beta-refund-policy.txt" in beta_files and "alpha-refund-policy.txt" not in beta_files,
        ", ".join(sorted(beta_files)) or "(none)",
    )


def list_projects() -> tuple[bool, str]:
    alpha = request_json(f"/rag/documents?{query_string('alpha')}")
    beta = request_json(f"/rag/documents?{query_string('beta')}")
    alpha_files = document_filenames(alpha)
    beta_files = document_filenames(beta)
    listed = expect(
        "alpha list isolation",
        "alpha-refund-policy.txt" in alpha_files and "beta-refund-policy.txt" not in alpha_files,
        ", ".join(sorted(alpha_files)) or "(none)",
    ) and expect(
        "beta list isolation",
        "beta-refund-policy.txt" in beta_files and "alpha-refund-policy.txt" not in beta_files,
        ", ".join(sorted(beta_files)) or "(none)",
    )
    return listed, source_id_for(alpha, "alpha-refund-policy.txt")


def delete_alpha(alpha_source_id: str) -> bool:
    deleted = request_json(
        f"/rag/documents/{urllib.parse.quote(alpha_source_id)}?{query_string('alpha')}",
        method="DELETE",
    )
    beta = request_json(f"/rag/documents?{query_string('beta')}")
    beta_files = document_filenames(beta)
    return expect(
        "delete alpha under alpha",
        deleted.get("project_id") == "alpha" and int(deleted.get("deleted_count") or 0) > 0,
    ) and expect(
        "beta remains after alpha delete",
        "beta-refund-policy.txt" in beta_files,
        ", ".join(sorted(beta_files)) or "(none)",
    )


def main() -> int:
    try:
        passed = ingest()
        passed = query_projects() and passed
        listed, alpha_source_id = list_projects()
        passed = listed and passed
        passed = delete_alpha(alpha_source_id) and passed
    except Exception as exc:
        print(f"FAIL: project isolation evaluation error - {exc}", file=sys.stderr)
        return 1

    if passed:
        print("project isolation evaluation passed")
        return 0

    print("project isolation evaluation failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
