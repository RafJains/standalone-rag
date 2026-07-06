from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


BASE_URL = "http://localhost:8090"
TOP_K = 3


@dataclass(frozen=True)
class TestDocument:
    filename: str
    doc_type: str
    text: str


@dataclass(frozen=True)
class RetrievalTest:
    query: str
    expected_filename: str


DOCUMENTS = [
    TestDocument(
        filename="refund-policy.txt",
        doc_type="refund",
        text=(
            "Refund policy: customers may cancel and request a refund within "
            "14 calendar days of purchase. Refund requests made after the "
            "cancellation window are reviewed only for documented billing errors."
        ),
    ),
    TestDocument(
        filename="warranty-policy.txt",
        doc_type="warranty",
        text=(
            "Warranty policy: covered products include a 24 month warranty period "
            "starting on the purchase date. The warranty covers defects in materials "
            "and workmanship."
        ),
    ),
    TestDocument(
        filename="data-retention-policy.txt",
        doc_type="data-retention",
        text=(
            "Data retention policy: user activity logs are retained for 90 days. "
            "After 90 days, activity logs are deleted unless a legal hold applies."
        ),
    ),
    TestDocument(
        filename="security-incident-policy.txt",
        doc_type="security",
        text=(
            "Security incident policy: suspected security incidents should be "
            "reported to the response team within one hour of discovery. Critical "
            "incidents require immediate escalation."
        ),
    ),
    TestDocument(
        filename="support-sla-policy.txt",
        doc_type="support-sla",
        text=(
            "Support SLA policy: priority support requests receive a first response "
            "within two business hours. Standard requests receive a first response "
            "within one business day."
        ),
    ),
]

TESTS = [
    RetrievalTest("What is the refund cancellation window?", "refund-policy.txt"),
    RetrievalTest("How long is the warranty period?", "warranty-policy.txt"),
    RetrievalTest("How long are user activity logs retained?", "data-retention-policy.txt"),
    RetrievalTest("When should security incidents be reported?", "security-incident-policy.txt"),
    RetrievalTest("What is the support SLA for priority requests?", "support-sla-policy.txt"),
]


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

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = body_text
        raise ApiError(exc.code, body) from exc


def ingest_documents() -> None:
    for document in DOCUMENTS:
        payload = {
            "text": document.text,
            "filename": document.filename,
            "doc_type": document.doc_type,
        }
        response = request_json("/rag/ingest/text", payload)
        duplicate = response.get("duplicate", False)
        status = "duplicate" if duplicate else "indexed"
        print(f"ingest: {document.filename} -> {status}")


def run_query(payload: dict[str, Any]) -> Any:
    return request_json("/rag/query", payload)


def retrieved_filenames(response: dict[str, Any]) -> list[str]:
    return [
        str(source.get("filename"))
        for source in response.get("sources", [])
        if source.get("filename")
    ]


def print_result(query: str, expected: str, filenames: list[str], passed: bool) -> None:
    outcome = "PASS" if passed else "FAIL"
    print(f"query: {query}")
    print(f"expected filename: {expected}")
    print(f"retrieved filenames: {', '.join(filenames) if filenames else '(none)'}")
    print(f"result: {outcome}")
    print()


def run_vector_tests() -> bool:
    passed_all = True
    for test in TESTS:
        response = run_query(
            {
                "question": test.query,
                "top_k": TOP_K,
                "retrieval_mode": "vector",
            }
        )
        filenames = retrieved_filenames(response)
        passed = test.expected_filename in filenames
        print_result(test.query, test.expected_filename, filenames, passed)
        passed_all = passed_all and passed
    return passed_all


def run_filter_tests() -> bool:
    response = run_query(
        {
            "question": "What is the refund cancellation window?",
            "top_k": TOP_K,
            "retrieval_mode": "vector",
            "doc_type": "refund",
        }
    )
    filenames = retrieved_filenames(response)
    filter_passed = "refund-policy.txt" in filenames and all(
        source.get("doc_type") == "refund" for source in response.get("sources", [])
    )
    print_result(
        "What is the refund cancellation window? [doc_type=refund]",
        "refund-policy.txt",
        filenames,
        filter_passed,
    )

    no_match = run_query(
        {
            "question": "What is the refund cancellation window?",
            "top_k": TOP_K,
            "retrieval_mode": "vector",
            "doc_type": "nonexistent-type",
        }
    )
    no_match_passed = (
        no_match.get("retrieved_chunk_count") == 0
        and no_match.get("sources") == []
        and no_match.get("answer")
        == "No relevant indexed information was found for this question."
    )
    print("query: What is the refund cancellation window? [doc_type=nonexistent-type]")
    print("expected filename: (none)")
    print(f"retrieved filenames: {retrieved_filenames(no_match) or '(none)'}")
    print(f"result: {'PASS' if no_match_passed else 'FAIL'}")
    print()
    return filter_passed and no_match_passed


def run_hybrid_test() -> bool:
    try:
        response = run_query(
            {
                "question": "What is the refund cancellation window?",
                "top_k": TOP_K,
                "retrieval_mode": "hybrid",
            }
        )
    except ApiError as exc:
        detail = exc.body.get("detail") if isinstance(exc.body, dict) else str(exc.body)
        if exc.status == 400 and "Hybrid retrieval is not available" in str(detail):
            print("hybrid: unavailable in current configuration")
            print()
            return True
        print(f"hybrid: FAIL ({exc})")
        print()
        return False

    filenames = retrieved_filenames(response)
    passed = "refund-policy.txt" in filenames
    print_result(
        "What is the refund cancellation window? [hybrid]",
        "refund-policy.txt",
        filenames,
        passed,
    )
    return passed


def main() -> int:
    try:
        ingest_documents()
        print()
        vector_passed = run_vector_tests()
        filters_passed = run_filter_tests()
        hybrid_acceptable = run_hybrid_test()
    except Exception as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1

    if vector_passed and filters_passed and hybrid_acceptable:
        print("retrieval evaluation passed")
        return 0

    print("retrieval evaluation failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
