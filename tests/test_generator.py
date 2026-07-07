import json
import urllib.error

import pytest

from app.config import RagSettings
from app.generator import GeneratorUnavailableError, format_context_blocks, generate_answer
from app.rag_types import RagDocument


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_generate_answer_posts_chat_completion_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "  answer text  "}}]})

    monkeypatch.setattr("app.generator.urllib.request.urlopen", fake_urlopen)
    settings = RagSettings(
        rag_generator_base_url="http://generator.test/v1",
        rag_generator_api_key="secret",
        rag_generator_model="Ministral-3-8B-Instruct",
        rag_generator_temperature=0.2,
        rag_generator_max_tokens=77,
        rag_generator_timeout_seconds=9,
    )

    answer = generate_answer(
        "What is covered?",
        [RagDocument("Coverage text", {"filename": "policy.txt", "doc_type": "policy", "chunk_index": 2})],
        settings=settings,
    )

    assert answer == "answer text"
    assert captured["url"] == "http://generator.test/v1/chat/completions"
    assert captured["timeout"] == 9
    assert captured["payload"]["model"] == "Ministral-3-8B-Instruct"
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["max_tokens"] == 77
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert captured["payload"]["messages"][1]["role"] == "user"


def test_generate_answer_rejects_unexpected_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.generator.urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"unexpected": []}),
    )

    with pytest.raises(GeneratorUnavailableError):
        generate_answer("Question?", [], settings=RagSettings())


@pytest.mark.parametrize(
    "raised",
    [
        OSError("connection failed"),
        urllib.error.HTTPError("http://generator.test", 500, "error", {}, None),
    ],
)
def test_generate_answer_wraps_connection_and_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
) -> None:
    def fake_urlopen(request, timeout):
        raise raised

    monkeypatch.setattr("app.generator.urllib.request.urlopen", fake_urlopen)

    with pytest.raises(GeneratorUnavailableError):
        generate_answer("Question?", [], settings=RagSettings())


def test_context_formatting_includes_source_metadata() -> None:
    context = format_context_blocks(
        [
            RagDocument(
                "Policy text",
                {
                    "filename": "policy.txt",
                    "doc_type": "policy",
                    "chunk_index": 3,
                    "page_number": 4,
                },
            )
        ]
    )

    assert "[Source 1]" in context
    assert "filename: policy.txt" in context
    assert "doc_type: policy" in context
    assert "chunk_index: 3" in context
    assert "page_number: 4" in context
    assert "Policy text" in context
