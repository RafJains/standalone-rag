from __future__ import annotations

import json
import urllib.error
import urllib.request

from app.config import RagSettings, get_rag_settings
from app.rag_types import RagDocument


class GeneratorUnavailableError(RuntimeError):
    """Raised when the configured local generator endpoint cannot answer."""


SYSTEM_PROMPT = (
    "You are a concise retrieval-augmented answering assistant. "
    "Answer only from the provided context. Do not invent facts. "
    "If the answer is missing from the context, say you do not know. "
    "Keep the answer concise."
)
HUMAN_PROMPT_TEMPLATE = "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"


def generate_answer(
    question: str,
    documents: list[RagDocument],
    settings: RagSettings | None = None,
) -> str:
    active_settings = settings or get_rag_settings()
    payload = {
        "model": active_settings.rag_generator_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HUMAN_PROMPT_TEMPLATE.format(
                    question=question,
                    context=format_context_blocks(documents),
                ),
            },
        ],
        "temperature": active_settings.rag_generator_temperature,
        "max_tokens": active_settings.rag_generator_max_tokens,
    }
    request = urllib.request.Request(
        _chat_completions_url(active_settings.rag_generator_base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {active_settings.rag_generator_api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=active_settings.rag_generator_timeout_seconds,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise GeneratorUnavailableError(
            "Local generator LLM is unavailable. Confirm the configured "
            "OpenAI-compatible endpoint is running and reachable."
        ) from exc

    try:
        return str(body["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise GeneratorUnavailableError(
            "Local generator LLM returned an unexpected response shape."
        ) from exc


def format_context_blocks(documents: list[RagDocument]) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"filename: {_metadata_value(metadata.get('filename'))}",
                    f"doc_type: {_metadata_value(metadata.get('doc_type'))}",
                    f"chunk_index: {_metadata_value(metadata.get('chunk_index'))}",
                    f"page_number: {_metadata_value(metadata.get('page_number'))}",
                    "text:",
                    document.page_content.strip(),
                ]
            )
        )
    return "\n\n".join(blocks)


def _chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _metadata_value(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)
