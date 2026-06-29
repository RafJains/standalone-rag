from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import RagSettings, get_rag_settings


class GeneratorUnavailableError(RuntimeError):
    """Raised when the configured local generator endpoint cannot answer."""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise retrieval-augmented answering assistant. "
            "Answer only from the provided context. Do not invent facts. "
            "If the answer is missing from the context, say you do not know. "
            "Keep the answer concise.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)


def generate_answer(
    question: str,
    documents: list[Document],
    settings: RagSettings | None = None,
) -> str:
    active_settings = settings or get_rag_settings()
    context = format_context_blocks(documents)
    llm = ChatOpenAI(
        base_url=active_settings.rag_generator_base_url,
        api_key=active_settings.rag_generator_api_key,
        model=active_settings.rag_generator_model,
        temperature=active_settings.rag_generator_temperature,
        max_tokens=active_settings.rag_generator_max_tokens,
        timeout=active_settings.rag_generator_timeout_seconds,
    )
    chain = RAG_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({"question": question, "context": context})
    except Exception as exc:
        raise GeneratorUnavailableError(
            "Local generator LLM is unavailable. Confirm the configured "
            "OpenAI-compatible endpoint is running and reachable."
        ) from exc

    return answer.strip()


def format_context_blocks(documents: list[Document]) -> str:
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


def _metadata_value(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)
