from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_rag_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_rag_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
