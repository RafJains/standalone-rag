from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Standalone RAG settings read only from the RAG service environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    rag_enabled: bool = True
    rag_workflow: Literal["langchain"] = "langchain"
    rag_vector_db: Literal["weaviate"] = "weaviate"
    rag_embedding_model: str = "BAAI/bge-m3"

    weaviate_url: str = "http://weaviate:8080"
    weaviate_collection: str = "KnowledgeBase"

    rag_top_k: int = Field(default=5, ge=1)
    rag_timeout_seconds: int = Field(default=30, ge=1)

    @property
    def vector_db(self) -> str:
        return self.rag_vector_db

    @property
    def embedding_model(self) -> str:
        return self.rag_embedding_model


@lru_cache
def get_rag_settings() -> RagSettings:
    return RagSettings()
