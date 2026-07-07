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
    rag_workflow: Literal["langgraph"] = "langgraph"
    rag_vector_db: Literal["weaviate"] = "weaviate"
    rag_embedding_model: str = "BAAI/bge-m3"

    weaviate_url: str = "http://weaviate:8080"
    weaviate_collection: str = "KnowledgeBase"

    rag_top_k: int = Field(default=5, ge=1)
    rag_timeout_seconds: int = Field(default=30, ge=1)
    rag_max_upload_mb: int = Field(default=25, ge=1)
    rag_require_api_key: bool = False
    rag_api_key: str = "dev-rag-key"
    rag_default_project_id: str = "default"
    rag_project_id_pattern: str = r"^[A-Za-z0-9_-]+$"

    rag_generator_base_url: str = "http://host.docker.internal:8001/v1"
    rag_generator_api_key: str = "not-needed"
    rag_generator_model: str = "Ministral-3-8B-Instruct"
    rag_generator_temperature: float = Field(default=0.1, ge=0)
    rag_generator_max_tokens: int = Field(default=512, ge=1)
    rag_generator_timeout_seconds: int = Field(default=60, ge=1)

    @property
    def vector_db(self) -> str:
        return self.rag_vector_db

    @property
    def embedding_model(self) -> str:
        return self.rag_embedding_model


@lru_cache
def get_rag_settings() -> RagSettings:
    return RagSettings()
