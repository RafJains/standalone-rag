from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_rag_settings


class LocalEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache
def get_embeddings() -> LocalEmbeddings:
    settings = get_rag_settings()
    return LocalEmbeddings(settings.embedding_model)
