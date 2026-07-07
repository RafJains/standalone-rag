from threading import Lock

from sentence_transformers import SentenceTransformer

from app.config import get_rag_settings


class LocalEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._encode_lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._encode_lock:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return embeddings.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings() -> LocalEmbeddings:
    global _EMBEDDINGS
    with _EMBEDDINGS_LOCK:
        if _EMBEDDINGS is None:
            settings = get_rag_settings()
            _EMBEDDINGS = LocalEmbeddings(settings.embedding_model)
        return _EMBEDDINGS


_EMBEDDINGS: LocalEmbeddings | None = None
_EMBEDDINGS_LOCK = Lock()
