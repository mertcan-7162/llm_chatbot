from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    _model: SentenceTransformer | None = None
    _model_name: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model

    def _get_model(self) -> SentenceTransformer:
        if Embedder._model is None or Embedder._model_name != self.model_name:
            logger.info("Loading embedding model: %s", self.model_name)
            Embedder._model = SentenceTransformer(self.model_name)
            Embedder._model_name = self.model_name
        return Embedder._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages with the 'passage: ' prefix for e5 models."""
        model = self._get_model()
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with the 'query: ' prefix for e5 models."""
        model = self._get_model()
        embedding = model.encode(f"query: {query}", normalize_embeddings=True)
        return embedding.tolist()
