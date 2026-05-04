import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingAdapter
from ...config.settings import settings


class SentenceTransformerAdapter(EmbeddingAdapter):
    def __init__(self) -> None:
        self.model = SentenceTransformer(
            settings.embedding.model,
            device=settings.embedding.device,
        )
        self._dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        bs = batch_size or settings.embedding.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype="float32")

    @property
    def dimension(self) -> int:
        return self._dimension
