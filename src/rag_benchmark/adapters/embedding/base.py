from abc import ABC, abstractmethod

import numpy as np


class EmbeddingAdapter(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode texts into vectors."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        raise NotImplementedError
