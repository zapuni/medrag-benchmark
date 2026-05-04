from abc import ABC, abstractmethod

from ...models.result import SearchResult


class RerankerAdapter(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Return reranked list of results."""
        raise NotImplementedError
