from ..adapters.reranker.cross_encoder_adapter import CrossEncoderAdapter
from ..models.result import SearchResult


class Reranker:
    def __init__(self) -> None:
        self._reranker = CrossEncoderAdapter()

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        return self._reranker.rerank(query, candidates, top_k)
