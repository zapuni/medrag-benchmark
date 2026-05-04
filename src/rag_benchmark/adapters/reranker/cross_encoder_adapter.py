from sentence_transformers import CrossEncoder

from .base import RerankerAdapter
from ...config.settings import settings
from ...models.result import SearchResult


class CrossEncoderAdapter(RerankerAdapter):
    def __init__(self) -> None:
        self.model = CrossEncoder(settings.reranker.model, device=settings.reranker.device)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
        reranked = []
        for rank, (item, score) in enumerate(ranked[:top_k], start=1):
            reranked.append(
                SearchResult(
                    doc_id=item.doc_id,
                    rank=rank,
                    score=float(score),
                    metadata=item.metadata,
                    text=item.text,
                )
            )
        return reranked
