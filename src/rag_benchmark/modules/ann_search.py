from typing import Iterable

import numpy as np

from ..indexing.flat_index import FlatIndexBuilder
from ..indexing.hnsw_index import HNSWIndexBuilder
from ..indexing.ivf_index import IVFIndexBuilder
from ..indexing.ivfpq_index import IVFPQIndexBuilder
from ..models.document import Document
from ..models.result import SearchResult


class AnnSearcher:
    def __init__(self, index_type: str, embeddings: np.ndarray, documents: list[Document]):
        self.index_type = index_type
        self.embeddings = embeddings
        self.documents = documents
        self.dimension = embeddings.shape[1]
        self._index = self._build_index()

    def _build_index(self):
        if self.index_type == "Flat":
            return FlatIndexBuilder(self.dimension).build(self.embeddings)
        if self.index_type == "HNSW":
            return HNSWIndexBuilder(self.dimension).build(self.embeddings)
        if self.index_type == "IVF":
            nlist = max(100, int(len(self.embeddings) ** 0.5))
            return IVFIndexBuilder(self.dimension, nlist=nlist).build(self.embeddings)
        if self.index_type == "IVF+PQ":
            nlist = max(100, int(len(self.embeddings) ** 0.5))
            return IVFPQIndexBuilder(self.dimension, nlist=nlist).build(self.embeddings)
        raise ValueError(f"Unknown index_type: {self.index_type}")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        distances, labels = self._index.search(query_vector.reshape(1, -1), top_k)
        results: list[SearchResult] = []
        for rank, (doc_idx, score) in enumerate(zip(labels[0], distances[0]), start=1):
            if doc_idx < 0:
                continue
            doc = self.documents[int(doc_idx)]
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    rank=rank,
                    score=float(score),
                    metadata=doc.metadata,
                    text=doc.content,
                )
            )
        return results

    def search_many(self, query_vectors: Iterable[np.ndarray], top_k: int) -> list[list[SearchResult]]:
        return [self.search(vec, top_k) for vec in query_vectors]
