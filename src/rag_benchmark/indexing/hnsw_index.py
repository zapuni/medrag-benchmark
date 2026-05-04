from pathlib import Path

import faiss
import numpy as np

from ..config.settings import settings


class HNSWIndexBuilder:
    def __init__(self, dimension: int, M: int = 32, ef_construction: int = 200) -> None:
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self._index = None

    def build(self, vectors: np.ndarray) -> "HNSWIndexBuilder":
        self._index = faiss.IndexHNSWFlat(
            self.dimension, self.M, faiss.METRIC_INNER_PRODUCT
        )
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.add(vectors)
        return self

    def search(self, query: np.ndarray, k: int, ef_search: int = 128):
        self._index.hnsw.efSearch = ef_search
        return self._index.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path, dimension: int) -> "HNSWIndexBuilder":
        builder = cls(dimension)
        builder._index = faiss.read_index(str(path))
        return builder
