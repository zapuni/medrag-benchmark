from pathlib import Path

import faiss
import numpy as np

from ..config.settings import settings


class FlatIndexBuilder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._cpu_index = None
        self._gpu_index = None

    def build(self, vectors: np.ndarray) -> "FlatIndexBuilder":
        self._cpu_index = faiss.IndexFlatIP(self.dimension)
        self._cpu_index.add(vectors)
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(res, settings.faiss.gpu_id, self._cpu_index)
        return self

    def search(self, query: np.ndarray, k: int):
        if self._gpu_index:
            return self._gpu_index.search(query, k)
        return self._cpu_index.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._cpu_index, str(path))

    @classmethod
    def load(cls, path: Path, dimension: int) -> "FlatIndexBuilder":
        builder = cls(dimension)
        builder._cpu_index = faiss.read_index(str(path))
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            builder._gpu_index = faiss.index_cpu_to_gpu(res, settings.faiss.gpu_id, builder._cpu_index)
        return builder
