from pathlib import Path

import faiss
import numpy as np

from ..config.settings import settings


class IVFPQIndexBuilder:
    def __init__(self, dimension: int, nlist: int = 100, M_pq: int = 48, nbits: int = 8) -> None:
        self.dimension = dimension
        self.nlist = nlist
        self.M_pq = M_pq
        self.nbits = nbits
        self._cpu_index = None
        self._gpu_index = None

    def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFPQIndexBuilder":
        n_vectors = len(vectors)
        max_nlist = max(1, n_vectors // 39)
        if self.nlist > max_nlist:
            self.nlist = max_nlist
        if self.nlist < 1:
            raise ValueError("Too few vectors for IVFPQ training (need at least 39).")
        quantizer = faiss.IndexFlatIP(self.dimension)
        self._cpu_index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            self.nlist,
            self.M_pq,
            self.nbits,
            faiss.METRIC_INNER_PRODUCT,
        )
        self._cpu_index.train(vectors)
        self._cpu_index.add(vectors)
        self._cpu_index.nprobe = min(nprobe, self.nlist)
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            cloner = faiss.GpuClonerOptions()
            cloner.useFloat16 = False
            self._gpu_index = faiss.index_cpu_to_gpu(
                res,
                settings.faiss.gpu_id,
                self._cpu_index,
                cloner,
            )
        return self

    def search(self, query: np.ndarray, k: int, nprobe: int | None = None):
        if nprobe is not None:
            self._cpu_index.nprobe = nprobe
            if self._gpu_index is not None:
                try:
                    faiss.downcast_index(self._gpu_index).setNumProbes(nprobe)
                except Exception:
                    pass
        if self._gpu_index:
            return self._gpu_index.search(query, k)
        return self._cpu_index.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._cpu_index, str(path))

    @classmethod
    def load(cls, path: Path, dimension: int) -> "IVFPQIndexBuilder":
        builder = cls(dimension)
        builder._cpu_index = faiss.read_index(str(path))
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            cloner = faiss.GpuClonerOptions()
            cloner.useFloat16 = False
            builder._gpu_index = faiss.index_cpu_to_gpu(
                res,
                settings.faiss.gpu_id,
                builder._cpu_index,
                cloner,
            )
        return builder
