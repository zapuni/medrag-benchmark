import time

import numpy as np
from tqdm import tqdm

from ..indexing.flat_index import FlatIndexBuilder
from ..indexing.hnsw_index import HNSWIndexBuilder
from ..indexing.ivf_index import IVFIndexBuilder
from ..indexing.ivfpq_index import IVFPQIndexBuilder
from ..models.result import BenchmarkResult
from .metrics import compute_recall_at_k


class BenchmarkRunner:
    def __init__(self, embeddings: np.ndarray, query_embeddings: np.ndarray, k: int = 10) -> None:
        self.embeddings = embeddings
        self.query_embeddings = query_embeddings
        self.k = k
        self.n_vectors = len(embeddings)
        self.n_queries = len(query_embeddings)
        self.dimension = embeddings.shape[1]

        print("Computing ground truth (Flat)...")
        self._ground_truth = self._compute_ground_truth()

    def _compute_ground_truth(self) -> np.ndarray:
        flat = FlatIndexBuilder(self.dimension).build(self.embeddings)
        _, labels = flat.search(self.query_embeddings, self.k)
        return labels

    def _benchmark_search(self, search_fn, desc: str):
        times = []
        all_labels = []
        for q in tqdm(self.query_embeddings, desc=desc, unit="q"):
            t0 = time.perf_counter()
            _, labels = search_fn(q.reshape(1, -1))
            times.append((time.perf_counter() - t0) * 1000)
            all_labels.append(labels[0])
        return times, np.array(all_labels)

    def benchmark_flat(self) -> BenchmarkResult:
        flat = FlatIndexBuilder(self.dimension).build(self.embeddings)
        times, _ = self._benchmark_search(lambda q: flat.search(q, self.k), desc="Flat search")
        return BenchmarkResult(
            index_type="Flat",
            n_vectors=self.n_vectors,
            latency_ms=float(np.mean(times)),
            latency_p95_ms=float(np.percentile(times, 95)),
            recall_at_k=1.0,
            k=self.k,
            params={},
            n_queries=self.n_queries,
        )

    def benchmark_hnsw(
        self, M: int = 32, ef_search: int = 128, ef_construction: int = 200
    ) -> BenchmarkResult:
        index = HNSWIndexBuilder(self.dimension, M=M, ef_construction=ef_construction).build(
            self.embeddings
        )
        times, retrieved = self._benchmark_search(
            lambda q: index.search(q, self.k, ef_search=ef_search),
            desc=f"HNSW search (ef={ef_search})",
        )
        recall = compute_recall_at_k(retrieved, self._ground_truth, self.k)
        return BenchmarkResult(
            index_type="HNSW",
            n_vectors=self.n_vectors,
            latency_ms=float(np.mean(times)),
            latency_p95_ms=float(np.percentile(times, 95)),
            recall_at_k=recall,
            k=self.k,
            params={"M": M, "efSearch": ef_search, "efConstruction": ef_construction},
            n_queries=self.n_queries,
        )

    def benchmark_ivf(self, nlist: int = 100, nprobe: int = 10) -> BenchmarkResult:
        index = IVFIndexBuilder(self.dimension, nlist=nlist).build(self.embeddings, nprobe=nprobe)
        times, retrieved = self._benchmark_search(
            lambda q: index.search(q, self.k, nprobe=nprobe),
            desc=f"IVF search (nprobe={nprobe})",
        )
        recall = compute_recall_at_k(retrieved, self._ground_truth, self.k)
        return BenchmarkResult(
            index_type="IVF",
            n_vectors=self.n_vectors,
            latency_ms=float(np.mean(times)),
            latency_p95_ms=float(np.percentile(times, 95)),
            recall_at_k=recall,
            k=self.k,
            params={"nlist": nlist, "nprobe": nprobe},
            n_queries=self.n_queries,
        )

    def benchmark_ivfpq(self, nlist: int = 100, M_pq: int = 48, nprobe: int = 10) -> BenchmarkResult:
        index = IVFPQIndexBuilder(self.dimension, nlist=nlist, M_pq=M_pq).build(
            self.embeddings, nprobe=nprobe
        )
        times, retrieved = self._benchmark_search(
            lambda q: index.search(q, self.k, nprobe=nprobe),
            desc=f"IVF+PQ search (nprobe={nprobe})",
        )
        recall = compute_recall_at_k(retrieved, self._ground_truth, self.k)
        return BenchmarkResult(
            index_type="IVF+PQ",
            n_vectors=self.n_vectors,
            latency_ms=float(np.mean(times)),
            latency_p95_ms=float(np.percentile(times, 95)),
            recall_at_k=recall,
            k=self.k,
            params={"nlist": nlist, "nprobe": nprobe, "M_pq": M_pq},
            n_queries=self.n_queries,
        )
