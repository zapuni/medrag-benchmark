import gc
import threading
import time
from typing import Optional

import faiss
import numpy as np
from tqdm import tqdm

from ..config.settings import settings
from ..models.result import BenchmarkResult
from .metrics import compute_recall_at_k
from .params import BenchmarkParams


def _make_gpu_res(gpu_id: int, temp_mb: int) -> faiss.StandardGpuResources:
    res = faiss.StandardGpuResources()
    res.setTempMemory(temp_mb * 1024 * 1024)
    return res


def _batched_search(
    index,
    queries: np.ndarray,
    k: int,
    batch_size: int,
    desc: str,
    position: int,
) -> tuple[list[float], np.ndarray]:
    times: list[float] = []
    labels_list: list[np.ndarray] = []
    for i in tqdm(
        range(0, len(queries), batch_size),
        desc=desc,
        unit="batch",
        position=position,
    ):
        batch = queries[i : i + batch_size]
        t0 = time.perf_counter()
        _, labels = index.search(batch, k)
        elapsed = (time.perf_counter() - t0) * 1000
        per_query = elapsed / max(1, len(batch))
        times.extend([per_query] * len(batch))
        labels_list.append(labels)
    return times, np.vstack(labels_list)


def _train_ivf_gpu(
    embeddings: np.ndarray,
    nlist: int,
    dimension: int,
    gpu_id: int,
) -> tuple[faiss.IndexIVFFlat, int]:
    res = _make_gpu_res(gpu_id, settings.faiss.temp_memory_mb)
    n = len(embeddings)
    safe_nlist = min(nlist, n // 39)

    print(f"[GPU{gpu_id}] IVF clustering: {n:,} vectors -> {safe_nlist} centroids")
    clus = faiss.Clustering(dimension, safe_nlist)
    clus.verbose = False
    clus.niter = settings.faiss.kmeans_niter
    clus.max_points_per_centroid = settings.faiss.kmeans_max_points_per_centroid

    flat_gpu = faiss.GpuIndexFlatIP(res, dimension)
    clus.train(embeddings, flat_gpu)

    centroids = faiss.vector_float_to_array(clus.centroids).reshape(safe_nlist, dimension)
    quantizer = faiss.IndexFlatIP(dimension)
    quantizer.add(centroids)

    cpu_index = faiss.IndexIVFFlat(quantizer, dimension, safe_nlist, faiss.METRIC_INNER_PRODUCT)
    cpu_index.is_trained = True
    cpu_index.add(embeddings)
    return cpu_index, safe_nlist


def _train_ivfpq_gpu(
    embeddings: np.ndarray,
    nlist: int,
    M_pq: int,
    nbits: int,
    dimension: int,
    gpu_id: int,
) -> tuple[faiss.IndexIVFPQ, int]:
    res = _make_gpu_res(gpu_id, settings.faiss.temp_memory_mb)
    n = len(embeddings)
    safe_nlist = min(nlist, n // 39)

    print(
        f"[GPU{gpu_id}] IVFPQ clustering: {n:,} vectors -> {safe_nlist} centroids, M_pq={M_pq}"
    )
    clus = faiss.Clustering(dimension, safe_nlist)
    clus.verbose = False
    clus.niter = settings.faiss.kmeans_niter
    clus.max_points_per_centroid = settings.faiss.kmeans_max_points_per_centroid

    flat_gpu = faiss.GpuIndexFlatIP(res, dimension)
    clus.train(embeddings, flat_gpu)

    centroids = faiss.vector_float_to_array(clus.centroids).reshape(safe_nlist, dimension)
    quantizer = faiss.IndexFlatIP(dimension)
    quantizer.add(centroids)

    cpu_index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        safe_nlist,
        M_pq,
        nbits,
        faiss.METRIC_INNER_PRODUCT,
    )
    cpu_index.is_trained = True
    cpu_index.add(embeddings)
    return cpu_index, safe_nlist


def _set_gpu_nprobe(index, nprobe: int) -> None:
    try:
        faiss.downcast_index(index).setNumProbes(nprobe)
        return
    except AttributeError:
        pass

    try:
        faiss.downcast_index(index).nprobe = nprobe
    except Exception:
        try:
            index.nprobe = nprobe
        except Exception:
            return


def _free_gpu_objects(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


class MultiGPUBenchmarkRunner:
    def __init__(
        self,
        embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        params: BenchmarkParams,
        gpu_ids: list[int] | None = None,
    ) -> None:
        self.embeddings = embeddings.astype("float32")
        self.query_embeddings = query_embeddings.astype("float32")
        self.params = params
        self.dimension = embeddings.shape[1]
        self.n_vectors = len(embeddings)
        self.n_queries = len(query_embeddings)
        self.gpu_ids = gpu_ids or settings.faiss.gpu_ids
        self.gpu0 = self.gpu_ids[0]
        self.gpu1 = self.gpu_ids[1]

        self.results_gpu0: list[BenchmarkResult] = []
        self.results_gpu1: list[BenchmarkResult] = []
        self._ground_truth: Optional[np.ndarray] = None
        self._ground_truth_ready = threading.Event()

    def _compute_recall(self, retrieved: np.ndarray, k: int) -> float:
        return compute_recall_at_k(retrieved, self._ground_truth, k)

    def _run_gpu0(self) -> None:
        gpu_id = self.gpu0
        k = settings.benchmark.top_k
        batch_size = settings.benchmark.query_batch_size

        print(f"[GPU{gpu_id}] Computing ground truth (Flat)")
        res = _make_gpu_res(gpu_id, settings.faiss.temp_memory_mb)
        flat_cpu = faiss.IndexFlatIP(self.dimension)
        flat_cpu.add(self.embeddings)
        flat_gpu = faiss.index_cpu_to_gpu(res, gpu_id, flat_cpu)

        times, labels = _batched_search(
            flat_gpu,
            self.query_embeddings,
            k,
            batch_size,
            desc="Flat ground truth",
            position=0,
        )
        self._ground_truth = labels
        self._ground_truth_ready.set()

        self.results_gpu0.append(
            BenchmarkResult(
                index_type="Flat",
                n_vectors=self.n_vectors,
                latency_ms=float(np.mean(times)),
                latency_p95_ms=float(np.percentile(times, 95)),
                recall_at_k=1.0,
                k=k,
                params={},
                n_queries=self.n_queries,
            )
        )

        del flat_gpu, flat_cpu
        _free_gpu_objects(res)

        print(f"[GPU{gpu_id}] Building HNSW M={self.params.M_hnsw}, efC={self.params.ef_construction}")
        faiss.omp_set_num_threads(settings.benchmark.hnsw_threads)
        hnsw_cpu = faiss.IndexHNSWFlat(self.dimension, self.params.M_hnsw)
        hnsw_cpu.hnsw.efConstruction = self.params.ef_construction
        t0 = time.perf_counter()
        hnsw_cpu.add(self.embeddings)
        build_min = (time.perf_counter() - t0) / 60
        print(f"[GPU{gpu_id}] HNSW built in {build_min:.1f} min")

        for ef in self.params.ef_search_values:
            hnsw_cpu.hnsw.efSearch = ef
            times, retrieved = _batched_search(
                hnsw_cpu,
                self.query_embeddings,
                k,
                batch_size,
                desc=f"HNSW search (ef={ef})",
                position=0,
            )
            recall = self._compute_recall(retrieved, k)
            self.results_gpu0.append(
                BenchmarkResult(
                    index_type="HNSW",
                    n_vectors=self.n_vectors,
                    latency_ms=float(np.mean(times)),
                    latency_p95_ms=float(np.percentile(times, 95)),
                    recall_at_k=recall,
                    k=k,
                    params={"M": self.params.M_hnsw, "efSearch": ef},
                    n_queries=self.n_queries,
                )
            )

    def _run_gpu1(self) -> None:
        gpu_id = self.gpu1
        k = settings.benchmark.top_k
        batch_size = settings.benchmark.query_batch_size

        self._ground_truth_ready.wait()
        print(f"[GPU{gpu_id}] Ground truth ready, starting IVF")

        ivf_cpu, safe_nlist = _train_ivf_gpu(
            self.embeddings, self.params.nlist, self.dimension, gpu_id
        )
        res = _make_gpu_res(gpu_id, settings.faiss.temp_memory_mb)
        ivf_gpu = faiss.index_cpu_to_gpu(res, gpu_id, ivf_cpu)

        for nprobe in self.params.nprobe_values:
            ivf_cpu.nprobe = min(nprobe, safe_nlist)
            _set_gpu_nprobe(ivf_gpu, min(nprobe, safe_nlist))
            times, labels = _batched_search(
                ivf_gpu,
                self.query_embeddings,
                k,
                batch_size,
                desc=f"IVF search (nprobe={nprobe})",
                position=1,
            )
            recall = self._compute_recall(labels, k)
            self.results_gpu1.append(
                BenchmarkResult(
                    index_type="IVF",
                    n_vectors=self.n_vectors,
                    latency_ms=float(np.mean(times)),
                    latency_p95_ms=float(np.percentile(times, 95)),
                    recall_at_k=recall,
                    k=k,
                    params={"nlist": safe_nlist, "nprobe": nprobe},
                    n_queries=self.n_queries,
                )
            )

        del ivf_gpu, ivf_cpu
        _free_gpu_objects(res)

        ivfpq_cpu, safe_nlist = _train_ivfpq_gpu(
            self.embeddings,
            self.params.nlist,
            self.params.M_pq,
            self.params.nbits,
            self.dimension,
            gpu_id,
        )

        if self.params.M_pq > 48:
            gpu_config = faiss.GpuIndexIVFPQConfig()
            gpu_config.useFloat16LookupTables = True
            gpu_config.device = gpu_id
            res2 = _make_gpu_res(gpu_id, settings.faiss.temp_memory_mb)
            ivfpq_gpu = faiss.GpuIndexIVFPQ(
                res2,
                self.dimension,
                safe_nlist,
                self.params.M_pq,
                self.params.nbits,
                faiss.METRIC_INNER_PRODUCT,
                gpu_config,
            )
            ivfpq_gpu.copyFrom(ivfpq_cpu)
        else:
            res2 = _make_gpu_res(gpu_id, settings.faiss.temp_memory_mb)
            ivfpq_gpu = faiss.index_cpu_to_gpu(res2, gpu_id, ivfpq_cpu)

        for nprobe in self.params.nprobe_values:
            ivfpq_cpu.nprobe = min(nprobe, safe_nlist)
            _set_gpu_nprobe(ivfpq_gpu, min(nprobe, safe_nlist))
            times, labels = _batched_search(
                ivfpq_gpu,
                self.query_embeddings,
                k,
                batch_size,
                desc=f"IVFPQ search (nprobe={nprobe})",
                position=1,
            )
            recall = self._compute_recall(labels, k)
            self.results_gpu1.append(
                BenchmarkResult(
                    index_type="IVF+PQ",
                    n_vectors=self.n_vectors,
                    latency_ms=float(np.mean(times)),
                    latency_p95_ms=float(np.percentile(times, 95)),
                    recall_at_k=recall,
                    k=k,
                    params={"nlist": safe_nlist, "nprobe": nprobe, "M_pq": self.params.M_pq},
                    n_queries=self.n_queries,
                )
            )

            _free_gpu_objects(ivfpq_gpu, ivfpq_cpu, res2)

    def run(self) -> list[BenchmarkResult]:
        print("Starting multi-GPU benchmark")
        print(f"Vectors: {self.n_vectors:,}, Queries: {self.n_queries}")
        print(f"GPU0: Flat + HNSW | GPU1: IVF + IVFPQ")

        tqdm.set_lock(threading.RLock())
        t0 = time.perf_counter()
        thread0 = threading.Thread(target=self._run_gpu0, name="gpu0")
        thread1 = threading.Thread(target=self._run_gpu1, name="gpu1")

        thread0.start()
        thread1.start()
        thread0.join()
        thread1.join()

        elapsed_min = (time.perf_counter() - t0) / 60
        print(f"Multi-GPU benchmark complete in {elapsed_min:.1f} min")
        return self.results_gpu0 + self.results_gpu1
