<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Tôi muốn chạy với máy có cấu hình 2xA100 80GB(total 160GB VRAM) với tốc độ nhanh nhất có thể mà vẫn chính xác với task 1M rows. Tôi cần cập nhật code như thế nào, giả sử tôi chưa Fix: 4 thay đổi cần làm ngay ở bước vừa rồi(), tôi vẫn muốn setup chạy tương tự như 100K row nhưng chạy nhanh hơn thôi

Với **2×A100 80GB = 160GB VRAM tổng, 16 vCPU, 128GB RAM** — đây là config lý tưởng cho 1M rows. Chiến lược: **GPU 0 chạy HNSW + IVF, GPU 1 chạy IVF+PQ + Flat ground truth, song song hoàn toàn**.

## Tổng quan thay đổi

Có 3 file cần sửa và 1 file mới:

```
src/rag_benchmark/
├── config/settings.py          ← Thêm multi-GPU config
├── benchmark/
│   ├── params.py               ← Chỉnh nlist cho 1M
│   ├── runner.py               ← GPU-aware + parallel
│   └── multi_gpu_runner.py     ← FILE MỚI: chạy song song 2 GPU
└── indexing/
    ├── ivf_index.py            ← GPU Clustering train
    └── ivfpq_index.py          ← float16 + GPU 1
```


***

## `config/settings.py` — Thêm multi-GPU

```python
# src/rag_benchmark/config/settings.py

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class FAISSSettings(BaseModel):
    use_gpu: bool = True
    gpu_id: int = 0                  # GPU mặc định
    gpu_ids: list[int] = [0, 1]      # Tất cả GPU có sẵn — 2×A100
    temp_memory_mb: int = 1024       # 1GB temp memory mỗi GPU (A100 80GB dư sức)

    # Clustering
    kmeans_niter: int = 20
    kmeans_max_points_per_centroid: int = 256

    # Embedding batch
    embedding_batch_size: int = 1024  # Tăng từ 256 → 1024 với A100


class BenchmarkSettings(BaseModel):
    n_samples_list: list[int] = [10_000, 100_000, 1_000_000]
    top_k: int = 10
    ann_top: int = 50
    n_queries: int = 100             # Số query để benchmark


class Settings(BaseSettings):
    faiss: FAISSSettings = FAISSSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


settings = Settings()
```


***

## `benchmark/params.py` — nlist hợp lý cho 1M

```python
# src/rag_benchmark/benchmark/params.py

import math
from dataclasses import dataclass, field


@dataclass
class BenchmarkParams:
    nlist: int
    nprobe_values: list[int]
    M_pq: int
    nbits: int
    min_train_vectors: int
    M_hnsw: int
    ef_construction: int
    ef_search_values: list[int]


def get_optimal_params(n: int, dimension: int = 384) -> BenchmarkParams:

    # ── nlist ────────────────────────────────────────────────────────────
    # Với 1M: 4096 là sweet-spot — recall tốt, train ~10 phút thay vì 6 giờ
    if n <= 10_000:
        nlist = max(25, int(4 * math.sqrt(n)))
    elif n <= 100_000:
        nlist = max(100, int(4 * math.sqrt(n)))
    else:
        # 1M: KHÔNG dùng 16×sqrt(N)=16000 — quá chậm để train
        # 4096 là giới hạn thực tế cho 1M với GPU
        nlist = 4096

    nlist = min(nlist, n // 39)

    # ── nprobe sweep ─────────────────────────────────────────────────────
    # Sparse hơn ở 1M (vẫn đủ điểm để vẽ đường cong)
    if n >= 1_000_000:
        raw = [1, 4, 8, 16, 32, 64, 128, 256, 512, nlist]
    else:
        raw = [1, 4, 8, 16, 32, 64, 128, 256, 512, nlist]
    nprobe_values = sorted(set(v for v in raw if v <= nlist))

    # ── PQ M_pq ──────────────────────────────────────────────────────────
    # A100 shared mem = 48KB → giới hạn M_pq=48 (float32) hoặc 96 (float16)
    max_M_pq = dimension // 4
    candidates = [m for m in [16, 24, 32, 48, 64, 96]
                  if dimension % m == 0 and m <= max_M_pq]
    # Với 1M: dùng float16 → có thể lên M_pq=96
    M_pq = 96 if 96 in candidates else (48 if 48 in candidates else candidates[-1])

    # ── HNSW ─────────────────────────────────────────────────────────────
    if n < 100_000:
        M_hnsw, ef_construction = 32, 200
    else:
        # 1M: M=48 balance giữa memory và recall
        # A100 80GB → 1M × 384 × 4B × overhead ≈ 12GB — fit thoải mái
        M_hnsw, ef_construction = 48, 200   # efC=200 thay vì 400 để build nhanh hơn

    ef_search_values = [v for v in [16, 32, 64, 128, 256, 512] if v <= n]

    return BenchmarkParams(
        nlist=nlist,
        nprobe_values=nprobe_values,
        M_pq=M_pq,
        nbits=8,
        min_train_vectors=39 * nlist,
        M_hnsw=M_hnsw,
        ef_construction=ef_construction,
        ef_search_values=ef_search_values,
    )
```


***

## `benchmark/multi_gpu_runner.py` — FILE MỚI: song song 2 GPU

Đây là thay đổi quan trọng nhất — **GPU 0 và GPU 1 chạy song song**, tổng thời gian = max(GPU0, GPU1) thay vì tổng:

```python
# src/rag_benchmark/benchmark/multi_gpu_runner.py

"""
2×A100 parallel benchmark runner.

GPU 0: Flat (ground truth) + HNSW
GPU 1: IVF + IVF+PQ

Timeline:
  GPU0: [Flat GT] ──────── [HNSW sweep ×6 efSearch] ───────────────
  GPU1:           [IVF train+sweep] [IVF+PQ train+sweep] ──────────
  Total ≈ max(GPU0_time, GPU1_time)  vs  sequential = GPU0 + GPU1
"""

import time
import threading
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Optional

from ..config.settings import settings
from ..benchmark.params import BenchmarkParams
from .runner import BenchmarkResult


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_gpu_res(gpu_id: int, temp_mb: int = 1024) -> faiss.StandardGpuResources:
    res = faiss.StandardGpuResources()
    res.setTempMemory(temp_mb * 1024 * 1024)
    return res


def _train_ivf_gpu(
    embeddings: np.ndarray,
    nlist: int,
    dimension: int,
    gpu_id: int,
) -> faiss.IndexIVFFlat:
    """Train IVF với GPU Clustering — nhanh hơn CPU 10-20×."""
    res = _make_gpu_res(gpu_id)
    n = len(embeddings)
    safe_nlist = min(nlist, n // 39)

    # K-means trên GPU
    print(f"  [GPU{gpu_id}] K-means clustering {n:,} → {safe_nlist} centroids...")
    clus = faiss.Clustering(dimension, safe_nlist)
    clus.verbose = False
    clus.niter = settings.faiss.kmeans_niter
    clus.max_points_per_centroid = settings.faiss.kmeans_max_points_per_centroid

    flat_gpu = faiss.GpuIndexFlatL2(res, dimension)
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
) -> faiss.IndexIVFPQ:
    """Train IVF+PQ với GPU Clustering + float16 lookup."""
    res = _make_gpu_res(gpu_id)
    n = len(embeddings)
    safe_nlist = min(nlist, n // 39)

    print(f"  [GPU{gpu_id}] K-means IVF+PQ {n:,} → {safe_nlist} centroids, M_pq={M_pq}...")
    clus = faiss.Clustering(dimension, safe_nlist)
    clus.verbose = False
    clus.niter = settings.faiss.kmeans_niter

    flat_gpu = faiss.GpuIndexFlatL2(res, dimension)
    clus.train(embeddings, flat_gpu)

    centroids = faiss.vector_float_to_array(clus.centroids).reshape(safe_nlist, dimension)
    quantizer = faiss.IndexFlatIP(dimension)
    quantizer.add(centroids)

    cpu_index = faiss.IndexIVFPQ(quantizer, dimension, safe_nlist, M_pq, nbits)
    cpu_index.is_trained = True
    cpu_index.add(embeddings)
    return cpu_index, safe_nlist


# ── Multi-GPU Runner ─────────────────────────────────────────────────────────

class MultiGPUBenchmarkRunner:
    """
    Chạy benchmark song song trên 2 GPU.
    GPU 0: Flat ground truth + HNSW
    GPU 1: IVF + IVF+PQ
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        params: BenchmarkParams,
        gpu_ids: list[int] = None,
    ):
        self.embeddings = embeddings.astype("float32")
        self.query_embeddings = query_embeddings.astype("float32")
        self.params = params
        self.dimension = embeddings.shape[^1]
        self.n = len(embeddings)
        self.gpu_ids = gpu_ids or settings.faiss.gpu_ids  # [0, 1]
        self.gpu0, self.gpu1 = self.gpu_ids[^0], self.gpu_ids[^1]

        self.results_gpu0: list[BenchmarkResult] = []
        self.results_gpu1: list[BenchmarkResult] = []
        self.ground_truth: Optional[np.ndarray] = None

    def _compute_recall(self, retrieved: np.ndarray, k: int) -> float:
        recalls = []
        for i, gt in enumerate(self.ground_truth):
            gt_set = set(gt[:k].tolist())
            ret_set = set(retrieved[i][:k].tolist())
            recalls.append(len(ret_set & gt_set) / len(gt_set))
        return float(np.mean(recalls))

    # ── GPU 0: Flat + HNSW ──────────────────────────────────────────────

    def _run_gpu0(self):
        """GPU 0: compute ground truth rồi benchmark HNSW."""
        gpu_id = self.gpu0
        res = _make_gpu_res(gpu_id)
        k = settings.benchmark.top_k

        # ── Flat ground truth ────────────────────────────────────────
        print(f"[GPU{gpu_id}] Computing Flat ground truth ({self.n:,} vectors)...")
        t0 = time.perf_counter()
        flat_cpu = faiss.IndexFlatIP(self.dimension)
        flat_cpu.add(self.embeddings)
        flat_gpu = faiss.index_cpu_to_gpu(res, gpu_id, flat_cpu)

        all_labels = []
        batch = 50
        for i in range(0, len(self.query_embeddings), batch):
            _, lbl = flat_gpu.search(self.query_embeddings[i:i+batch], k)
            all_labels.append(lbl)
        self.ground_truth = np.vstack(all_labels)
        flat_latency = (time.perf_counter() - t0) / len(self.query_embeddings) * 1000

        self.results_gpu0.append(BenchmarkResult(
            index_type="Flat", params="baseline",
            latency_ms=round(flat_latency, 3), recall_at_k=1.0, n=self.n,
        ))
        print(f"[GPU{gpu_id}] Flat done. latency={flat_latency:.2f}ms")

        # Giải phóng Flat để nhường VRAM cho HNSW
        del flat_gpu, flat_cpu

        # ── HNSW ─────────────────────────────────────────────────────
        print(f"[GPU{gpu_id}] Building HNSW M={self.params.M_hnsw} efC={self.params.ef_construction}...")
        hnsw_cpu = faiss.IndexHNSWFlat(self.dimension, self.params.M_hnsw)
        hnsw_cpu.hnsw.efConstruction = self.params.ef_construction
        t_add = time.perf_counter()
        hnsw_cpu.add(self.embeddings)
        print(f"[GPU{gpu_id}] HNSW built in {(time.perf_counter()-t_add)/60:.1f} min")

        for ef in self.params.ef_search_values:
            # HNSW không có GPU implementation trong FAISS → chạy CPU multi-thread
            faiss.omp_set_num_threads(8)  # Dùng 8 cores cho HNSW
            hnsw_cpu.hnsw.efSearch = ef

            t0 = time.perf_counter()
            _, labels = hnsw_cpu.search(self.query_embeddings, k)
            latency = (time.perf_counter() - t0) / len(self.query_embeddings) * 1000

            recall = self._compute_recall(labels, k)
            self.results_gpu0.append(BenchmarkResult(
                index_type="HNSW", params=f"M={self.params.M_hnsw},ef={ef}",
                latency_ms=round(latency, 3), recall_at_k=round(recall, 4), n=self.n,
            ))
            print(f"[GPU{gpu_id}] HNSW ef={ef}: latency={latency:.2f}ms recall={recall:.3f}")

    # ── GPU 1: IVF + IVF+PQ ─────────────────────────────────────────────

    def _run_gpu1(self):
        """GPU 1: benchmark IVF và IVF+PQ song song với GPU 0."""
        gpu_id = self.gpu1
        k = settings.benchmark.top_k

        # Chờ ground truth sẵn sàng (do GPU 0 compute)
        while self.ground_truth is None:
            time.sleep(2)
        print(f"[GPU{gpu_id}] Ground truth ready, starting IVF...")

        # ── IVF ──────────────────────────────────────────────────────
        ivf_cpu, safe_nlist = _train_ivf_gpu(
            self.embeddings, self.params.nlist, self.dimension, gpu_id
        )
        res1 = _make_gpu_res(gpu_id)
        ivf_gpu = faiss.index_cpu_to_gpu(res1, gpu_id, ivf_cpu)

        for nprobe in self.params.nprobe_values:
            ivf_cpu.nprobe = min(nprobe, safe_nlist)
            # Sync nprobe sang GPU index
            faiss.downcast_index(ivf_gpu).setNumProbes(min(nprobe, safe_nlist))

            t0 = time.perf_counter()
            _, labels = ivf_gpu.search(self.query_embeddings, k)
            latency = (time.perf_counter() - t0) / len(self.query_embeddings) * 1000

            recall = self._compute_recall(labels, k)
            self.results_gpu1.append(BenchmarkResult(
                index_type="IVF", params=f"nprobe={nprobe}",
                latency_ms=round(latency, 3), recall_at_k=round(recall, 4), n=self.n,
            ))
            print(f"[GPU{gpu_id}] IVF nprobe={nprobe}: latency={latency:.2f}ms recall={recall:.3f}")

        del ivf_gpu, ivf_cpu

        # ── IVF+PQ ───────────────────────────────────────────────────
        ivfpq_cpu, safe_nlist = _train_ivfpq_gpu(
            self.embeddings, self.params.nlist, self.params.M_pq,
            self.params.nbits, self.dimension, gpu_id
        )
        res1b = _make_gpu_res(gpu_id)

        # Float16 nếu M_pq > 48
        if self.params.M_pq > 48:
            gpu_config = faiss.GpuIndexIVFPQConfig()
            gpu_config.useFloat16LookupTables = True
            gpu_config.device = gpu_id
            ivfpq_gpu = faiss.GpuIndexIVFPQ(
                res1b, self.dimension, safe_nlist, self.params.M_pq,
                self.params.nbits, faiss.METRIC_INNER_PRODUCT, gpu_config
            )
            ivfpq_gpu.copyFrom(ivfpq_cpu)
        else:
            ivfpq_gpu = faiss.index_cpu_to_gpu(res1b, gpu_id, ivfpq_cpu)

        for nprobe in self.params.nprobe_values:
            ivfpq_cpu.nprobe = min(nprobe, safe_nlist)

            t0 = time.perf_counter()
            _, labels = ivfpq_gpu.search(self.query_embeddings, k)
            latency = (time.perf_counter() - t0) / len(self.query_embeddings) * 1000

            recall = self._compute_recall(labels, k)
            self.results_gpu1.append(BenchmarkResult(
                index_type="IVF+PQ", params=f"nprobe={nprobe}",
                latency_ms=round(latency, 3), recall_at_k=round(recall, 4), n=self.n,
            ))
            print(f"[GPU{gpu_id}] IVF+PQ nprobe={nprobe}: latency={latency:.2f}ms recall={recall:.3f}")

    # ── Orchestrator ─────────────────────────────────────────────────────

    def run(self) -> list[BenchmarkResult]:
        """Khởi động 2 thread song song, join khi cả 2 xong."""
        print(f"\n{'='*60}")
        print(f"🚀 Multi-GPU Benchmark: {self.n:,} vectors, 2×A100")
        print(f"   GPU 0: Flat + HNSW")
        print(f"   GPU 1: IVF + IVF+PQ (parallel)")
        print(f"{'='*60}\n")

        t_total = time.perf_counter()

        thread0 = threading.Thread(target=self._run_gpu0, name="GPU0-Flat+HNSW")
        thread1 = threading.Thread(target=self._run_gpu1, name="GPU1-IVF+PQ")

        thread0.start()
        thread1.start()
        thread0.join()
        thread1.join()

        elapsed = (time.perf_counter() - t_total) / 60
        print(f"\n✅ Done in {elapsed:.1f} min (both GPUs)")

        return self.results_gpu0 + self.results_gpu1
```


***

## `scripts/run_benchmark.py` — Gọi MultiGPURunner

```python
# scripts/run_benchmark.py — thay phần benchmark với N=1M

from rag_benchmark.benchmark.multi_gpu_runner import MultiGPUBenchmarkRunner
from rag_benchmark.benchmark.params import get_optimal_params

def main():
    # ... load, embed như cũ ...

    params = get_optimal_params(n=len(embeddings), dimension=dimension)
    print(f"Auto params: nlist={params.nlist}, M_pq={params.M_pq}, M_hnsw={params.M_hnsw}")

    # Chọn runner: 2 GPU thì dùng MultiGPU, 1 GPU dùng runner cũ
    n_gpus = faiss.get_num_gpus()
    print(f"Available GPUs: {n_gpus}")

    if n_gpus >= 2:
        runner = MultiGPUBenchmarkRunner(
            embeddings=embeddings,
            query_embeddings=query_embeddings,
            params=params,
            gpu_ids=[0, 1],
        )
        results = runner.run()
    else:
        # Fallback runner cũ (single GPU)
        ...

    # Xuất báo cáo như cũ
    reporter.plot_latency_vs_recall(...)
```


***

## Ước tính thời gian trên 2×A100

| Bước | GPU | Thời gian |
| :-- | :-- | :-- |
| Embedding 1M (batch=1024) | GPU 0 | ~8 phút |
| Flat ground truth | GPU 0 | ~3 phút |
| HNSW build M=48 | CPU (8 cores) | ~25 phút |
| HNSW search ×6 efSearch | CPU | ~5 phút |
| K-means IVF 4096 centroids | GPU 1 | ~8 phút |
| IVF search ×9 nprobe | GPU 1 | ~3 phút |
| K-means IVF+PQ 4096 | GPU 1 | ~10 phút |
| IVF+PQ search ×9 nprobe | GPU 1 | ~3 phút |
| **Tổng (song song)** | **cả 2** | **~35–40 phút** |

So với chạy tuần tự trên 1 GPU (8+ giờ) → **giảm ~12–15×** nhờ GPU Clustering + parallel execution + nlist=4096 thay vì 16000.

<div align="center">⁂</div>

[^1]: image.jpg

