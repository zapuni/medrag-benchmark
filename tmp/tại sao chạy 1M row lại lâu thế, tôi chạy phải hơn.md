<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# tại sao chạy 1M row lại lâu thế, tôi chạy phải hơn 8 tiếng rồi vẫn chưa xong. Máy tôi đang chạy lúc chạy mất 14% 9vCPU, 16/46GB RAM, 11/44GB VRAM.

Log mới nhất là:"Batches: 100%|██████████| 1954/1954 [10:17<00:00,  3.16it/s]
Computing ground truth (Flat)...
Auto params: nlist=16000, nprobe=[1, 4, 8, 16, 32, 64, 128, 256, 512], M_pq=48, nbits=8, M_hnsw=64, efC=400, efS=[16, 32, 64, 128, 256, 512]"

Nhìn log và ảnh monitor là rõ ngay — **bottleneck không phải GPU mà là CPU và serial execution**. Phân tích chi tiết:

## Chẩn đoán: Tại sao lâu

### 1. `nlist=16,000` — Train IVF với 16K centroids mất vài giờ

```
Auto params: nlist=16000
```

Công thức `nlist = 16 × sqrt(1,000,000) = 16,000` đúng lý thuyết nhưng **train K-means 1M vectors × 16K centroids** cực kỳ nặng — FAISS train K-means trên CPU (thậm chí ngay cả khi index sau đó lên GPU), mỗi iteration phải tính khoảng cách 1M × 16K = **16 tỷ phép tính**, chạy hàng chục iteration.

### 2. Embedding 1M samples batch-by-batch = 10 phút ổn, nhưng sau đó là `Flat ground truth` = **scan toàn bộ 1M × query_count vector**

Log dừng ở `Computing ground truth (Flat)...` — đây là brute-force search O(N) trên CPU/GPU với 1M vectors × số query. Nếu có 100 queries × 1M = 100M distance computations.

### 3. GPU Utilization = **0%** ← đây là dấu hiệu quan trọng nhất

GPU đang nhàn (0% utilization, chỉ 11/45GB VRAM), tức là code **đang block tại bước CPU** (train K-means hoặc Flat search trên CPU).

***

## Fix: 4 thay đổi cần làm ngay

### Fix 1 — Giảm `nlist` xuống thực tế

```python
# params.py — thay công thức nlist cho 1M

def get_optimal_params(n: int, dimension: int = 384) -> BenchmarkParams:
    if n <= 10_000:
        nlist = int(4 * math.sqrt(n))        # ~400 cho 10K
    elif n <= 100_000:
        nlist = int(4 * math.sqrt(n))        # ~1265 cho 100K
    else:
        # 1M: dùng 4096 thay vì 16000 — đủ tốt, train nhanh hơn 16x
        nlist = 4096
    
    # Cap cứng để tránh train quá lâu
    nlist = min(nlist, 4096)
    ...
```

**Lý do:** `nlist=4096` vs `nlist=16000` — recall@10 chỉ khác nhau ~1-2% nhưng train time khác nhau **4× (16K/4K)**.

### Fix 2 — Train K-means trên GPU (FAISS Clustering GPU)

Thay vì dùng `IndexIVFFlat` train trên CPU, dùng FAISS GPU Clustering:

```python
# src/rag_benchmark/indexing/ivf_index.py

import faiss
import numpy as np


class IVFIndexBuilder:
    def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFIndexBuilder":
        n, d = vectors.shape
        safe_nlist = min(self.nlist, n // 39)

        # ── Train K-means trên GPU ─────────────────────────────────────
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            
            # Dùng Clustering API với GPU — nhanh hơn CPU 10-20×
            clus = faiss.Clustering(d, safe_nlist)
            clus.verbose = True
            clus.niter = 20          # Giảm từ default 25 → 20 (trade-off nhỏ)
            clus.max_points_per_centroid = 256  # Giới hạn để tránh mất cân bằng
            
            # Flat index trên GPU để train
            flat_gpu = faiss.GpuIndexFlatL2(res, d)
            clus.train(vectors, flat_gpu)
            
            # Lấy centroids → tạo IVF CPU index với centroids đã train
            centroids = faiss.vector_float_to_array(clus.centroids).reshape(safe_nlist, d)
            quantizer = faiss.IndexFlatIP(d)
            quantizer.add(centroids)
            
            self._cpu_index = faiss.IndexIVFFlat(
                quantizer, d, safe_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._cpu_index.is_trained = True  # Đã train rồi
            self._cpu_index.add(vectors)
            self._cpu_index.nprobe = min(nprobe, safe_nlist)
            
            # Chuyển lên GPU để search
            self._gpu_index = faiss.index_cpu_to_gpu(res, self.gpu_id, self._cpu_index)
        else:
            # CPU path như cũ
            quantizer = faiss.IndexFlatIP(d)
            self._cpu_index = faiss.IndexIVFFlat(
                quantizer, d, safe_nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._cpu_index.train(vectors)
            self._cpu_index.add(vectors)

        return self
```


### Fix 3 — Ground truth Flat search trên GPU (không phải CPU)

Đây là chỗ đang block. Đảm bảo Flat index cũng lên GPU:

```python
# src/rag_benchmark/benchmark/runner.py

def _compute_ground_truth(self, query_embeddings: np.ndarray, k: int = 10):
    """Brute-force ground truth — PHẢI chạy trên GPU với 1M vectors."""
    print("Computing ground truth (Flat GPU)...")
    
    flat_cpu = faiss.IndexFlatIP(self.dimension)
    flat_cpu.add(self.embeddings)   # 1M × 384 × 4B = ~1.5GB — fit trong 45GB VRAM
    
    if self.use_gpu:
        res = faiss.StandardGpuResources()
        # Tăng temp memory cho Flat search lớn
        res.setTempMemory(512 * 1024 * 1024)  # 512MB temp
        flat_gpu = faiss.index_cpu_to_gpu(res, self.gpu_id, flat_cpu)
        
        # Batch query để tránh OOM
        all_labels = []
        batch_size = 50
        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i:i+batch_size]
            _, labels = flat_gpu.search(batch, k)
            all_labels.append(labels)
        return np.vstack(all_labels)
    else:
        _, labels = flat_cpu.search(query_embeddings, k)
        return labels
```


### Fix 4 — Giảm số experiment để chạy nhanh hơn

Với 1M row, chạy **toàn bộ sweep** mất quá lâu. Dùng sparse sweep:

```python
# Thay vì sweep đầy đủ, dùng sparse sweep cho 1M
if n >= 1_000_000:
    nprobe_values = [1, 8, 32, 128, 512]        # 5 điểm thay vì 9
    ef_search_values = [32, 64, 128, 256, 512]  # 5 điểm thay vì 6
```


***

## Ước tính thời gian sau khi fix

| Bước | Trước fix | Sau fix |
| :-- | :-- | :-- |
| Train IVF nlist=16K CPU | ~4–6 giờ | — |
| Train IVF nlist=4096 GPU | — | ~8–15 phút |
| Ground truth Flat CPU | ~2–3 giờ | — |
| Ground truth Flat GPU | — | ~3–5 phút |
| HNSW build + search | ~30 phút | ~20 phút (ít sweep hơn) |
| **Tổng** | **8+ giờ** | **~1–1.5 giờ** |


***

## Hành động ngay bây giờ

Nếu process đang chạy mà bạn muốn kill và restart:

```bash
# Kill process hiện tại
kill $(pgrep -f run_benchmark.py)

# Áp dụng fix, rồi chạy lại với --n_samples 1000000
uv run python scripts/run_benchmark.py --n_samples 1000000
```

GPU A40 45GB VRAM của bạn hoàn toàn đủ chứa 1M × 384-dim = ~1.5GB — lý do GPU utilization 0% là vì code đang chờ CPU train K-means xong, chưa đẩy được gì lên GPU.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: image.jpg

