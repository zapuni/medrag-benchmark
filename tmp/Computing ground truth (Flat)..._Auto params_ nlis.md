<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Computing ground truth (Flat)...

Auto params: nlist=1264, nprobe=[1, 4, 8, 16, 32, 64, 128, 256, 512], M_pq=96, nbits=8, M_hnsw=48, efC=200, efS=[16, 32, 64, 128, 256, 512]
Traceback (most recent call last):
  File "/workspace/giapcn/medrag/scripts/run_benchmark.py", line 92, in <module>
    main()
  File "/workspace/giapcn/medrag/scripts/run_benchmark.py", line 66, in main
    results.append(runner.benchmark_ivfpq(nlist=params.nlist, M_pq=params.M_pq, nprobe=nprobe))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/src/rag_benchmark/benchmark/runner.py", line 91, in benchmark_ivfpq
    index = IVFPQIndexBuilder(self.dimension, nlist=nlist, M_pq=M_pq).build(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/src/rag_benchmark/indexing/ivfpq_index.py", line 39, in build
    self._gpu_index = faiss.index_cpu_to_gpu(res, settings.faiss.gpu_id, self._cpu_index)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/.venv/lib/python3.12/site-packages/faiss/swigfaiss.py", line 15473, in index_cpu_to_gpu
    return _swigfaiss.index_cpu_to_gpu(provider, device, index, options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Error in void faiss::gpu::GpuIndexIVFPQ::verifyPQSettings_() const at /project/faiss/faiss/gpu/GpuIndexIVFPQ.cu:607: Error: 'requiredSmemSize <= getMaxSharedMemPerBlock(config_.device)' failed: Device 0 has 49152 bytes of shared memory, while 8 bits per code and 96 sub-quantizers requires 98304 bytes. Consider useFloat16LookupTables and/or reduce parameters

Đây là giới hạn phần cứng cố định của CUDA, không phải bug.

## Nguyên nhân gốc rễ

GPU shared memory limit là **49,152 bytes (48 KiB) per block** — đây là giới hạn phần cứng của tất cả GPU NVIDIA từ Ampere trở xuống (bao gồm A40).

FAISS GPU tính shared memory theo công thức:

```
requiredSmemSize = 4 bytes × 256 centroids × M_pq sub-quantizers
                 = 4 × 256 × 96 = 98,304 bytes  ← VỚI float32
                 = 2 × 256 × 96 = 49,152 bytes  ← VỚI float16 (vừa đủ)
```

**Giới hạn GPU của FAISS** :

- `nbits=8` + float32 → M_pq tối đa = **48**
- `nbits=8` + float16 → M_pq tối đa = **96**

***

## 3 cách fix

### Cách 1 — Bật `useFloat16LookupTables` ✅ (khuyến nghị cho M_pq=96)

```python
# src/rag_benchmark/indexing/ivfpq_index.py

def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFPQIndexBuilder":
    # ... train CPU index như cũ ...

    if settings.faiss.use_gpu:
        self._res = faiss.StandardGpuResources()
        
        # Cấu hình GPU config với float16 lookup tables
        config = faiss.GpuIndexIVFPQConfig()
        config.useFloat16LookupTables = True   # ← FIX: 2 bytes thay 4 bytes
        config.device = settings.faiss.gpu_id
        
        self._gpu_index = faiss.index_cpu_to_gpu(
            self._res, settings.faiss.gpu_id, self._cpu_index
            # Lưu ý: index_cpu_to_gpu không nhận config trực tiếp
            # Dùng GpuIndexIVFPQ constructor thay thế (xem bên dưới)
        )
```

`index_cpu_to_gpu` không cho phép pass `GpuIndexIVFPQConfig`. Phải dùng cách build trực tiếp trên GPU:

```python
# src/rag_benchmark/indexing/ivfpq_index.py — FULL FIX

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class IVFPQIndexBuilder:
    def __init__(self, dimension: int, nlist: int = 100, M_pq: int = 48, nbits: int = 8):
        assert dimension % M_pq == 0, (
            f"dimension={dimension} phải chia hết cho M_pq={M_pq}."
        )
        self.dimension = dimension
        self.nlist = nlist
        self.M_pq = M_pq
        self.nbits = nbits
        self._cpu_index = None
        self._gpu_index = None
        self._res = None

    def _needs_float16(self) -> bool:
        """
        GPU shared memory limit: 48KiB = 49152 bytes.
        float32: 4 × 256 × M_pq <= 49152 → M_pq <= 48
        float16: 2 × 256 × M_pq <= 49152 → M_pq <= 96
        """
        return self.nbits == 8 and self.M_pq > 48

    def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFPQIndexBuilder":
        n = len(vectors)
        safe_nlist = min(self.nlist, n // 39)
        if safe_nlist < self.nlist:
            print(f"⚠️  nlist: {self.nlist} → {safe_nlist}")
            self.nlist = safe_nlist

        # Train trên CPU trước (hiệu quả hơn train trên GPU với large N)
        cpu_quantizer = faiss.IndexFlatIP(self.dimension)
        self._cpu_index = faiss.IndexIVFPQ(
            cpu_quantizer, self.dimension, self.nlist, self.M_pq, self.nbits
        )
        # Sample để train nếu N quá lớn
        if n > 100 * self.nlist:
            train_size = min(n, 100 * self.nlist)
            idx = np.random.choice(n, train_size, replace=False)
            print(f"Training on {train_size:,}/{n:,} samples...")
            self._cpu_index.train(vectors[idx])
        else:
            self._cpu_index.train(vectors)

        self._cpu_index.add(vectors)
        self._cpu_index.nprobe = min(nprobe, self.nlist)

        # Chuyển lên GPU nếu cần
        if settings.faiss.use_gpu:
            self._res = faiss.StandardGpuResources()

            if self._needs_float16():
                # Dùng GpuIndexIVFPQ constructor với useFloat16LookupTables=True
                print(f"ℹ️  M_pq={self.M_pq} > 48, bật useFloat16LookupTables...")
                
                gpu_config = faiss.GpuIndexIVFPQConfig()
                gpu_config.useFloat16LookupTables = True   # ← KEY FIX
                gpu_config.device = settings.faiss.gpu_id

                # Build GPU index từ đầu với config
                gpu_quantizer = faiss.GpuIndexFlatL2(self._res, self.dimension, faiss.GpuIndexFlatConfig())
                self._gpu_index = faiss.GpuIndexIVFPQ(
                    self._res,
                    self.dimension,
                    self.nlist,
                    self.M_pq,
                    self.nbits,
                    faiss.METRIC_INNER_PRODUCT,
                    gpu_config
                )
                # Copy vectors đã train/add từ CPU index sang GPU
                self._gpu_index.copyFrom(self._cpu_index)
            else:
                # M_pq <= 48, dùng float32 bình thường
                self._gpu_index = faiss.index_cpu_to_gpu(
                    self._res, settings.faiss.gpu_id, self._cpu_index
                )

        return self

    def search(self, query: np.ndarray, k: int = 50, nprobe: int = None):
        if nprobe is not None:
            effective_nprobe = min(nprobe, self.nlist)
            self._cpu_index.nprobe = effective_nprobe
            if self._gpu_index is not None and hasattr(self._gpu_index, 'setNumProbes'):
                self._gpu_index.setNumProbes(effective_nprobe)
        idx = self._gpu_index if self._gpu_index is not None else self._cpu_index
        return idx.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._cpu_index, str(path))

    @classmethod
    def load(cls, path: Path, dimension: int, nlist: int, M_pq: int = 48) -> "IVFPQIndexBuilder":
        builder = cls(dimension, nlist=nlist, M_pq=M_pq)
        builder._cpu_index = faiss.read_index(str(path))
        if settings.faiss.use_gpu:
            builder._res = faiss.StandardGpuResources()
            if builder._needs_float16():
                gpu_config = faiss.GpuIndexIVFPQConfig()
                gpu_config.useFloat16LookupTables = True
                gpu_config.device = settings.faiss.gpu_id
                builder._gpu_index = faiss.GpuIndexIVFPQ(
                    builder._res, dimension, nlist, M_pq, 8,
                    faiss.METRIC_INNER_PRODUCT, gpu_config
                )
                builder._gpu_index.copyFrom(builder._cpu_index)
            else:
                builder._gpu_index = faiss.index_cpu_to_gpu(
                    builder._res, settings.faiss.gpu_id, builder._cpu_index
                )
        return builder
```


***

### Cách 2 — Giảm M_pq ≤ 48 (đơn giản nhất)

Nếu không cần nén tối đa, M_pq=48 vẫn nén tốt:

```python
# params.py — cap M_pq tối đa 48 cho GPU float32
MAX_M_PQ_GPU_FLOAT32 = 48
MAX_M_PQ_GPU_FLOAT16 = 96

# Với dim=384:
# M_pq=48 → mỗi sub-vector = 384/48 = 8 dim → compression 32×
# M_pq=96 → mỗi sub-vector = 384/96 = 4 dim → compression 32× (nbits=8)
```


***

### Cách 3 — Chạy IVF+PQ hoàn toàn trên CPU

Phù hợp cho benchmark để so sánh độ chính xác (recall) mà không bị giới hạn GPU:

```python
# Trong IVFPQIndexBuilder.build(), bỏ qua GPU nếu M_pq > 96:
if settings.faiss.use_gpu and self.M_pq <= 96:
    ...  # GPU path
else:
    print("⚠️  Chạy IVF+PQ trên CPU (M_pq > 96 hoặc GPU disabled)")
    # self._gpu_index = None → search() sẽ dùng self._cpu_index
```


***

## Bảng giới hạn M_pq theo GPU shared memory

| nbits | useFloat16 | Shared mem per centroid | Max M_pq | Ghi chú |
| :-- | :-- | :-- | :-- | :-- |
| 8 | False (float32) | 4 bytes × 256 = 1024B | **48** | Default |
| 8 | **True** (float16) | 2 bytes × 256 = 512B | **96** | ← Cần bật |
| 4 | False | 4 bytes × 16 = 64B | Không giới hạn | Ít dùng |

Với A40 (48KiB shared memory/block), A40 = Ampere nên giới hạn này là cố định .
<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/facebookresearch/faiss/issues/927

[^2]: https://www.facebook.com/groups/faissusers/posts/1143321559425462/

[^3]: https://github.com/facebookresearch/faiss/issues/269

[^4]: https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/

[^5]: https://github.com/facebookresearch/faiss/issues/3207

[^6]: https://milvus.io/ai-quick-reference/what-optimizations-do-libraries-like-faiss-implement-to-maintain-high-throughput-for-vector-search-on-cpus-and-how-do-these-differ-when-utilizing-gpu-acceleration

[^7]: https://github.com/facebookresearch/faiss/issues/1641

[^8]: https://arxiv.org/html/2401.08281v4

[^9]: https://opensearch.org/blog/optimizing-opensearch-with-fp16-quantization/

[^10]: https://github.com/facebookresearch/faiss/issues/1178

[^11]: https://discuss.huggingface.co/t/runtimeerror-error-in-void-faiss-allocmemoryspace/1358

[^12]: https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint

[^13]: https://github.com/milvus-io/milvus/issues/6723

[^14]: https://bge-model.com/tutorial/3_Indexing/3.1.4.html

[^15]: https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/

