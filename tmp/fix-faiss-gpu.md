<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Loading 1,000 documents from MedRAG/wikipedia...

`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'MedRAG/wikipedia' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.

Loading:   0%|          | 0/1000 [00:00<?, ?it/s]
Loading:   0%|          | 1/1000 [00:00<05:58,  2.79it/s]
Loading:  49%|████▉     | 491/1000 [00:00<00:00, 1396.84it/s]
Loading: 100%|██████████| 1000/1000 [00:00<00:00, 1897.96it/s]
Loaded 1,000 documents
/workspace/giapcn/medrag/src/rag_benchmark/adapters/embedding/sentence_transformer_adapter.py:14: FutureWarning: The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.
  self._dimension = self.model.get_sentence_embedding_dimension()

Batches:   0%|          | 0/2 [00:00<?, ?it/s]
Batches:  50%|█████     | 1/2 [00:01<00:01,  1.54s/it]
Batches: 100%|██████████| 2/2 [00:01<00:00,  1.24it/s]
Batches: 100%|██████████| 2/2 [00:01<00:00,  1.09it/s]
Traceback (most recent call last):
  File "/workspace/giapcn/medrag/scripts/build_index.py", line 51, in <module>
    main()
  File "/workspace/giapcn/medrag/scripts/build_index.py", line 34, in main
    index = HNSWIndexBuilder(embedder.dimension, M=32).build(embeddings)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/src/rag_benchmark/indexing/hnsw_index.py", line 23, in build
    self._gpu_index = faiss.index_cpu_to_gpu(res, settings.faiss.gpu_id, self._cpu_index)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/.venv/lib/python3.12/site-packages/faiss/swigfaiss.py", line 15473, in index_cpu_to_gpu
    return _swigfaiss.index_cpu_to_gpu(provider, device, index, options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Error in virtual faiss::Index* faiss::gpu::ToGpuCloner::clone_Index(const faiss::Index*) at /project/faiss/faiss/gpu/GpuCloner.cpp:253: This index type is not implemented on GPU.

Lỗi này là **thiết kế của FAISS** — `IndexHNSWFlat` không hỗ trợ GPU. Đây là giới hạn cố định của thư viện .

## Nguyên nhân gốc rễ

FAISS GPU **chỉ hỗ trợ 4 loại index** :


| GPU Index | CPU Equivalent | Ghi chú |
| :-- | :-- | :-- |
| `GpuIndexFlat` | `IndexFlat` | ✅ GPU |
| `GpuIndexIVFFlat` | `IndexIVFFlat` | ✅ GPU |
| `GpuIndexIVFPQ` | `IndexIVFPQ` | ✅ GPU |
| `GpuIndexIVFScalarQuantizer` | `IndexIVFScalarQuantizer` | ✅ GPU |
| ~~`GpuIndexHNSW`~~ | `IndexHNSWFlat` | ❌ **KHÔNG tồn tại** |

**HNSW là thuật toán graph-based** — cấu trúc dữ liệu pointer-heavy không phù hợp với SIMD/CUDA của GPU . Giải pháp đúng: **HNSW chạy trên CPU**, còn Flat/IVF/IVF+PQ mới chạy trên GPU .

***

## Fix toàn bộ các index builders

### `hnsw_index.py` — Xóa GPU, chạy CPU (đây là đúng)

```python
# src/rag_benchmark/indexing/hnsw_index.py

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class HNSWIndexBuilder:
    """
    HNSW Index — Graph-based ANN (PTIT slide 9: "QUAN TRỌNG NHẤT").

    ⚠️  HNSW KHÔNG hỗ trợ GPU trong FAISS — chạy CPU là đúng thiết kế.
    HNSW đã rất nhanh trên CPU nhờ cấu trúc graph (O(log N)).
    Chỉ IVF, IVFFlat, IVFPQ mới chuyển được lên GPU.

    Tuning (slide 24):
      M         : số neighbor mỗi node → cao thì accuracy↑, RAM↑
      efSearch  : beam width lúc search → cao thì recall↑, latency↑
    """

    def __init__(self, dimension: int, M: int = 32, ef_construction: int = 200):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self._index: faiss.IndexHNSWFlat | None = None

    def build(self, vectors: np.ndarray) -> "HNSWIndexBuilder":
        self._index = faiss.IndexHNSWFlat(self.dimension, self.M)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.add(vectors)
        return self

    def search(
        self, query: np.ndarray, k: int = 50, ef_search: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._index is not None, "Gọi .build() trước"
        self._index.hnsw.efSearch = ef_search
        return self._index.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path, dimension: int, M: int = 32) -> "HNSWIndexBuilder":
        builder = cls(dimension, M=M)
        builder._index = faiss.read_index(str(path))
        return builder
```


***

### `flat_index.py` — GPU ✅

```python
# src/rag_benchmark/indexing/flat_index.py

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class FlatIndexBuilder:
    """
    Flat (brute-force) — baseline, accuracy=100%.
    GPU: GpuIndexFlatIP — tăng tốc đáng kể khi N lớn.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._cpu_index: faiss.IndexFlatIP | None = None
        self._gpu_index = None

    def build(self, vectors: np.ndarray) -> "FlatIndexBuilder":
        self._cpu_index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine với normalized)
        self._cpu_index.add(vectors)
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(
                res, settings.faiss.gpu_id, self._cpu_index
            )
        return self

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self._gpu_index if self._gpu_index is not None else self._cpu_index
        return idx.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._cpu_index, str(path))
```


***

### `ivf_index.py` — GPU ✅

```python
# src/rag_benchmark/indexing/ivf_index.py

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class IVFIndexBuilder:
    """
    IVF (Inverted File Index) — clustering-based ANN.
    GPU: GpuIndexIVFFlat — training + search đều tăng tốc.

    Tuning (slide 25):
      nlist  : số cluster ≈ sqrt(N)
      nprobe : số cluster scan → cao thì recall↑, latency↑
    """

    def __init__(self, dimension: int, nlist: int = 100):
        self.dimension = dimension
        self.nlist = nlist
        self._cpu_index: faiss.IndexIVFFlat | None = None
        self._gpu_index = None
        self._res = None

    def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFIndexBuilder":
        assert len(vectors) >= self.nlist * 39, (
            f"Cần ít nhất {self.nlist * 39} vectors để train {self.nlist} clusters. "
            f"Hiện có {len(vectors)}. Giảm nlist hoặc tăng N."
        )

        if settings.faiss.use_gpu:
            # Build trực tiếp trên GPU — nhanh hơn build CPU rồi chuyển
            self._res = faiss.StandardGpuResources()
            config = faiss.GpuIndexIVFFlatConfig()
            config.device = settings.faiss.gpu_id
            gpu_quantizer = faiss.GpuIndexFlatIP(
                self._res,
                faiss.GpuIndexFlatConfig()  # quantizer cũng trên GPU
            )
            # Dùng CPU quantizer đơn giản hơn, chuyển index sau
            cpu_quantizer = faiss.IndexFlatIP(self.dimension)
            self._cpu_index = faiss.IndexIVFFlat(
                cpu_quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._cpu_index.train(vectors)
            self._cpu_index.add(vectors)
            self._cpu_index.nprobe = nprobe
            # Chuyển lên GPU sau khi build xong
            self._gpu_index = faiss.index_cpu_to_gpu(
                self._res, settings.faiss.gpu_id, self._cpu_index
            )
        else:
            cpu_quantizer = faiss.IndexFlatIP(self.dimension)
            self._cpu_index = faiss.IndexIVFFlat(
                cpu_quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._cpu_index.train(vectors)
            self._cpu_index.add(vectors)
            self._cpu_index.nprobe = nprobe

        return self

    def search(
        self, query: np.ndarray, k: int = 50, nprobe: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if nprobe is not None:
            self._cpu_index.nprobe = nprobe
            if self._gpu_index is not None:
                # Cập nhật nprobe trên GPU index
                faiss.downcast_index(self._gpu_index).setNumProbes(nprobe)
        idx = self._gpu_index if self._gpu_index is not None else self._cpu_index
        return idx.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._cpu_index, str(path))
```


***

### `ivfpq_index.py` — GPU ✅

```python
# src/rag_benchmark/indexing/ivfpq_index.py

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class IVFPQIndexBuilder:
    """
    IVF+PQ — memory-constrained deployment (slide 11-12).
    GPU: GpuIndexIVFPQ — training + search đều tăng tốc.

    PQ nén: 384-dim × 4B = 1,536B → M_pq × 1B = 48B (32× nhỏ hơn)
    ⚠️  dimension phải chia hết cho M_pq (384 / 48 = 8 ✅)
    """

    def __init__(
        self, dimension: int, nlist: int = 100, M_pq: int = 48, nbits: int = 8
    ):
        assert dimension % M_pq == 0, (
            f"dimension={dimension} phải chia hết cho M_pq={M_pq}. "
            f"Thử M_pq=32 (384/32=12✅) hoặc M_pq=48 (384/48=8✅)"
        )
        self.dimension = dimension
        self.nlist = nlist
        self.M_pq = M_pq
        self.nbits = nbits
        self._cpu_index: faiss.IndexIVFPQ | None = None
        self._gpu_index = None
        self._res = None

    def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFPQIndexBuilder":
        assert len(vectors) >= self.nlist * 39, (
            f"Cần ít nhất {self.nlist * 39} vectors để train. Hiện có {len(vectors)}."
        )

        cpu_quantizer = faiss.IndexFlatIP(self.dimension)
        self._cpu_index = faiss.IndexIVFPQ(
            cpu_quantizer, self.dimension, self.nlist, self.M_pq, self.nbits
        )
        self._cpu_index.train(vectors)
        self._cpu_index.add(vectors)
        self._cpu_index.nprobe = nprobe

        if settings.faiss.use_gpu:
            self._res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(
                self._res, settings.faiss.gpu_id, self._cpu_index
            )

        return self

    def search(
        self, query: np.ndarray, k: int = 50, nprobe: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if nprobe is not None:
            self._cpu_index.nprobe = nprobe
        idx = self._gpu_index if self._gpu_index is not None else self._cpu_index
        return idx.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self._cpu_index, str(path))
```


***

## Tóm tắt — Index nào chạy ở đâu

| Index | GPU? | Lý do |
| :-- | :-- | :-- |
| **Flat** | ✅ GPU (`GpuIndexFlatIP`) | Brute-force hưởng lợi nhiều nhất từ GPU |
| **HNSW** | ❌ CPU only | Graph structure không map được lên CUDA |
| **IVF** | ✅ GPU (`GpuIndexIVFFlat`) | Tăng tốc cả train lẫn search |
| **IVF+PQ** | ✅ GPU (`GpuIndexIVFPQ`) | Tăng tốc cả train lẫn search |

**HNSW trên CPU A40 vẫn rất nhanh** — O(log N) với 1M vectors chỉ ~2-5ms/query, không cần GPU . Đây là design đúng, không phải limitation cần workaround.

***

## Fix nhanh `FutureWarning` trong adapter

```python
# Thay dòng deprecated:
self._dimension = self.model.get_sentence_embedding_dimension()

# Bằng dòng mới:
self._dimension = self.model.get_embedding_dimension()
```

<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

[^2]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU

[^3]: https://github.com/facebookresearch/faiss/issues/1348

[^4]: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

[^5]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU/d7d56b16fd83d947032380414818e2a39e41dc25

[^6]: https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/

[^7]: https://github.com/facebookresearch/faiss/wiki/Hybrid-CPU-GPU-search-and-multiple-GPUs/c1a6b19d9d13ef27a8806f59863319ef702b8941

[^8]: https://bge-model.com/tutorial/3_Indexing/3.1.3.html

[^9]: https://github-wiki-see.page/m/tarang-jain/faiss/wiki/GPU-Faiss-with-cuVS

[^10]: https://github.com/facebookresearch/faiss/wiki/Hybrid-CPU-GPU-search-and-multiple-GPUs

[^11]: https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/GpuIndexIVF.h

[^12]: https://deepwiki.com/facebookresearch/faiss/6.3-gpu-acceleration\&rut=c5e16b3fd1e08d115275afa11519e0d475cd77b53700428550255b89ba9a26cf

[^13]: https://openi.pcl.ac.cn/thomas-yanxin/faiss/wiki/Hybrid-CPU-GPU-search-and-multiple-GPUs

[^14]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

[^15]: https://bge-model.com/tutorial/3_Indexing/3.1.2.html

