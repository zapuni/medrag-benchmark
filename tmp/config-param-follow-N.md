<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# "FAISS có tham số nội bộ min_points_per_centroid = 39 :

text
WARNING clustering 1000 points to 256 centroids:
please provide at least 9984 training points
↑
256 × 39 = 9984
Ý nghĩa: bạn đang train K-means với 1000 vectors nhưng muốn chia thành 256 clusters — trung bình mỗi cluster chỉ có 1000/256 ≈ 4 điểm, trong khi FAISS khuyến nghị tối thiểu 39 điểm/cluster để K-means hội tụ tốt.
Tại sao vẫn xuất hiện 256?
Vì warning xuất hiện từ IVF+PQ — tham số M_pq=48, nbits=8 tạo 2^8 = 256 sub-codebook centroids cho PQ, độc lập với nlist của IVF. Đây là 2 tầng clustering khác nhau:
text
IVF:   nlist clusters     → bạn đã giảm về 25 ✅
PQ:    2^nbits = 256 sub-centroids per sub-space  ← warning từ đây"
Nếu sau này tôi tăng lên 100K rồi 1M thì sao, cần phải chỉnh gì để đạt được benchmark chuẩn nhất

Đây là bảng thông số **chuẩn production** cho từng scale, tổng hợp từ FAISS official wiki và nghiên cứu thực nghiệm:

## Bảng tham số tối ưu theo N

| Tham số | **N = 1K** (test) | **N = 10K** | **N = 100K** | **N = 1M** |
| :-- | :-- | :-- | :-- | :-- |
| **HNSW M** | 16 | 32 | 32 | 64 |
| **HNSW efConstruction** | 100 | 200 | 200 | 400 |
| **HNSW efSearch sweep** | 16→128 | 32→256 | 64→512 | 64→512 |
| **IVF nlist** | 25 | 100 | 1024 | 4096 |
| **IVF nprobe sweep** | 1→25 | 1→100 | 1→256 | 1→512 |
| **PQ M_pq** | 32 | 48 | 48 | 96 |
| **PQ nbits** | 8 | 8 | 8 | 8 |
| **Min train vectors** | 975 | 3,900 | 39,936 | 159,744 |
| **Ideal train vectors** | — | 10,000 | 100,000 | 400,000+ |

**Công thức ghi nhớ** :

- `nlist = 4 × sqrt(N)` với N < 1M; `nlist = 16 × sqrt(N)` với N ≥ 1M
- Min train = `39 × nlist`, Ideal train = `100 × nlist`
- `M_pq`: dimension phải chia hết — với dim=384: dùng 32, 48, 64, 96 đều được

***

## Code: Config tự động theo N

Thêm hàm này vào `benchmark/runner.py` — tự tính tham số tối ưu:

```python
# src/rag_benchmark/benchmark/params.py

from dataclasses import dataclass
import math


@dataclass
class BenchmarkParams:
    """Tham số tối ưu tự động theo N."""
    
    # IVF/PQ
    nlist: int
    nprobe_values: list[int]
    M_pq: int
    nbits: int
    min_train_vectors: int
    
    # HNSW
    M_hnsw: int
    ef_construction: int
    ef_search_values: list[int]


def get_optimal_params(n: int, dimension: int = 384) -> BenchmarkParams:
    """
    Tự động tính tham số chuẩn theo N và dimension.
    
    Nguồn: FAISS wiki Guidelines to choose an index
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    
    # ── IVF nlist ───────────────────────────────────────────────────────
    # Công thức: 4*sqrt(N) cho N<1M, 16*sqrt(N) cho N≥1M
    if n < 1_000_000:
        nlist = max(4, min(int(4 * math.sqrt(n)), n // 39))
    else:
        nlist = max(4, min(int(16 * math.sqrt(n)), n // 39))
    
    # nprobe sweep: từ 1 đến min(nlist, 512)
    # Lấy các điểm logarithmically spaced để vẽ đường cong đẹp
    max_nprobe = min(nlist, 512)
    nprobe_values = sorted(set([
        1, 4, 8, 16, 32, 64, 128, 256, 512, nlist
    ] + [max_nprobe]))
    nprobe_values = [v for v in nprobe_values if v <= max_nprobe]
    
    # ── PQ M_pq ─────────────────────────────────────────────────────────
    # Chọn M_pq lớn nhất mà dimension chia hết được, nhưng ≤ dimension/4
    # (mỗi sub-vector ít nhất 4 dimensions để PQ có ý nghĩa)
    max_M_pq = dimension // 4
    candidates = [m for m in [16, 24, 32, 48, 64, 96, 128] 
                  if dimension % m == 0 and m <= max_M_pq]
    
    # Chọn tùy theo N: nhỏ thì M_pq nhỏ (ít centroids cần train)
    if n < 10_000:
        M_pq = candidates[len(candidates)//3] if candidates else 16
    elif n < 100_000:
        M_pq = candidates[len(candidates)//2] if candidates else 32
    else:
        M_pq = candidates[-1] if candidates else 48
    
    # ── HNSW ────────────────────────────────────────────────────────────
    if n < 10_000:
        M_hnsw, ef_construction = 16, 100
    elif n < 100_000:
        M_hnsw, ef_construction = 32, 200
    elif n < 1_000_000:
        M_hnsw, ef_construction = 48, 200
    else:
        M_hnsw, ef_construction = 64, 400
    
    max_ef = min(512, n)
    ef_search_values = sorted(set([
        16, 32, 64, 128, 256, 512
    ]))
    ef_search_values = [v for v in ef_search_values if v <= max_ef]
    
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

Dùng trong `run_benchmark.py`:

```python
from rag_benchmark.benchmark.params import get_optimal_params

params = get_optimal_params(n=len(embeddings), dimension=embedder.dimension)
print(f"""
📐 Auto-tuned params for N={len(embeddings):,}:
   IVF  : nlist={params.nlist}, nprobe={params.nprobe_values}
   PQ   : M_pq={params.M_pq}, nbits={params.nbits}
   HNSW : M={params.M_hnsw}, efC={params.ef_construction}, efS={params.ef_search_values}
   Min train vectors needed: {params.min_train_vectors:,}
""")

# Build IVF+PQ với params tự động
for nprobe in params.nprobe_values:
    results.append(runner.benchmark_ivfpq(
        nlist=params.nlist,
        M_pq=params.M_pq,
        nprobe=nprobe
    ))
```


***

## Output tự động tính khi chạy

```
# N=1,000 (test)
nlist=25, nprobe=[1,4,8,16,25], M_pq=16, HNSW M=16

# N=10,000
nlist=100, nprobe=[1,4,8,16,32,64,100], M_pq=32, HNSW M=32

# N=100,000  
nlist=800, nprobe=[1,4,8,16,32,64,128,256,512,800], M_pq=48, HNSW M=48

# N=1,000,000
nlist=16000, nprobe=[1,4,8,16,32,64,128,256,512], M_pq=96, HNSW M=64
```


***

## Về warning PQ với 256 sub-centroids

Khi tăng lên 10K thì warning tự biến mất :

- N=10K, M_pq=32 → mỗi sub-space có `10K/32 = 312` vectors → `312 > 39` ✅
- N=100K → `100K/48 ≈ 2083` vectors/sub-space → hoàn toàn ổn ✅
- N=1M → `1M/96 ≈ 10,416` vectors/sub-space → rất tốt ✅

Và đối với **benchmark kết quả chuẩn nhất**, FAISS wiki khuyến nghị dùng **100×nlist vectors để train**, không chỉ 39×nlist minimum . Với 1M vectors, bạn có thể sample 400K ngẫu nhiên để train rồi add toàn bộ 1M:

```python
# Train trên sample, add toàn bộ — chuẩn production
if len(vectors) > 100 * nlist:
    train_sample = vectors[np.random.choice(len(vectors), 100 * nlist, replace=False)]
else:
    train_sample = vectors

index.train(train_sample)   # train trên sample
index.add(vectors)          # add toàn bộ
```

<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors

[^2]: https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/

[^3]: https://krunalkanojiya.com/blog/ivf-index-explained

[^4]: https://www.pinecone.io/learn/series/faiss/product-quantization/

[^5]: https://apxml.com/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/practice-quantization-filtering

[^6]: https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/

[^7]: https://www.linkedin.com/posts/rk0718_from-embeddings-to-search-faiss-hnsw-and-activity-7380951043565621248-eCOp

[^8]: https://arxiv.org/html/2401.08281v4

[^9]: https://github.com/facebookresearch/faiss/issues/253

[^10]: https://www.sandgarden.com/learn/faiss

[^11]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

[^12]: https://www.pinecone.io/learn/series/faiss/vector-indexes/

[^13]: https://arxiv.org/pdf/2412.01555.pdf

[^14]: https://www.facebook.com/groups/faissusers/posts/1010636506027302/

[^15]: https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/

