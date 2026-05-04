<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Loaded 1,000 documents from cache

Batches:   0%|          | 0/2 [00:00<?, ?it/s]
Batches:  50%|█████     | 1/2 [00:01<00:01,  1.50s/it]
Batches: 100%|██████████| 2/2 [00:01<00:00,  1.26it/s]
Batches: 100%|██████████| 2/2 [00:01<00:00,  1.11it/s]
Traceback (most recent call last):
  File "/workspace/giapcn/medrag/scripts/build_index.py", line 52, in <module>
    main()
  File "/workspace/giapcn/medrag/scripts/build_index.py", line 40, in main
    index = IVFIndexBuilder(embedder.dimension, nlist=nlist).build(embeddings)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/src/rag_benchmark/indexing/ivf_index.py", line 17, in build
    assert len(vectors) >= self.nlist * 39, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Need at least 3900 vectors to train 100 clusters

Lỗi đơn giản: với N=1000 vectors, `nlist=100` clusters yêu cầu tối thiểu `100 × 39 = 3900` vectors để train K-means.

## Fix ngay — 2 chỗ cần sửa

### 1. `build_index.py` — tự động tính `nlist` theo N

```python
# scripts/build_index.py

def main():
    # ...load documents, embed...
    
    n_vectors = len(embeddings)
    
    # Rule of thumb: nlist ≈ sqrt(N), nhưng tối thiểu phải đảm bảo
    # N >= nlist * 39  →  nlist <= N / 39
    # Công thức an toàn:
    nlist = min(
        max(4, int(n_vectors ** 0.5)),   # sqrt(N), ít nhất 4
        n_vectors // 39                   # đảm bảo đủ vectors để train
    )
    
    print(f"N={n_vectors}, nlist={nlist} (sqrt={int(n_vectors**0.5)}, max_safe={n_vectors//39})")

    # Build indexes
    if args.index in ("all", "hnsw"):
        index = HNSWIndexBuilder(embedder.dimension, M=32).build(embeddings)
        index.save(Path(settings.faiss.index_save_dir) / "hnsw.index")
        print("✅ HNSW saved")

    if args.index in ("all", "ivf"):
        index = IVFIndexBuilder(embedder.dimension, nlist=nlist).build(embeddings)
        index.save(Path(settings.faiss.index_save_dir) / "ivf.index")
        print("✅ IVF saved")

    if args.index in ("all", "ivfpq"):
        index = IVFPQIndexBuilder(embedder.dimension, nlist=nlist, M_pq=48).build(embeddings)
        index.save(Path(settings.faiss.index_save_dir) / "ivfpq.index")
        print("✅ IVF+PQ saved")
```


### 2. `ivf_index.py` và `ivfpq_index.py` — bỏ assert cứng, tự tính nlist an toàn

```python
# src/rag_benchmark/indexing/ivf_index.py  (và ivfpq_index.py tương tự)

def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFIndexBuilder":
    n = len(vectors)
    
    # Tự điều chỉnh nlist nếu không đủ vectors
    safe_nlist = min(self.nlist, n // 39)
    if safe_nlist < self.nlist:
        print(
            f"⚠️  nlist tự động giảm từ {self.nlist} → {safe_nlist} "
            f"(N={n} cần ít nhất {self.nlist * 39} để train {self.nlist} clusters)"
        )
        self.nlist = safe_nlist

    if self.nlist < 1:
        raise ValueError(f"N={n} quá nhỏ để dùng IVF. Cần ít nhất 39 vectors.")

    cpu_quantizer = faiss.IndexFlatIP(self.dimension)
    self._cpu_index = faiss.IndexIVFFlat(
        cpu_quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
    )
    self._cpu_index.train(vectors)
    self._cpu_index.add(vectors)
    self._cpu_index.nprobe = min(nprobe, self.nlist)  # nprobe không được > nlist
    # ...phần GPU giữ nguyên
```


***

## Quy tắc nlist theo N (bảng tham khảo)

| N vectors | nlist an toàn | nlist tối ưu (sqrt) | Ghi chú |
| :-- | :-- | :-- | :-- |
| **1K** | ≤ 25 | ~32 | Dùng 25 |
| **10K** | ≤ 256 | ~100 | Dùng 100 |
| **100K** | ≤ 2564 | ~316 | Dùng 256–512 |
| **1M** | ≤ 25641 | ~1000 | Dùng 1000–4096 |

**Quy tắc chính xác:** `nlist ≤ N / 39` để K-means hội tụ (mỗi cluster cần tối thiểu 39 điểm training).  Và `nprobe ≤ nlist` luôn luôn.

