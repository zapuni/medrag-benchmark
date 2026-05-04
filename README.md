# RAG VectorDB Benchmark — MedRAG/Wikipedia

> **Nguồn lý thuyết:** VectorDB-2026 (PTIT) · **Dataset:** HuggingFace MedRAG/Wikipedia  
> **Mục tiêu:** So sánh Flat vs HNSW & IVF vs IVF+PQ theo Latency (ms) và Recall@k trên pipeline 5 bước

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu trúc thư mục dự án](#2-cấu-trúc-thư-mục-dự-án)
3. [Cài đặt môi trường với `uv`](#3-cài-đặt-môi-trường-với-uv)
4. [Cấu hình `pyproject.toml`](#4-cấu-hình-pyprojecttoml)
5. [Cấu hình `.env` cho LLM API](#5-cấu-hình-env-cho-llm-api)
6. [Dataset: MedRAG/Wikipedia](#6-dataset-medragwikipedia)
7. [Pipeline 5 bước](#7-pipeline-5-bước)
8. [Thiết kế module & adapter](#8-thiết-kế-module--adapter)
9. [Benchmark Plan: Flat vs HNSW vs IVF vs IVF+PQ](#9-benchmark-plan-flat-vs-hnsw-vs-ivf-vs-ivfpq)
10. [Code mẫu từng module](#10-code-mẫu-từng-module)
11. [Chạy benchmark & xuất báo cáo](#11-chạy-benchmark--xuất-báo-cáo)
12. [Câu hỏi thảo luận từ slide](#12-câu-hỏi-thảo-luận-từ-slide)

---

## 1. Tổng quan dự án

### Mục tiêu học tập (từ slide 34 — VectorDB-2026)

Theo yêu cầu báo cáo trong slide **"Viết báo cáo"** của PTIT:

| Tiêu chí | Chi tiết |
|---|---|
| **So sánh** | Flat vs HNSW · IVF vs IVF+PQ |
| **Metrics** | Latency (ms/query) · Recall@k |
| **Quy mô N** | 10K → 100K → 1M vectors |
| **Tuning** | `M`, `efSearch` (HNSW) · `nlist`, `nprobe` (IVF) |
| **Output** | Plot latency vs recall (scatter chart) |

### Pipeline RAG áp dụng (từ slide 31)

```
Query rewrite → ANN Search (top 50) → Filter metadata → Rerank → Top-k → LLM
     [1]              [2]                   [3]             [4]       [5]
```

### Môi trường kỹ thuật

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Package manager | `uv` | Nhanh hơn pip/poetry, lockfile chuẩn |
| GPU | NVIDIA A40 (48GB VRAM) | → dùng `faiss-gpu` |
| Embedding | `sentence-transformers` | All-MiniLM-L6-v2 (384-dim) hoặc BGE-large (1024-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Chạy nhanh trên GPU |
| LLM | OpenAI API (GPT-4o) qua `.env` | Có thể swap sang Ollama/vLLM |
| Dataset | `MedRAG/wikipedia` (HuggingFace) | Corpus y tế tiêu chuẩn cho RAG |

---

## 2. Cấu trúc thư mục dự án

```
rag-vectordb-benchmark/
│
├── pyproject.toml              ← Khai báo dependencies, scripts, metadata
├── uv.lock                     ← Lockfile tự động sinh bởi uv
├── .env                        ← Biến môi trường (KHÔNG commit vào git)
├── .env.example                ← Template .env để share
├── .gitignore
│
├── src/
│   └── rag_benchmark/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py     ← Pydantic Settings — load .env
│       │
│       ├── adapters/           ← Adapter pattern: swap implementations
│       │   ├── __init__.py
│       │   ├── base.py         ← Abstract base classes
│       │   ├── llm/
│       │   │   ├── __init__.py
│       │   │   ├── base.py     ← LLMAdapter ABC
│       │   │   ├── openai_adapter.py
│       │   │   └── ollama_adapter.py
│       │   ├── embedding/
│       │   │   ├── __init__.py
│       │   │   ├── base.py     ← EmbeddingAdapter ABC
│       │   │   └── sentence_transformer_adapter.py
│       │   └── reranker/
│       │       ├── __init__.py
│       │       ├── base.py     ← RerankerAdapter ABC
│       │       └── cross_encoder_adapter.py
│       │
│       ├── models/             ← Data models (Pydantic)
│       │   ├── __init__.py
│       │   ├── document.py     ← Document, Chunk, Metadata
│       │   ├── query.py        ← Query, RewrittenQuery
│       │   └── result.py       ← SearchResult, BenchmarkResult
│       │
│       ├── modules/            ← Core pipeline modules
│       │   ├── __init__.py
│       │   ├── query_rewriter.py   ← Bước 1: Query Rewrite
│       │   ├── ann_search.py       ← Bước 2: ANN Search (FAISS GPU)
│       │   ├── metadata_filter.py  ← Bước 3: Filter Metadata
│       │   ├── reranker.py         ← Bước 4: Rerank (Cross-encoder)
│       │   └── llm_generator.py    ← Bước 5: Top-k → LLM
│       │
│       ├── indexing/           ← FAISS index builders
│       │   ├── __init__.py
│       │   ├── flat_index.py
│       │   ├── hnsw_index.py
│       │   ├── ivf_index.py
│       │   └── ivfpq_index.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── rag_pipeline.py     ← Orchestrate 5 bước
│       │
│       ├── benchmark/
│       │   ├── __init__.py
│       │   ├── runner.py           ← Chạy benchmark experiments
│       │   ├── metrics.py          ← Tính Latency, Recall@k
│       │   └── reporter.py         ← Xuất CSV + charts
│       │
│       └── data/
│           ├── __init__.py
│           └── dataset_loader.py   ← Load MedRAG/Wikipedia
│
├── scripts/
│   ├── download_dataset.py         ← Script tải dataset
│   ├── build_index.py              ← Script build FAISS index
│   └── run_benchmark.py            ← Script chạy toàn bộ benchmark
│
├── notebooks/
│   └── explore_dataset.ipynb       ← EDA dataset
│
├── data/
│   ├── raw/                        ← Dataset gốc
│   └── processed/                  ← Embeddings đã tính sẵn
│
├── indexes/                        ← FAISS index files (.index)
│
├── results/
│   ├── csv/                        ← Kết quả benchmark dạng CSV
│   └── charts/                     ← Biểu đồ PNG/HTML
│
└── report/
    └── benchmark_report.md         ← Báo cáo cuối cùng
```

---

## 3. Cài đặt môi trường với `uv`

### 3.1 Cài `uv` (nếu chưa có)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Kiểm tra
uv --version
```

### 3.2 Khởi tạo dự án

```bash
# Tạo thư mục và init project
mkdir rag-vectordb-benchmark && cd rag-vectordb-benchmark
uv init --name rag-benchmark --python 3.11

# Tạo virtual environment
uv venv --python 3.11

# Kích hoạt (Linux/macOS)
source .venv/bin/activate
```

### 3.3 Cài dependencies

```bash
# Core dependencies
uv add faiss-gpu-cu12          # FAISS GPU (CUDA 12.x — A40 dùng CUDA 12)
uv add sentence-transformers   # Embedding models
uv add datasets                # HuggingFace datasets
uv add transformers torch      # Cho GPU inference
uv add openai                  # OpenAI API
uv add pydantic pydantic-settings  # Config & validation
uv add python-dotenv           # Load .env

# Visualization & reporting
uv add plotly pandas numpy kaleido

# Dev dependencies
uv add --dev pytest ipykernel jupyterlab ruff mypy

# Sync (install tất cả)
uv sync
```

### 3.4 Kiểm tra FAISS GPU

```bash
python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'GPU count: {faiss.get_num_gpus()}')"
# Expected: GPU count: 1 (A40)
```

---


## 11. Chạy benchmark & xuất báo cáo

### 11.1 Script tổng thể

```bash
# 1. Cài đặt môi trường
uv sync

# 2. Tải dataset
uv run python scripts/download_dataset.py --n-samples 10000

# 3. Build embeddings + indexes
uv run python scripts/build_index.py --n-samples 10000 --index all

# 4. Chạy benchmark
uv run python scripts/run_benchmark.py --n-samples 10000 --report

# 5. Scale lên 100K
uv run python scripts/run_benchmark.py --n-samples 100000 --report

# 6. Scale lên 1M (A40 có 48GB VRAM — đủ cho 1M × 384-dim float32 = ~1.5GB)
uv run python scripts/run_benchmark.py --n-samples 1000000 --report
```





## 12. Câu hỏi thảo luận từ slide

*(Slide 35 — VectorDB-2026, PTIT)*

### ❓ Khi nào dùng HNSW vs IVF?

| Tình huống | Lựa chọn | Lý do |
|---|---|---|
| Production, latency < 5ms | **HNSW** | O(log N), recall cao >95% |
| N > 10M vectors, RAM hạn chế | **IVF+PQ** | Memory-constrained, 32× nhỏ hơn |
| Large scale, RAM đủ | **HNSW** | Best accuracy/speed tradeoff |
| Research/baseline | **Flat** | 100% accuracy, dùng để tính recall |
| Streaming updates nhiều | **IVF** | HNSW rebuild tốn kém hơn |

### ❓ Có nên always dùng Hybrid Search?

**Không phải lúc nào cũng cần.** Hybrid Search (BM25 + vector) thêm complexity:
- **Dùng:** Query chứa keyword y tế đặc biệt (`ibuprofen`, `ICD-10`), domain-specific thuật ngữ
- **Không cần:** Query semantic thuần (`"explain diabetes in simple terms"`)
- **Trade-off:** Tốn 2× thời gian retrieval, cần tune trọng số α cho fusion

### ❓ Trade-off accuracy vs cost?

```
                High Accuracy
                     │
              Flat ──┼── (100%, chậm nhất, O(N))
                     │
              HNSW ──┼── (~95-99%, nhanh, RAM cao)
                     │
               IVF ──┼── (~90-95%, nhanh vừa, RAM vừa)
                     │
           IVF+PQ ───┼── (~75-85%, nhanh nhất, RAM thấp nhất)
                     │
                Low Accuracy / Low Cost
```

**Key insight từ slide 36:** *"Tuning quan trọng hơn chọn model"* — HNSW với efSearch=256 tốt hơn HNSW efSearch=16 nhiều hơn là việc chọn giữa HNSW và IVF với cùng tham số mặc định.

---