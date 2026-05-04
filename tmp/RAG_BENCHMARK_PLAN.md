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

## 4. Cấu hình `pyproject.toml`

```toml
[project]
name = "rag-benchmark"
version = "0.1.0"
description = "RAG VectorDB Benchmark — MedRAG/Wikipedia (PTIT VectorDB-2026)"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]

dependencies = [
    "faiss-gpu-cu12>=1.8.0",
    "sentence-transformers>=3.0.0",
    "datasets>=3.0.0",
    "transformers>=4.40.0",
    "torch>=2.3.0",
    "openai>=1.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "python-dotenv>=1.0.0",
    "plotly>=5.22.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "kaleido>=0.2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ipykernel>=6.29.0",
    "jupyterlab>=4.2.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

[project.scripts]
download-dataset = "scripts.download_dataset:main"
build-index      = "scripts.build_index:main"
run-benchmark    = "scripts.run_benchmark:main"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "ipykernel>=6.29.0",
    "jupyterlab>=4.2.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_benchmark"]
```

---

## 5. Cấu hình `.env` cho LLM API

### `.env.example` (commit vào git)

```dotenv
# ==============================================
# LLM Configuration
# ==============================================
LLM_PROVIDER=openai                  # openai | ollama | vllm
OPENAI_API_KEY=sk-proj-...           # OpenAI API key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini             # hoặc gpt-4o

# Nếu dùng Ollama thay OpenAI
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ==============================================
# Embedding Configuration
# ==============================================
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda                # cuda | cpu
EMBEDDING_BATCH_SIZE=512

# ==============================================
# Reranker Configuration
# ==============================================
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cuda

# ==============================================
# Dataset Configuration
# ==============================================
DATASET_NAME=MedRAG/wikipedia
DATASET_SPLIT=train
DATASET_MAX_SAMPLES=10000            # 10000 | 100000 | 1000000
DATASET_CACHE_DIR=./data/raw

# ==============================================
# FAISS Configuration
# ==============================================
FAISS_USE_GPU=true
FAISS_GPU_ID=0                       # GPU A40 device ID
INDEX_SAVE_DIR=./indexes

# ==============================================
# Benchmark Configuration
# ==============================================
BENCHMARK_TOP_K=10                   # Recall@10
BENCHMARK_TOP_ANN=50                 # ANN lấy Top 50 trước
BENCHMARK_N_QUERIES=100              # Số query để benchmark
RESULTS_DIR=./results
```

### `.env` (KHÔNG commit — thêm vào `.gitignore`)

```dotenv
# Copy từ .env.example và điền key thật
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-ABC123...     # ← Điền key thật ở đây
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
...
```

### `src/rag_benchmark/config/settings.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class LLMSettings(BaseSettings):
    provider: Literal["openai", "ollama", "vllm"] = "openai"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


class EmbeddingSettings(BaseSettings):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    batch_size: int = 512

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EMBEDDING_", extra="ignore")


class DatasetSettings(BaseSettings):
    name: str = "MedRAG/wikipedia"
    split: str = "train"
    max_samples: int = 10_000
    cache_dir: str = "./data/raw"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATASET_", extra="ignore")


class FAISSSettings(BaseSettings):
    use_gpu: bool = True
    gpu_id: int = 0
    index_save_dir: str = "./indexes"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FAISS_", extra="ignore")


class BenchmarkSettings(BaseSettings):
    top_k: int = 10
    top_ann: int = 50
    n_queries: int = 100
    results_dir: str = "./results"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BENCHMARK_", extra="ignore")


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    dataset: DatasetSettings = DatasetSettings()
    faiss: FAISSSettings = FAISSSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Singleton — import từ mọi nơi
settings = Settings()
```

---

## 6. Dataset: MedRAG/Wikipedia

### 6.1 Về dataset

Dataset **`MedRAG/wikipedia`** trên HuggingFace là corpus Wikipedia được tiền xử lý cho bài toán RAG y tế:

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn** | Wikipedia (tiếng Anh) |
| **Số lượng** | ~6.4M passages (chunks) |
| **Kích thước** | ~22GB |
| **Fields** | `id`, `title`, `content`, `contents` |
| **Mục đích** | Retrieval corpus cho câu hỏi y tế (MIRAGE benchmark) |
| **Format** | Parquet, streaming hỗ trợ |

### 6.2 Cách load (streaming — phù hợp với corpus lớn)

```python
from datasets import load_dataset

# Streaming — không cần tải toàn bộ về máy trước
dataset = load_dataset(
    "MedRAG/wikipedia",
    split="train",
    streaming=True,      # Quan trọng với 6.4M rows
    trust_remote_code=True
)

# Xem một sample
for row in dataset.take(3):
    print(row.keys())       # dict_keys(['id', 'title', 'content', 'contents'])
    print(row['title'])
    print(row['content'][:200])
    break
```

### 6.3 Fields quan trọng

```python
# Mỗi row có dạng:
{
    "id": "6794574_0",           # Wikipedia article ID + chunk index
    "title": "Diabetes mellitus type 2",   # Tiêu đề bài Wikipedia
    "content": "Diabetes mellitus...",     # Nội dung đoạn văn (~200 tokens)
    "contents": "Diabetes mellitus type 2\nDiabetes mellitus..."  # title + content
}
```

**Lưu ý thiết kế:**
- Dùng field `contents` (= title + "\n" + content) để embed — bao gồm ngữ cảnh từ tiêu đề
- Field `id` theo format `{wiki_id}_{chunk_index}` — dùng để truy vết nguồn
- Chunk đã được chia sẵn (≈100-200 tokens/chunk) — không cần chunking thêm

### 6.4 Metadata filtering strategy

```python
# Metadata cho mỗi vector:
metadata = {
    "id": row["id"],
    "title": row["title"],
    "source": "wikipedia",
    # Có thể extract thêm từ title:
    # "topic": extract_medical_topic(row["title"])  
}

# Filter examples (slide 17 — Metadata filtering):
# doc_type = "wikipedia"    → lọc theo nguồn
# title contains "diabetes" → lọc theo chủ đề y tế
```

---

## 7. Pipeline 5 bước

Theo slide 31 VectorDB-2026 — **Pipeline gợi ý**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline (5 bước)                         │
│                                                                   │
│  [1] Query       [2] ANN          [3] Filter    [4] Rerank       │
│  Rewrite    →   Search (50) →   Metadata  →  Cross-encoder  →  │
│                                                                   │
│  [5] Top-k → LLM (GPT-4o / Ollama)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Bước 1: Query Rewrite

**Mục đích:** Mở rộng và làm rõ query, tăng khả năng tìm kiếm semantic.

**Input:** Query gốc từ người dùng  
**Output:** Query đã được rewrite/expand  
**Kỹ thuật:**
- Dùng LLM để paraphrase + bổ sung từ khóa y tế
- Prompt template: `"Rewrite the following medical question to be more specific and add relevant medical keywords: {query}"`
- Fallback: keyword expansion rule-based nếu không có LLM

**File:** `src/rag_benchmark/modules/query_rewriter.py`

---

### Bước 2: ANN Search — Top 50

**Mục đích:** Tìm nhanh 50 candidates từ corpus (O(log n) với HNSW).

**Input:** Query embedding vector  
**Output:** Top-50 document IDs + distances  
**Index options:** Flat (baseline) | HNSW | IVF | IVF+PQ  
**Tool:** FAISS GPU (A40 — 48GB VRAM, xử lý 1M+ vectors)

**Key code (slide 33):**
```python
import faiss

# Chuyển index lên GPU A40
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # GPU ID = 0

# Search
distances, labels = gpu_index.search(query_embedding, k=50)
```

**File:** `src/rag_benchmark/modules/ann_search.py`

---

### Bước 3: Filter Metadata

**Mục đích:** Loại bỏ kết quả không phù hợp, giảm noise retrieval.

**Input:** Top-50 IDs từ bước 2 + điều kiện lọc  
**Output:** Filtered list (thường còn 20-50 documents)  
**Logic (slide 17):**
```python
# Ví dụ filter conditions:
filtered = [
    doc for doc in candidates
    if doc.metadata["source"] == "wikipedia"
    and query_topic in doc.metadata["title"].lower()
]
```

**File:** `src/rag_benchmark/modules/metadata_filter.py`

---

### Bước 4: Rerank — Cross-Encoder

**Mục đích:** Sắp xếp lại kết quả chính xác hơn Bi-encoder (slide 29).

**Input:** Filtered candidates (~20-50 docs) + original query  
**Output:** Top-5 documents đã rerank  
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Lý do chỉ rerank ~50:** Cross-encoder tốn O(N×L) time → chỉ áp dụng trên tập nhỏ

**Key code:**
```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
pairs = [(query, doc.content) for doc in candidates]
scores = cross_encoder.predict(pairs)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
top5 = [doc for doc, _ in ranked[:5]]
```

**File:** `src/rag_benchmark/modules/reranker.py`

---

### Bước 5: Top-k → LLM

**Mục đích:** Tổng hợp context và sinh câu trả lời.

**Input:** Top-5 documents + original query  
**Output:** Câu trả lời ngôn ngữ tự nhiên  
**LLM:** OpenAI GPT-4o-mini (qua `.env`) / Ollama / vLLM  
**Prompt template:**
```
System: You are a medical information assistant. Answer based only on the provided context.

Context:
{context_1}
---
{context_2}
---
...

Question: {original_query}
Answer:
```

**File:** `src/rag_benchmark/modules/llm_generator.py`

---

## 8. Thiết kế module & adapter

### 8.1 Adapter Pattern — LLM

```python
# src/rag_benchmark/adapters/llm/base.py

from abc import ABC, abstractmethod
from ..models.result import GenerationResult


class LLMAdapter(ABC):
    """Abstract base class cho mọi LLM provider."""

    @abstractmethod
    async def generate(self, prompt: str, context: list[str]) -> GenerationResult:
        """Sinh câu trả lời từ query + context."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Kiểm tra LLM provider có hoạt động không."""
        ...
```

```python
# src/rag_benchmark/adapters/llm/openai_adapter.py

from openai import AsyncOpenAI
from .base import LLMAdapter
from ...config.settings import settings
from ...models.result import GenerationResult


class OpenAIAdapter(LLMAdapter):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm.openai_api_key,
            base_url=settings.llm.openai_base_url,
        )
        self.model = settings.llm.openai_model

    async def generate(self, query: str, context: list[str]) -> GenerationResult:
        context_text = "\n---\n".join(context)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical information assistant."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ],
            temperature=0.1,
        )
        return GenerationResult(
            answer=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens,
        )

    def health_check(self) -> bool:
        try:
            import asyncio
            asyncio.run(self.client.models.list())
            return True
        except Exception:
            return False
```

### 8.2 Adapter Pattern — Embedding

```python
# src/rag_benchmark/adapters/embedding/base.py

from abc import ABC, abstractmethod
import numpy as np


class EmbeddingAdapter(ABC):
    """Abstract base class cho embedding models."""

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Chuyển texts thành vectors, normalize=True (cosine search)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Trả về số chiều của embedding."""
        ...
```

```python
# src/rag_benchmark/adapters/embedding/sentence_transformer_adapter.py

import numpy as np
from sentence_transformers import SentenceTransformer
from .base import EmbeddingAdapter
from ...config.settings import settings


class SentenceTransformerAdapter(EmbeddingAdapter):
    def __init__(self):
        self.model = SentenceTransformer(
            settings.embedding.model,
            device=settings.embedding.device
        )
        self._dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = None) -> np.ndarray:
        bs = batch_size or settings.embedding.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,   # L2 norm → cosine similarity = inner product
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    @property
    def dimension(self) -> int:
        return self._dimension
```

### 8.3 FAISS Index Builders

```python
# src/rag_benchmark/indexing/hnsw_index.py

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class HNSWIndexBuilder:
    """
    HNSW Index — Graph-based ANN (slide 9: "QUAN TRỌNG NHẤT").
    
    Tham số tuning (slide 24):
    - M: số neighbor mỗi node (Cao → accuracy↑, memory↑)
    - efSearch: số candidate (Cao → recall↑, latency↑)
    """

    def __init__(self, dimension: int, M: int = 32, ef_construction: int = 200):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self._cpu_index = None
        self._gpu_index = None

    def build(self, vectors: np.ndarray) -> "HNSWIndexBuilder":
        """Build HNSW index từ numpy array."""
        self._cpu_index = faiss.IndexHNSWFlat(self.dimension, self.M)
        self._cpu_index.hnsw.efConstruction = self.ef_construction
        self._cpu_index.add(vectors)

        # Chuyển lên GPU A40
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(res, settings.faiss.gpu_id, self._cpu_index)

        return self

    def search(self, query: np.ndarray, k: int = 50, ef_search: int = 128):
        """Tìm k nearest neighbors."""
        if self._gpu_index:
            self._cpu_index.hnsw.efSearch = ef_search  # Tune at query time
            return self._gpu_index.search(query, k)
        return self._cpu_index.search(query, k)

    def save(self, path: Path):
        faiss.write_index(self._cpu_index, str(path))

    @classmethod
    def load(cls, path: Path, dimension: int) -> "HNSWIndexBuilder":
        builder = cls(dimension)
        builder._cpu_index = faiss.read_index(str(path))
        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            builder._gpu_index = faiss.index_cpu_to_gpu(
                res, settings.faiss.gpu_id, builder._cpu_index
            )
        return builder
```

```python
# src/rag_benchmark/indexing/ivfpq_index.py

import faiss
import numpy as np
from pathlib import Path
from ..config.settings import settings


class IVFPQIndexBuilder:
    """
    IVF+PQ Index — Memory-constrained deployment (slide 11-12).
    
    PQ nén: 768-dim × 4 bytes = 3,072 bytes → ~96 bytes (32× nhỏ hơn)
    
    Tham số tuning (slide 25):
    - nlist: số cluster (Nhiều cluster = finer partitioning)
    - nprobe: số cluster query (nprobe↑ → accuracy↑, latency↑)
    - M_pq: số sub-quantizers (D phải chia hết cho M_pq)
    - nbits: bits mỗi sub-quantizer (thường 8)
    """

    def __init__(self, dimension: int, nlist: int = 100, M_pq: int = 48, nbits: int = 8):
        self.dimension = dimension
        self.nlist = nlist
        self.M_pq = M_pq  # dimension (384) phải chia hết cho M_pq
        self.nbits = nbits
        self._cpu_index = None
        self._gpu_index = None

    def build(self, vectors: np.ndarray, nprobe: int = 10) -> "IVFPQIndexBuilder":
        """Build IVF+PQ index. Cần train trước khi add."""
        assert len(vectors) >= self.nlist * 39, \
            f"Cần ít nhất {self.nlist * 39} vectors để train {self.nlist} clusters"

        quantizer = faiss.IndexFlatIP(self.dimension)
        self._cpu_index = faiss.IndexIVFPQ(
            quantizer, self.dimension, self.nlist, self.M_pq, self.nbits
        )
        self._cpu_index.train(vectors)
        self._cpu_index.add(vectors)
        self._cpu_index.nprobe = nprobe

        if settings.faiss.use_gpu:
            res = faiss.StandardGpuResources()
            self._gpu_index = faiss.index_cpu_to_gpu(res, settings.faiss.gpu_id, self._cpu_index)

        return self

    def search(self, query: np.ndarray, k: int = 50, nprobe: int = None):
        if nprobe:
            self._cpu_index.nprobe = nprobe
        if self._gpu_index:
            return self._gpu_index.search(query, k)
        return self._cpu_index.search(query, k)

    def save(self, path: Path):
        faiss.write_index(self._cpu_index, str(path))
```

### 8.4 Data Models (Pydantic)

```python
# src/rag_benchmark/models/document.py

from pydantic import BaseModel, Field
from typing import Optional
import numpy as np


class DocumentMetadata(BaseModel):
    id: str
    title: str
    source: str = "wikipedia"
    topic: Optional[str] = None


class Document(BaseModel):
    id: str
    content: str
    metadata: DocumentMetadata

    class Config:
        arbitrary_types_allowed = True


class Chunk(BaseModel):
    """Chunk đã được embed, sẵn sàng đưa vào FAISS index."""
    doc_id: str
    chunk_index: int
    text: str
    metadata: DocumentMetadata
    # vector: np.ndarray  # Không lưu trong Pydantic model, lưu riêng
```

```python
# src/rag_benchmark/models/result.py

from pydantic import BaseModel
from typing import Optional
from .document import DocumentMetadata


class SearchResult(BaseModel):
    doc_id: str
    rank: int
    score: float
    metadata: DocumentMetadata
    text: str


class BenchmarkResult(BaseModel):
    """Kết quả benchmark một experiment."""
    index_type: str              # "Flat" | "HNSW" | "IVF" | "IVF+PQ"
    n_vectors: int               # 10K | 100K | 1M
    latency_ms: float            # ms/query (trung bình)
    latency_p95_ms: float        # P95 latency
    recall_at_k: float           # Recall@K
    k: int                       # K value
    params: dict                 # {M, efSearch} hoặc {nlist, nprobe}
    n_queries: int               # Số query benchmark


class GenerationResult(BaseModel):
    answer: str
    model: str
    tokens_used: int
    latency_ms: Optional[float] = None
```

---

## 9. Benchmark Plan: Flat vs HNSW vs IVF vs IVF+PQ

### 9.1 Experiment Matrix

**Theo slide 34 — "Viết báo cáo":**

#### Group A: Flat vs HNSW

| Experiment | Index | N | M | efSearch | Metric |
|---|---|---|---|---|---|
| A-0 (baseline) | Flat | 10K/100K/1M | — | — | Latency, Recall=1.0 |
| A-1 | HNSW | 10K | 8 | 16,32,64,128,256 | Latency, Recall@10 |
| A-2 | HNSW | 10K | 16 | 16,32,64,128,256 | Latency, Recall@10 |
| A-3 | HNSW | 10K | 32 | 16,32,64,128,256 | Latency, Recall@10 |
| A-4 | HNSW | 10K | 64 | 16,32,64,128,256 | Latency, Recall@10 |
| A-5..A-8 | HNSW | 100K | 8,16,32,64 | như trên | Latency, Recall@10 |
| A-9..A-12 | HNSW | 1M | 8,16,32,64 | như trên | Latency, Recall@10 |

#### Group B: IVF vs IVF+PQ

| Experiment | Index | N | nlist | nprobe | Metric |
|---|---|---|---|---|---|
| B-0 | IVF | 10K | 100 | 1,4,8,16,32,64,100 | Latency, Recall@10 |
| B-1 | IVF | 100K | 316 | 1,4,8,16,32,64,100 | Latency, Recall@10 |
| B-2 | IVF | 1M | 1000 | 1,4,8,16,32,64,100 | Latency, Recall@10 |
| B-3 | IVF+PQ | 10K | 100 | 1,4,8,16,32,64 | Latency, Recall@10 |
| B-4 | IVF+PQ | 100K | 316 | 1,4,8,16,32,64 | Latency, Recall@10 |
| B-5 | IVF+PQ | 1M | 1000 | 1,4,8,16,32,64 | Latency, Recall@10 |

**nlist rule of thumb:** `nlist ≈ sqrt(N)` — 10K→100, 100K→316, 1M→1000

### 9.2 Metrics Definition

```
Recall@K = |Retrieved_K ∩ Ground_Truth_K| / |Ground_Truth_K|

Trong đó:
- Ground_Truth_K = Top-K từ Flat (brute-force, accuracy=100%)
- Retrieved_K = Top-K từ ANN index đang test

Latency = thời gian search 1 query (trung bình trên N_QUERIES=100 queries)
         = (total_time / N_QUERIES) × 1000  [ms]
```

### 9.3 Expected Results (từ lý thuyết slide VectorDB-2026)

**Flat vs HNSW:**
- Flat: ~45ms/query (10K), ~450ms (100K), ~4500ms (1M) — O(N)
- HNSW M=32, ef=128: ~2ms/query tất cả scales — O(log N)
- HNSW Recall@10: ~0.95 với M=32, ef=128

**IVF vs IVF+PQ:**
- IVF nprobe=64: ~5ms, Recall@10 ~0.94
- IVF+PQ nprobe=64: ~1.9ms, Recall@10 ~0.78 (đánh đổi accuracy cho memory)
- Memory: IVF+PQ nhỏ hơn IVF thuần ~32× (slide 11)

---

## 10. Code mẫu từng module

### 10.1 Dataset Loader

```python
# src/rag_benchmark/data/dataset_loader.py

from datasets import load_dataset, IterableDataset
from ..config.settings import settings
from ..models.document import Document, DocumentMetadata
from tqdm import tqdm
import numpy as np


def load_medrag_wikipedia(max_samples: int = None) -> tuple[list[Document], list[str]]:
    """
    Load MedRAG/Wikipedia dataset.
    
    Returns:
        documents: List of Document objects
        texts: List of text strings (for embedding)
    """
    n = max_samples or settings.dataset.max_samples
    print(f"Loading {n:,} documents from {settings.dataset.name}...")

    dataset: IterableDataset = load_dataset(
        settings.dataset.name,
        split=settings.dataset.split,
        streaming=True,
        trust_remote_code=True,
        cache_dir=settings.dataset.cache_dir,
    )

    documents = []
    texts = []

    for i, row in enumerate(tqdm(dataset, total=n, desc="Loading")):
        if i >= n:
            break

        # Dùng 'contents' = title + "\n" + content để có context đầy đủ
        text = row.get("contents", f"{row['title']}\n{row['content']}")

        doc = Document(
            id=row["id"],
            content=text[:512],  # Truncate để giảm memory
            metadata=DocumentMetadata(
                id=row["id"],
                title=row["title"],
                source="wikipedia",
            )
        )
        documents.append(doc)
        texts.append(doc.content)

    print(f"Loaded {len(documents):,} documents")
    return documents, texts
```

### 10.2 Benchmark Runner

```python
# src/rag_benchmark/benchmark/runner.py

import time
import numpy as np
from pathlib import Path
from ..models.result import BenchmarkResult
from ..indexing.flat_index import FlatIndexBuilder
from ..indexing.hnsw_index import HNSWIndexBuilder
from ..indexing.ivf_index import IVFIndexBuilder
from ..indexing.ivfpq_index import IVFPQIndexBuilder
from .metrics import compute_recall_at_k


class BenchmarkRunner:
    """
    Chạy benchmark experiments theo plan ở Section 9.
    So sánh Flat vs HNSW | IVF vs IVF+PQ.
    """

    def __init__(self, embeddings: np.ndarray, query_embeddings: np.ndarray, k: int = 10):
        self.embeddings = embeddings
        self.query_embeddings = query_embeddings
        self.k = k
        self.n_vectors = len(embeddings)
        self.n_queries = len(query_embeddings)
        self.dimension = embeddings.shape[1]

        # Ground truth từ Flat search
        print("Computing ground truth (Flat/brute-force)...")
        self._ground_truth = self._compute_ground_truth()

    def _compute_ground_truth(self) -> np.ndarray:
        """Flat search = 100% accurate ground truth."""
        flat = FlatIndexBuilder(self.dimension).build(self.embeddings)
        _, labels = flat.search(self.query_embeddings, self.k)
        return labels  # shape: (n_queries, k)

    def benchmark_flat(self) -> BenchmarkResult:
        """Baseline benchmark với Flat index."""
        flat = FlatIndexBuilder(self.dimension).build(self.embeddings)

        times = []
        for q in self.query_embeddings:
            t0 = time.perf_counter()
            flat.search(q.reshape(1, -1), self.k)
            times.append((time.perf_counter() - t0) * 1000)

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

    def benchmark_hnsw(self, M: int = 32, ef_search: int = 128) -> BenchmarkResult:
        """Benchmark HNSW với specific M và efSearch."""
        index = HNSWIndexBuilder(self.dimension, M=M).build(self.embeddings)

        times = []
        all_labels = []

        for q in self.query_embeddings:
            t0 = time.perf_counter()
            _, labels = index.search(q.reshape(1, -1), self.k, ef_search=ef_search)
            times.append((time.perf_counter() - t0) * 1000)
            all_labels.append(labels[0])

        retrieved = np.array(all_labels)
        recall = compute_recall_at_k(retrieved, self._ground_truth, self.k)

        return BenchmarkResult(
            index_type="HNSW",
            n_vectors=self.n_vectors,
            latency_ms=float(np.mean(times)),
            latency_p95_ms=float(np.percentile(times, 95)),
            recall_at_k=recall,
            k=self.k,
            params={"M": M, "efSearch": ef_search},
            n_queries=self.n_queries,
        )

    def benchmark_ivf(self, nlist: int = 100, nprobe: int = 10) -> BenchmarkResult:
        """Benchmark IVF với specific nlist và nprobe."""
        index = IVFIndexBuilder(self.dimension, nlist=nlist).build(self.embeddings, nprobe=nprobe)
        # ... (similar pattern)

    def benchmark_ivfpq(self, nlist: int = 100, M_pq: int = 48, nprobe: int = 10) -> BenchmarkResult:
        """Benchmark IVF+PQ với specific params."""
        index = IVFPQIndexBuilder(self.dimension, nlist=nlist, M_pq=M_pq).build(
            self.embeddings, nprobe=nprobe
        )
        # ... (similar pattern)
```

### 10.3 Metrics

```python
# src/rag_benchmark/benchmark/metrics.py

import numpy as np


def compute_recall_at_k(
    retrieved: np.ndarray,   # shape: (n_queries, k)
    ground_truth: np.ndarray,  # shape: (n_queries, k)
    k: int
) -> float:
    """
    Recall@K = trung bình tỷ lệ kết quả đúng trong Top-K.
    
    Ground truth = kết quả từ Flat (brute-force, 100% accurate).
    """
    recalls = []
    for i in range(len(retrieved)):
        gt_set = set(ground_truth[i][:k].tolist())
        retrieved_set = set(retrieved[i][:k].tolist())
        recalls.append(len(retrieved_set & gt_set) / len(gt_set))
    return float(np.mean(recalls))


def compute_latency_stats(times_ms: list[float]) -> dict:
    """Tính các thống kê latency."""
    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
```

### 10.4 Reporter — Xuất biểu đồ

```python
# src/rag_benchmark/benchmark/reporter.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


class BenchmarkReporter:
    """Xuất báo cáo benchmark theo yêu cầu slide 34."""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "csv").mkdir(exist_ok=True)
        (self.results_dir / "charts").mkdir(exist_ok=True)

    def save_csv(self, df: pd.DataFrame, name: str):
        path = self.results_dir / "csv" / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    def plot_latency_vs_recall(self, df: pd.DataFrame, title: str, filename: str):
        """
        Plot latency vs recall (scatter) — yêu cầu từ slide 34.
        Higher-left corner = better (low latency + high recall).
        """
        fig = go.Figure()

        colors = {"Flat": "gray", "HNSW": "blue", "IVF": "green", "IVF+PQ": "orange"}

        for index_type in df["index_type"].unique():
            sub = df[df["index_type"] == index_type].sort_values("recall_at_k")
            fig.add_trace(go.Scatter(
                x=sub["latency_ms"],
                y=sub["recall_at_k"],
                mode="lines+markers",
                name=index_type,
                marker=dict(size=8, color=colors.get(index_type, "purple")),
                hovertemplate=(
                    f"<b>{index_type}</b><br>"
                    "Latency: %{x:.2f}ms<br>"
                    "Recall@10: %{y:.3f}<br>"
                    "%{text}"
                ),
                text=sub.get("params_str", [""] * len(sub)),
            ))

        fig.update_layout(
            title={
                "text": f"{title}<br>"
                        "<span style='font-size:14px;font-weight:normal'>"
                        "Source: MedRAG/Wikipedia | Upper-left = better</span>"
            },
            xaxis_title="Latency (ms/query)",
            yaxis_title="Recall@10",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        )

        path = self.results_dir / "charts" / f"{filename}.png"
        fig.write_image(str(path))
        fig.write_html(str(path).replace(".png", ".html"))
        print(f"Saved chart: {path}")
        return fig
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

### 11.2 `scripts/run_benchmark.py`

```python
#!/usr/bin/env python3
"""
Main benchmark script.
Usage: uv run python scripts/run_benchmark.py --n-samples 10000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_benchmark.config.settings import settings
from rag_benchmark.data.dataset_loader import load_medrag_wikipedia
from rag_benchmark.adapters.embedding.sentence_transformer_adapter import SentenceTransformerAdapter
from rag_benchmark.benchmark.runner import BenchmarkRunner
from rag_benchmark.benchmark.reporter import BenchmarkReporter
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    # 1. Load dataset
    documents, texts = load_medrag_wikipedia(max_samples=args.n_samples)

    # 2. Embedding
    embedder = SentenceTransformerAdapter()
    print(f"Embedding {len(texts):,} documents (dim={embedder.dimension})...")
    embeddings = embedder.encode(texts)

    # 3. Sample queries
    import numpy as np
    query_ids = np.random.choice(len(embeddings), args.n_queries, replace=False)
    query_embeddings = embeddings[query_ids]

    # 4. Run benchmark
    runner = BenchmarkRunner(embeddings, query_embeddings, k=args.k)
    results = []

    # Flat baseline
    results.append(runner.benchmark_flat())

    # HNSW sweep
    for M in [8, 16, 32, 64]:
        for ef_search in [16, 32, 64, 128, 256]:
            results.append(runner.benchmark_hnsw(M=M, ef_search=ef_search))

    # IVF sweep
    nlist = max(100, int(args.n_samples ** 0.5))
    for nprobe in [1, 4, 8, 16, 32, 64, nlist]:
        results.append(runner.benchmark_ivf(nlist=nlist, nprobe=nprobe))

    # IVF+PQ sweep
    for nprobe in [1, 4, 8, 16, 32, 64]:
        results.append(runner.benchmark_ivfpq(nlist=nlist, M_pq=48, nprobe=nprobe))

    # 5. Report
    if args.report:
        df = pd.DataFrame([r.model_dump() for r in results])
        reporter = BenchmarkReporter()
        reporter.save_csv(df, f"benchmark_N{args.n_samples}")

        # Plot Flat vs HNSW
        hnsw_df = df[df["index_type"].isin(["Flat", "HNSW"])]
        reporter.plot_latency_vs_recall(
            hnsw_df,
            title=f"Flat vs HNSW — Latency-Recall Tradeoff (N={args.n_samples:,})",
            filename=f"flat_vs_hnsw_N{args.n_samples}"
        )

        # Plot IVF vs IVF+PQ
        ivf_df = df[df["index_type"].isin(["Flat", "IVF", "IVF+PQ"])]
        reporter.plot_latency_vs_recall(
            ivf_df,
            title=f"IVF vs IVF+PQ — Memory-Accuracy Tradeoff (N={args.n_samples:,})",
            filename=f"ivf_vs_ivfpq_N{args.n_samples}"
        )

    print("Benchmark complete!")


if __name__ == "__main__":
    main()
```

---

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

## Appendix: .gitignore

```gitignore
# Virtual environment
.venv/

# Environment variables (KHÔNG commit key!)
.env

# Data directories (quá lớn)
data/raw/
data/processed/
indexes/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# Results (có thể commit hoặc không)
# results/   ← uncomment nếu không muốn commit results

# OS
.DS_Store
Thumbs.db
```

---

*Tài liệu được tổng hợp từ: VectorDB-2026.pdf (PTIT) · pipeline.md · report-form-2.md · faiss-3.md · llm-4.md · HuggingFace MedRAG/Wikipedia*
