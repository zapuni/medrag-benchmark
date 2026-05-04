[project]
name = "rag-benchmark"
version = "0.1.0"
description = "RAG VectorDB Benchmark — MedRAG/Wikipedia"
requires-python = ">=3.10"

dependencies = [
    # ─── Core PyTorch stack (align CUDA 12.8) ───────────────────────
    "torch==2.8.0",
    "torchvision==0.23.0",

    # ─── FAISS GPU (cu12 build — dynamically links CUDA Runtime) ────
    "faiss-gpu-cu12>=1.8.0",

    # ─── Embedding & Reranker ────────────────────────────────────────
    "sentence-transformers>=3.0.0",
    "transformers==4.57.1",
    "accelerate==1.6.0",

    # ─── Dataset ─────────────────────────────────────────────────────
    "datasets>=3.0.0",
    "huggingface-hub>=0.36.0",

    # ─── LLM / API ───────────────────────────────────────────────────
    "openai>=2.0.0",

    # ─── Config & Validation ─────────────────────────────────────────
    "pydantic>=2.0.0",
    "pydantic-settings>=2.3.0",
    "python-dotenv>=1.0.0",

    # ─── Reporting ───────────────────────────────────────────────────
    "plotly>=5.22.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "kaleido>=0.2.1",
    "tqdm>=4.67.0",
]

[dependency-groups]        # ← dùng đây thay tool.uv.dev-dependencies (deprecated)
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "ipykernel>=6.29.0",
    "jupyterlab>=4.2.0",
]

[tool.uv]
package = false            # không build thành package installable

# ─── PyTorch index cho CUDA 12.8 (cu128) ────────────────────────────
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch     = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
# faiss-gpu-cu12 lấy từ PyPI mặc định — KHÔNG cần custom index

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"