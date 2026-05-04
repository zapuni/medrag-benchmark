# 1. Cài đặt môi trường
uv sync

# 2. Tải dataset (streaming) va cache 10k mau ra CSV
mkdir -p logs data/processed
PYTHONUNBUFFERED=1 uv run python scripts/download_dataset.py \
	--n-samples 10000 \
	--output ./data/processed/medrag_wikipedia_10000.csv \
	2>&1 | tee logs/download_dataset_10000.log

# 3. Build embeddings + indexes tu cache
PYTHONUNBUFFERED=1 uv run python scripts/build_index.py \
	--n-samples 10000 \
	--index all \
	--cache-path ./data/processed/medrag_wikipedia_10000.csv \
	2>&1 | tee logs/build_index_10000.log

# 4. Chay benchmark tu cache (N=10000)
PYTHONUNBUFFERED=1 uv run python scripts/run_benchmark.py \
	--n-samples 10000 \
	--k 10 \
	--n-queries 100 \
	--cache-path ./data/processed/medrag_wikipedia_10000.csv \
	--report \
	2>&1 | tee logs/run_benchmark_10000.log

# 5. Scale len 100K (cache + benchmark)
```
PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/download_dataset.py \
	--n-samples 100000 \
	--output ./data/processed/medrag_wikipedia_100000.csv \
	2>&1 | tee logs/download_dataset_100000.log

PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/run_benchmark.py \
	--n-samples 100000 \
	--k 10 \
	--n-queries 100 \
	--cache-path ./data/processed/medrag_wikipedia_100000.csv \
	--report \
	2>&1 | tee logs/run_benchmark_100000.log
```
# 6. Scale len 1M (A40 co 48GB VRAM)
```
PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/download_dataset.py \
	--n-samples 1000000 \
	--output ./data/processed/medrag_wikipedia_1000000.csv \
	2>&1 | tee logs/download_dataset_1000000.log

PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/run_benchmark.py \
	--n-samples 1000000 \
	--k 10 \
	--n-queries 100 \
	--cache-path ./data/processed/medrag_wikipedia_1000000.csv \
	--report \
	2>&1 | tee logs/run_benchmark_1000000.log
```

# Full lệnh chạy từ 10K->1M:
```
mkdir -p logs data/processed
PYTHONUNBUFFERED=1 uv run python scripts/download_dataset.py \
	--n-samples 10000 \
	--output ./data/processed/medrag_wikipedia_10K.csv \
	2>&1 | tee logs/download_dataset_10K.log

PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/download_dataset.py \
	--n-samples 100000 \
	--output ./data/processed/medrag_wikipedia_100K.csv \
	2>&1 | tee logs/download_dataset_100K.log

PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/download_dataset.py \
	--n-samples 1000000 \
	--output ./data/processed/medrag_wikipedia_1M.csv \
	2>&1 | tee logs/download_dataset_1M.log

PYTHONUNBUFFERED=1 uv run python scripts/run_benchmark.py \
	--n-samples 10000 \
	--k 10 \
	--n-queries 100 \
	--cache-path ./data/processed/medrag_wikipedia_10K.csv \
	--report \
	2>&1 | tee logs/run_benchmark_10K.log

PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/run_benchmark.py \
	--n-samples 100000 \
	--k 10 \
	--n-queries 100 \
	--cache-path ./data/processed/medrag_wikipedia_100K.csv \
	--report \
	2>&1 | tee logs/run_benchmark_100K.log

PYTHONUNBUFFERED=1 timeout --signal=SIGINT 12h uv run python scripts/run_benchmark.py \
	--n-samples 1000000 \
	--k 10 \
	--n-queries 100 \
	--cache-path ./data/processed/medrag_wikipedia_1M.csv \
	--report \
	2>&1 | tee logs/run_benchmark_1M.log
```