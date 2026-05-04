#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

from rag_benchmark.adapters.embedding.sentence_transformer_adapter import SentenceTransformerAdapter
from rag_benchmark.benchmark.params import get_optimal_params
from rag_benchmark.benchmark.reporter import BenchmarkReporter
from rag_benchmark.benchmark.runner import BenchmarkRunner
from rag_benchmark.benchmark.multi_gpu_runner import MultiGPUBenchmarkRunner
from rag_benchmark.config.settings import settings
from rag_benchmark.data.dataset_loader import load_medrag_wikipedia


def _params_to_str(params: dict) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=settings.dataset.max_samples)
    parser.add_argument("--k", type=int, default=settings.benchmark.top_k)
    parser.add_argument("--n-queries", type=int, default=settings.benchmark.n_queries)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--cache-path", type=str, default=None)
    args = parser.parse_args()

    documents, texts = load_medrag_wikipedia(max_samples=args.n_samples, cache_path=args.cache_path)
    embedder = SentenceTransformerAdapter()
    embeddings = embedder.encode(texts)

    rng = np.random.default_rng(seed=42)
    query_ids = rng.choice(len(embeddings), args.n_queries, replace=False)
    query_embeddings = embeddings[query_ids]

    results = []

    params = get_optimal_params(n=len(embeddings), dimension=embedder.dimension)
    print(
        "Auto params: "
        f"nlist={params.nlist}, nprobe={params.nprobe_values}, "
        f"M_pq={params.M_pq}, nbits={params.nbits}, "
        f"M_hnsw={params.M_hnsw}, efC={params.ef_construction}, "
        f"efS={params.ef_search_values}"
    )

    n_gpus = faiss.get_num_gpus()
    if n_gpus >= 2 and len(embeddings) >= 1_000_000:
        runner = MultiGPUBenchmarkRunner(
            embeddings=embeddings,
            query_embeddings=query_embeddings,
            params=params,
            gpu_ids=settings.faiss.gpu_ids,
        )
        results.extend(runner.run())
    else:
        runner = BenchmarkRunner(embeddings, query_embeddings, k=args.k)
        results.append(runner.benchmark_flat())

        for ef_search in tqdm(params.ef_search_values, desc="HNSW sweep"):
            results.append(
                runner.benchmark_hnsw(
                    M=params.M_hnsw,
                    ef_search=ef_search,
                    ef_construction=params.ef_construction,
                )
            )

        for nprobe in tqdm(params.nprobe_values, desc="IVF sweep"):
            results.append(runner.benchmark_ivf(nlist=params.nlist, nprobe=nprobe))

        for nprobe in tqdm(params.nprobe_values, desc="IVFPQ sweep"):
            results.append(
                runner.benchmark_ivfpq(nlist=params.nlist, M_pq=params.M_pq, nprobe=nprobe)
            )

    if args.report:
        df = pd.DataFrame([r.model_dump() for r in results])
        df["params_str"] = df["params"].apply(_params_to_str)
        reporter = BenchmarkReporter(results_dir=settings.benchmark.results_dir)
        reporter.save_csv(df, f"benchmark_N{args.n_samples}")

        hnsw_df = df[df["index_type"].isin(["Flat", "HNSW"])]
        reporter.plot_latency_vs_recall(
            hnsw_df,
            title=f"Flat vs HNSW (N={args.n_samples:,})",
            filename=f"flat_vs_hnsw_N{args.n_samples}",
        )

        ivf_df = df[df["index_type"].isin(["Flat", "IVF", "IVF+PQ"])]
        reporter.plot_latency_vs_recall(
            ivf_df,
            title=f"IVF vs IVF+PQ (N={args.n_samples:,})",
            filename=f"ivf_vs_ivfpq_N{args.n_samples}",
        )

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
