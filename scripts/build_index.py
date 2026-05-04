#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from rag_benchmark.adapters.embedding.sentence_transformer_adapter import SentenceTransformerAdapter
from rag_benchmark.config.settings import settings
from rag_benchmark.data.dataset_loader import load_medrag_wikipedia
from rag_benchmark.indexing.flat_index import FlatIndexBuilder
from rag_benchmark.indexing.hnsw_index import HNSWIndexBuilder
from rag_benchmark.indexing.ivf_index import IVFIndexBuilder
from rag_benchmark.indexing.ivfpq_index import IVFPQIndexBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=settings.dataset.max_samples)
    parser.add_argument("--index", type=str, default="all", choices=["flat", "hnsw", "ivf", "ivfpq", "all"])
    parser.add_argument("--cache-path", type=str, default=None)
    args = parser.parse_args()

    documents, texts = load_medrag_wikipedia(max_samples=args.n_samples, cache_path=args.cache_path)
    embedder = SentenceTransformerAdapter()
    embeddings = embedder.encode(texts)

    save_dir = Path(settings.faiss.index_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.index in ("flat", "all"):
        index = FlatIndexBuilder(embedder.dimension).build(embeddings)
        index.save(save_dir / f"flat_N{len(documents)}.index")

    if args.index in ("hnsw", "all"):
        index = HNSWIndexBuilder(embedder.dimension, M=32).build(embeddings)
        index.save(save_dir / f"hnsw_N{len(documents)}.index")

    n_vectors = len(embeddings)
    safe_nlist = min(max(4, int(n_vectors ** 0.5)), max(1, n_vectors // 39))

    if args.index in ("ivf", "all"):
        index = IVFIndexBuilder(embedder.dimension, nlist=safe_nlist).build(embeddings)
        index.save(save_dir / f"ivf_N{len(documents)}.index")

    if args.index in ("ivfpq", "all"):
        index = IVFPQIndexBuilder(embedder.dimension, nlist=safe_nlist, M_pq=48).build(embeddings)
        index.save(save_dir / f"ivfpq_N{len(documents)}.index")

    print("Index build complete.")


if __name__ == "__main__":
    main()
