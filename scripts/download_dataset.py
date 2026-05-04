#!/usr/bin/env python3
import argparse
from pathlib import Path

from rag_benchmark.config.settings import settings
from rag_benchmark.data.dataset_loader import save_medrag_wikipedia_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=settings.dataset.max_samples)
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/medrag_wikipedia_cache.csv",
    )
    args = parser.parse_args()

    Path(settings.dataset.cache_dir).mkdir(parents=True, exist_ok=True)
    save_medrag_wikipedia_cache(max_samples=args.n_samples, output_path=args.output)


if __name__ == "__main__":
    main()
