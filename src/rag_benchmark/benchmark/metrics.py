import numpy as np


def compute_recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    recalls = []
    for i in range(len(retrieved)):
        gt_set = set(ground_truth[i][:k].tolist())
        retrieved_set = set(retrieved[i][:k].tolist())
        recalls.append(len(retrieved_set & gt_set) / max(1, len(gt_set)))
    return float(np.mean(recalls))


def compute_latency_stats(times_ms: list[float]) -> dict:
    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
