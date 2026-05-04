from dataclasses import dataclass
import math


@dataclass
class BenchmarkParams:
    nlist: int
    nprobe_values: list[int]
    M_pq: int
    nbits: int
    min_train_vectors: int
    M_hnsw: int
    ef_construction: int
    ef_search_values: list[int]


def get_optimal_params(n: int, dimension: int = 384) -> BenchmarkParams:
    if n <= 10_000:
        nlist = max(25, int(4 * math.sqrt(n)))
    elif n <= 100_000:
        nlist = max(100, int(4 * math.sqrt(n)))
    else:
        nlist = 4096

    nlist = min(nlist, n // 39)

    max_nprobe = min(nlist, 512)
    nprobe_values = sorted({1, 4, 8, 16, 32, 64, 128, 256, 512, nlist})
    nprobe_values = [v for v in nprobe_values if v <= max_nprobe]

    max_M_pq = dimension // 4
    candidates = [
        m
        for m in [16, 24, 32, 48, 64, 96, 128]
        if dimension % m == 0 and m <= max_M_pq
    ]

    # if n < 10_000:
    #     M_pq = candidates[len(candidates) // 3] if candidates else 16
    # elif n < 100_000:
    #     M_pq = candidates[len(candidates) // 2] if candidates else 32
    # elif n < 1_000_000:
    #     M_pq = candidates[-1] if candidates else 48
    # else:
    #     M_pq = 48 if 48 in candidates else (candidates[-1] if candidates else 48)
    M_pq = 48

    if n < 10_000:
        M_hnsw, ef_construction = 16, 100
    elif n < 100_000:
        M_hnsw, ef_construction = 32, 200
    elif n < 1_000_000:
        M_hnsw, ef_construction = 48, 200
    else:
        M_hnsw, ef_construction = 48, 200

    max_ef = min(512, n)
    ef_search_values = sorted({16, 32, 64, 128, 256, 512})
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
