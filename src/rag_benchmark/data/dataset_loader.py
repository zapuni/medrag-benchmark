import math
from pathlib import Path

import pandas as pd
from datasets import IterableDataset, load_dataset
from tqdm import tqdm

from ..config.settings import settings
from ..models.document import Document, DocumentMetadata


def load_medrag_wikipedia(
    max_samples: int | None = None, cache_path: str | None = None
) -> tuple[list[Document], list[str]]:
    if cache_path:
        return _load_from_cache(cache_path, max_samples=max_samples)

    n = max_samples or settings.dataset.max_samples
    print(f"Loading {n:,} documents from {settings.dataset.name}...")
    dataset: IterableDataset = load_dataset(
        settings.dataset.name,
        split=settings.dataset.split,
        streaming=True,
        trust_remote_code=True,
        cache_dir=settings.dataset.cache_dir,
    )

    documents: list[Document] = []
    texts: list[str] = []

    for i, row in enumerate(tqdm(dataset, total=n, desc="Loading")):
        if i >= n:
            break
        text = row.get("contents", f"{row['title']}\n{row['content']}")
        doc = Document(
            id=row["id"],
            content=text[:512],
            metadata=DocumentMetadata(
                id=row["id"],
                title=row["title"],
                source="wikipedia",
            ),
        )
        documents.append(doc)
        texts.append(doc.content)

    print(f"Loaded {len(documents):,} documents")
    return documents, texts


def save_medrag_wikipedia_cache(max_samples: int, output_path: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset: IterableDataset = load_dataset(
        settings.dataset.name,
        split=settings.dataset.split,
        streaming=True,
        trust_remote_code=True,
        cache_dir=settings.dataset.cache_dir,
    )

    rows = []
    for i, row in enumerate(tqdm(dataset, total=max_samples, desc="Caching")):
        if i >= max_samples:
            break
        contents = row.get("contents", f"{row['title']}\n{row['content']}")
        rows.append(
            {
                "id": row["id"],
                "title": row["title"],
                "content": row.get("content", ""),
                "contents": contents,
            }
        )

    df = pd.DataFrame(rows)
    if output.suffix == ".csv":
        df.to_csv(output, index=False)
    elif output.suffix in {".parquet", ".pq"}:
        df.to_parquet(output, index=False)
    else:
        raise ValueError("output_path must end with .csv or .parquet")

    print(f"Cached dataset: {output}")
    return output


def _safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _load_from_cache(
    cache_path: str, max_samples: int | None = None
) -> tuple[list[Document], list[str]]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("cache_path must end with .csv or .parquet")

    if max_samples:
        df = df.head(max_samples)

    documents: list[Document] = []
    texts: list[str] = []
    skipped = 0
    for row in df.itertuples(index=False):
        doc_id = _safe_str(getattr(row, "id", None), default=str(len(documents)))
        title = _safe_str(getattr(row, "title", None), default="Unknown Title")
        content = _safe_str(getattr(row, "content", None))
        contents = _safe_str(getattr(row, "contents", None))
        text = contents or f"{title}\n{content}".strip()
        if not text:
            skipped += 1
            continue
        try:
            doc = Document(
                id=doc_id,
                content=text[:512],
                metadata=DocumentMetadata(
                    id=doc_id,
                    title=title,
                    source="wikipedia",
                ),
            )
        except Exception:
            skipped += 1
            continue
        documents.append(doc)
        texts.append(doc.content)

    if skipped:
        print(f"Skipped {skipped:,} invalid rows")
    print(f"Loaded {len(documents):,} documents from cache")
    return documents, texts
