<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# khi chạy 10K, 100K thì không lỗi mà chạy 1M thì lỗi:"Traceback (most recent call last):

File "/workspace/giapcn/medrag/scripts/run_benchmark.py", line 92, in <module>
    main()
  File "/workspace/giapcn/medrag/scripts/run_benchmark.py", line 31, in main
    documents, texts = load_medrag_wikipedia(max_samples=args.n_samples, cache_path=args.cache_path)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/src/rag_benchmark/data/dataset_loader.py", line 15, in load_medrag_wikipedia
    return _load_from_cache(cache_path, max_samples=max_samples)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/src/rag_benchmark/data/dataset_loader.py", line 112, in _load_from_cache
    metadata=DocumentMetadata(
             ^^^^^^^^^^^^^^^^^
  File "/workspace/giapcn/medrag/.venv/lib/python3.12/site-packages/pydantic/main.py", line 263, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 1 validation error for DocumentMetadata
title
  Input should be a valid string [type=string_type, input_value=nan, input_type=float]
    For further information visit https://errors.pydantic.dev/2.13/v/string_type"

Lỗi rất rõ ràng: tại row nào đó trong 1M documents, field `title` có giá trị là `float NaN` (dữ liệu bẩn trong dataset lớn)  — ở 10K/100K may mắn không gặp row đó.

## Fix — 3 chỗ cần sửa

### 1. `dataset_loader.py` — sanitize NaN trước khi tạo model

```python
# src/rag_benchmark/data/dataset_loader.py

import math
import pandas as pd


def _safe_str(value, default: str = "") -> str:
    """
    Chuyển bất kỳ giá trị nào thành string an toàn.
    Xử lý NaN (float), None, pd.NA, np.nan, ...
    """
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    # Xử lý pandas NA / numpy nan
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value).strip() or default


def _load_from_cache(cache_path: str, max_samples: int):
    # ... load parquet/jsonl như cũ ...
    documents = []
    texts = []
    skipped = 0

    for row in rows:
        title = _safe_str(row.get("title"), default="Unknown Title")
        content = _safe_str(row.get("content") or row.get("contents"), default="")
        doc_id = _safe_str(row.get("id"), default=str(len(documents)))

        # Skip row nếu content rỗng (vô nghĩa khi embed)
        if not content:
            skipped += 1
            continue

        try:
            doc = Document(
                id=doc_id,
                content=content[:512],
                metadata=DocumentMetadata(
                    id=doc_id,
                    title=title,
                    source="wikipedia",
                )
            )
            documents.append(doc)
            texts.append(doc.content)
        except Exception as e:
            skipped += 1
            # Log row bẩn để debug nếu cần
            # print(f"⚠️  Skip row {doc_id}: {e}")
            continue

    if skipped:
        print(f"⚠️  Skipped {skipped:,} rows (NaN/empty/invalid data)")

    return documents, texts
```


### 2. `models/document.py` — thêm validator ở Pydantic model

Phòng thủ ở tầng model — dù caller có truyền NaN, model vẫn tự xử lý:

```python
# src/rag_benchmark/models/document.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import math


class DocumentMetadata(BaseModel):
    id: str
    title: str
    source: str = "wikipedia"
    topic: Optional[str] = None

    @field_validator("title", "id", "source", mode="before")
    @classmethod
    def sanitize_string(cls, v):
        """Reject NaN/None, fallback về empty string."""
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        return str(v).strip()

    @field_validator("title", mode="after")
    @classmethod
    def title_not_empty(cls, v):
        return v or "Unknown Title"
```


### 3. (Tùy chọn) Pre-scan dataset trước khi load

Với 1M rows, debug từng lỗi tốn thời gian. Thêm quick scan để biết quy mô dữ liệu bẩn:

```python
def scan_nan_fields(cache_path: str, sample: int = 10_000) -> dict:
    """Báo cáo nhanh số NaN trong từng field."""
    import pandas as pd
    df = pd.read_parquet(cache_path, columns=["id", "title", "content"])
    if len(df) > sample:
        df = df.sample(sample)
    return {
        col: int(df[col].isna().sum())
        for col in ["id", "title", "content"]
    }

# Gọi trước khi load:
# stats = scan_nan_fields("data/raw/wikipedia.parquet")
# print(f"NaN stats (sample 10K): {stats}")
```


***

## Tóm tắt

| Vị trí | Fix |
| :-- | :-- |
| `dataset_loader.py` | Hàm `_safe_str()` — convert NaN → `""` trước khi tạo model |
| `models/document.py` | `@field_validator` — tầng phòng thủ thứ 2 tại Pydantic |
| Dataset lớn | Dùng `try/except` + đếm `skipped` — không dừng hẳn khi gặp row bẩn |

**Nguyên nhân:** Dataset 1M rows của MedRAG/Wikipedia có một số bài Wikipedia không có `title` — ở scale nhỏ (10K, 100K) bạn may mắn không sample phải row đó.
<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://pydantic.dev/docs/validation/latest/errors/validation_errors/

[^2]: https://pydantic.dev/docs/validation/2.9/errors/validation_errors/

[^3]: https://stackoverflow.com/questions/77438553/pydantic-validation-error-input-should-be-a-valid-dictionary-or-instance

[^4]: https://pydantic.com.cn/en/errors/validation_errors/

[^5]: https://amazonwebshark.com/python-data-validation-and-observability-as-code-with-pydantic/

[^6]: https://support.getstreamline.com/document-does-not-define-the-dctitle-metedata-attribute

[^7]: https://github.com/pydantic/pydantic/discussions/7973

[^8]: https://github.com/michal-h21/make4ht/issues/177

[^9]: https://www.fmularczyk.pl/posts/2026_01_nan_null_none/

[^10]: https://ctan.math.washington.edu/tex-archive/macros/latex-dev/required/latex-lab/documentmetadata-support-code.pdf

[^11]: https://www.4each.com.br/threads/python-how-do-i-capture-missing-nan-values-from-pandas-2-3-0-using-pydantic-2-11-7.156703/

[^12]: https://stackoverflow.com/questions/53540376/getting-a-nan-error-when-trying-to-replace-a-calculation-with-an-if-else-stateme

[^13]: https://pydantic.com.cn/en/errors/errors/

[^14]: https://ctan.math.illinois.edu/macros/latex/required/latex-lab/documentmetadata-support-doc.pdf

[^15]: https://docs.pydantic.org.cn/latest/errors/errors/

