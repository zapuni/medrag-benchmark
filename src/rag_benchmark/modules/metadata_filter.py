from ..models.result import SearchResult


class MetadataFilter:
    def __init__(self, source: str = "wikipedia") -> None:
        self.source = source

    def filter(self, candidates: list[SearchResult], keyword: str | None = None) -> list[SearchResult]:
        filtered: list[SearchResult] = []
        for item in candidates:
            if item.metadata.source != self.source:
                continue
            if keyword and keyword.lower() not in item.metadata.title.lower():
                continue
            filtered.append(item)
        return filtered
