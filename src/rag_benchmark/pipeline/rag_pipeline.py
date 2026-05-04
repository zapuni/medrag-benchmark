from dataclasses import dataclass

import numpy as np

from ..adapters.embedding.sentence_transformer_adapter import SentenceTransformerAdapter
from ..config.settings import settings
from ..data.dataset_loader import load_medrag_wikipedia
from ..models.result import GenerationResult, SearchResult
from ..modules.ann_search import AnnSearcher
from ..modules.llm_generator import LLMGenerator
from ..modules.metadata_filter import MetadataFilter
from ..modules.query_rewriter import QueryRewriter
from ..modules.reranker import Reranker


@dataclass
class PipelineOutput:
    rewritten_query: str
    ann_results: list[SearchResult]
    filtered_results: list[SearchResult]
    reranked_results: list[SearchResult]
    generation: GenerationResult | None


class RAGPipeline:
    def __init__(self, index_type: str = "HNSW") -> None:
        self.index_type = index_type
        self.query_rewriter = QueryRewriter()
        self.reranker = Reranker()
        self.filterer = MetadataFilter()
        self.embedder = SentenceTransformerAdapter()
        self.llm = None
        if settings.llm.provider == "openai" and settings.llm.openai_api_key:
            self.llm = LLMGenerator()

        self.documents, self.texts = load_medrag_wikipedia()
        self.embeddings = self.embedder.encode(self.texts)
        self.searcher = AnnSearcher(index_type, self.embeddings, self.documents)

    def run(self, query: str, top_ann: int = 50, top_k: int = 5) -> PipelineOutput:
        rewritten = self.query_rewriter.rewrite(query)
        q_vec = self.embedder.encode([rewritten.rewritten])[0]
        ann_results = self.searcher.search(q_vec, top_ann)
        filtered = self.filterer.filter(ann_results)
        reranked = self.reranker.rerank(query, filtered, top_k=top_k)
        generation = None
        if self.llm:
            contexts = [r.text for r in reranked]
            generation = self.llm.generate(query, contexts)
        return PipelineOutput(
            rewritten_query=rewritten.rewritten,
            ann_results=ann_results,
            filtered_results=filtered,
            reranked_results=reranked,
            generation=generation,
        )
