from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: Literal["openai", "ollama", "vllm"] = "openai"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


class EmbeddingSettings(BaseSettings):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    batch_size: int = 1024

    model_config = SettingsConfigDict(env_file=".env", env_prefix="EMBEDDING_", extra="ignore")


class RerankerSettings(BaseSettings):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RERANKER_", extra="ignore")


class DatasetSettings(BaseSettings):
    name: str = "MedRAG/wikipedia"
    split: str = "train"
    max_samples: int = 10_000
    cache_dir: str = "./data/raw"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATASET_", extra="ignore")


class FAISSSettings(BaseSettings):
    use_gpu: bool = True
    gpu_id: int = 0
    gpu_ids: list[int] = [0, 1]
    temp_memory_mb: int = 1024
    kmeans_niter: int = 20
    kmeans_max_points_per_centroid: int = 256
    index_save_dir: str = "./indexes"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FAISS_", extra="ignore")


class BenchmarkSettings(BaseSettings):
    top_k: int = 10
    top_ann: int = 50
    n_queries: int = 100
    query_batch_size: int = 128
    hnsw_threads: int = 8
    results_dir: str = "./results"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BENCHMARK_", extra="ignore")


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    dataset: DatasetSettings = DatasetSettings()
    faiss: FAISSSettings = FAISSSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
