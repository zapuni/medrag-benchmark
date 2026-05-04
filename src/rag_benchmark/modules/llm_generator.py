import asyncio

from ..adapters.llm.openai_adapter import OpenAIAdapter
from ..config.settings import settings
from ..models.result import GenerationResult


class LLMGenerator:
    def __init__(self) -> None:
        if settings.llm.provider != "openai":
            raise ValueError("Only openai provider is implemented in this project.")
        if not settings.llm.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        self._llm = OpenAIAdapter()

    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        return asyncio.run(self._llm.generate(query, contexts))
