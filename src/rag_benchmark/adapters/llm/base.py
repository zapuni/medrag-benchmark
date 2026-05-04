from abc import ABC, abstractmethod

from ...models.result import GenerationResult


class LLMAdapter(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, query: str, context: list[str]) -> GenerationResult:
        """Generate an answer from query and context."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the adapter is healthy."""
        raise NotImplementedError
