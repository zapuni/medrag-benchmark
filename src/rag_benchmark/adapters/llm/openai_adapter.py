import asyncio
import time

from openai import AsyncOpenAI

from .base import LLMAdapter
from ...config.settings import settings
from ...models.result import GenerationResult


class OpenAIAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.llm.openai_api_key,
            base_url=settings.llm.openai_base_url,
        )
        self.model = settings.llm.openai_model

    async def generate(self, query: str, context: list[str]) -> GenerationResult:
        context_text = "\n---\n".join(context)
        t0 = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical information assistant."},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}",
                },
            ],
            temperature=0.1,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            answer=response.choices[0].message.content or "",
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
        )

    def health_check(self) -> bool:
        try:
            asyncio.run(self.client.models.list())
            return True
        except Exception:
            return False
