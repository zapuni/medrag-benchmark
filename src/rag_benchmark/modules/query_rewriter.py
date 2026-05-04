import asyncio

from ..adapters.llm.openai_adapter import OpenAIAdapter
from ..config.settings import settings
from ..models.query import RewrittenQuery


class QueryRewriter:
    def __init__(self) -> None:
        self._llm = None
        if settings.llm.provider == "openai" and settings.llm.openai_api_key:
            self._llm = OpenAIAdapter()

    def rewrite(self, query: str) -> RewrittenQuery:
        if not self._llm:
            return RewrittenQuery(original=query, rewritten=query)
        prompt = (
            "Rewrite the following medical question to be more specific and add relevant "
            f"medical keywords:\n\n{query}"
        )
        rewritten = asyncio.run(self._llm.generate(prompt, context=[])).answer
        text = rewritten.strip() if rewritten else query
        return RewrittenQuery(original=query, rewritten=text)
