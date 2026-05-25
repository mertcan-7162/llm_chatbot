from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.config import settings
from app.llm.prompts import CHAT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self.model = settings.llm_model

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = settings.openai_api_key
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to your .env file to enable query answering."
                )
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def answer_with_context(
        self,
        query: str,
        retrieved_chunks: list[str],
    ) -> str:
        client = self._get_client()
        context = "\n---\n".join(retrieved_chunks)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def chat(self, query: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
