from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.config import settings
from app.llm.prompts import CHAT_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT, SUMMARY_PROMPT

if TYPE_CHECKING:
    from app.models.schemas import ChatMessage

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

    def _build_messages(
        self,
        system_prompt: str,
        query: str,
        document_context: str | None = None,
        conversation_history: list[ChatMessage] | None = None,
        summary: str | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{summary}",
            })
        if document_context:
            messages.append({
                "role": "system",
                "content": f"Relevant document excerpts:\n{document_context}",
            })
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})
        return messages

    async def answer_with_context(
        self,
        query: str,
        retrieved_chunks: list[str],
        conversation_history: list[ChatMessage] | None = None,
        summary: str | None = None,
    ) -> str:
        client = self._get_client()
        context = "\n---\n".join(retrieved_chunks)
        messages = self._build_messages(
            RAG_SYSTEM_PROMPT, query, context, conversation_history, summary,
        )
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def chat(
        self,
        query: str,
        conversation_history: list[ChatMessage] | None = None,
        summary: str | None = None,
    ) -> str:
        client = self._get_client()
        messages = self._build_messages(
            CHAT_SYSTEM_PROMPT, query, None, conversation_history, summary,
        )
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    async def summarize(self, messages: list[ChatMessage]) -> str:
        formatted = "\n".join(f"{m.role}: {m.content}" for m in messages)
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": formatted},
            ],
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content or ""
