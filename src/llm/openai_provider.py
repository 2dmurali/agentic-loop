from __future__ import annotations

from openai import AsyncOpenAI

from src.llm.base import LLMProvider
from src.models import Message


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-4o",
        max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.max_tokens = max_tokens

    async def generate(self, messages: list[Message], system_prompt: str) -> str:
        """Send messages with system prompt, return response text."""
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(
            {"role": msg.role, "content": msg.content} for msg in messages
        )

        response = await self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=openai_messages,
        )

        return response.choices[0].message.content
