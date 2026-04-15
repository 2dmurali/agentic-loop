from __future__ import annotations

from anthropic import AsyncAnthropic

from src.llm.base import LLMProvider
from src.models import Message


class ClaudeProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model_id: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_id = model_id
        self.max_tokens = max_tokens

    async def generate(self, messages: list[Message], system_prompt: str) -> str:
        """Send messages with system prompt, return response text."""
        anthropic_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = await self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=anthropic_messages,
        )

        return response.content[0].text
