from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Message


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message], system_prompt: str) -> str:
        """Send messages with system prompt, return response text."""
        ...
