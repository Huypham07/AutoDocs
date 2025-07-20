from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str = '') -> str:
        """Generate a response from the LLM given a prompt and context."""
        raise NotImplementedError()
