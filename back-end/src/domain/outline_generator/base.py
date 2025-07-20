from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel


class OutlineGeneratorInput(BaseModel):
    """Input data for outline generation."""
    repo_url: str
    access_token: Optional[str] = None
    owner: str
    repo_name: str


class OutlineGeneratorOutput(BaseModel):
    """Output data for outline generation."""
    outline: str


class BaseOutlineGenerator(ABC):
    """Base class for outline generators."""

    @abstractmethod
    def generate(self, input_data: OutlineGeneratorInput):
        """Generate an outline based on the provided repository URL."""
        raise NotImplementedError()

    @abstractmethod
    async def generate_stream(self, input_data: OutlineGeneratorInput):
        """Generate an outline based on the provided repository URL."""
        raise NotImplementedError()
