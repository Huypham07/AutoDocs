from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from api.models.docs_gen import Page
from pydantic import BaseModel


class ContentGeneratorInput(BaseModel):
    """Input data for content generation."""
    repo_url: str
    owner: str
    repo_name: str
    page: Page


class BaseContentGenerator(ABC):
    """Base class for content generators."""

    @abstractmethod
    def generate(self, input_data: ContentGeneratorInput) -> str:
        """Generate content based on the provided repository URL or file."""
        raise NotImplementedError()
