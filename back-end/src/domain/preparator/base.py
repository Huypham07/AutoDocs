from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional

from adalflow.core.types import Document
from adalflow.core.types import List
from pydantic import BaseModel


class PreparatorInput(BaseModel):
    """Input data for preparator."""
    repo_url: str
    access_token: Optional[str] = None


class PreparatorOutput(BaseModel):
    """Output data for preparator."""
    transformed_docs: List[Document]


class BasePreparator(ABC):
    """Base class for preparators."""

    @abstractmethod
    def prepare(self, input_data: PreparatorInput) -> PreparatorOutput:
        """Prepare data for further processing."""
        raise NotImplementedError()
