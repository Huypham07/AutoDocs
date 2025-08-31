from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional

from adalflow.core.types import Document
from adalflow.core.types import List


class BasePreparator(ABC):
    """Base class for preparators."""

    @abstractmethod
    def prepare(self, repo_url: str, access_token: Optional[str] = None) -> List[Document]:
        """Prepare data for further processing."""
        raise NotImplementedError()
