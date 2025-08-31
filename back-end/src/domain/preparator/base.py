from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

from langchain.schema import Document


class BasePreparator(ABC):
    """Base class for preparators."""

    @abstractmethod
    def prepare(self, repo_url: str, access_token: Optional[str] = None) -> List[Document]:
        """Prepare data for further processing."""
        raise NotImplementedError()
