from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from shared.base import BaseModel


class BaseService(ABC, BaseModel):
    @abstractmethod
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError()


class AsyncBaseService(ABC, BaseModel):
    @abstractmethod
    async def process(self, inputs: Any) -> Any:
        raise NotImplementedError()
