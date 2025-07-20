from __future__ import annotations

from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    class Config:
        """Configuration of the Pydantic Object"""

        # Allowing arbitrary types for class validation
        arbitrary_types_allowed = True
