from __future__ import annotations

from typing import Optional

from api.models.docs_gen import Page
from pydantic import BaseModel


class GeneratorInput(BaseModel):
    """Input data for documentation generation."""
    repo_url: str
    access_token: Optional[str] = None


class GeneratorOutput(BaseModel):
    """Output data for documentation generation."""
    status: str = 'pending'
    is_existing: bool = False
    processing_time: Optional[float] = None
    document_id: Optional[str] = None


class PageProcessingTask(BaseModel):
    """Task data for processing individual pages."""
    document_id: str
    page: Page
    owner: str
    repo_name: str
    repo_url: str
    access_token: Optional[str] = None
