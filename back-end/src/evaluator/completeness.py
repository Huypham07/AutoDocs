from __future__ import annotations

from api.models.docs_gen import Structure

from .base import BaseEvaluator


class CompletenessEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            name='Completeness Evaluator',
            description='Evaluates the completeness of the documentation.',
        )

    def evaluate(self, structure: Structure) -> float:
        # Implement the evaluation logic for completeness
        return self.score
