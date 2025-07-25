from __future__ import annotations

from api.models.docs_gen import Structure

from .base import BaseEvaluator


class HelpfulnessEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            name='Helpfulness Evaluator',
            description='Evaluates the helpfulness of the documentation.',
        )

    def evaluate(self, structure: Structure) -> float:
        # Implement the evaluation logic for completeness
        return self.score
