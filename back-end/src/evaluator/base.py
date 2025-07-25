from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from api.models.docs_gen import Structure


class BaseEvaluator(ABC):
    """
    Base class for all docsgen evaluator

    This class provides the foundation for implementing various docsgen quality evaluator. Each Evaluator should focus on a specific aspect of docsgen quality such as completeness, helpfullness, or redundancy.

    Attributes:
        score (float): The evaluator score, range from 0 to 1.
        name (str): The name of the evaluator.
        description (str): A description of what this evaluator checks.
    """

    def __init__(self, name: str, description: str):
        self._score: float = 0.0
        self._name = name
        self._description = description

    @property
    def score(self) -> float:
        """
        Return the current score of the evaluator.

        Returns:
            float: A score between 0 and 1 representing the quality of the documentation.
        """
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError('Score must be between 0 and 1.')
        self._score = value

    @abstractmethod
    def evaluate(self, structure: Structure) -> float:
        """
        Evaluate the given structure and return a score.

        Args:
            structure (Structure): The documentation structure to evaluate.

        Returns:
            float: The evaluation score for the documentation.
        """
        raise NotImplementedError()
