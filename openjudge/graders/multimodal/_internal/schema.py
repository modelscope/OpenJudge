# -*- coding: utf-8 -*-
"""
Multimodal Evaluation Schema Definitions

Pydantic models for structured evaluation data.
"""

from typing import List, Tuple

from pydantic import BaseModel, field_validator


class EvaluationSteps(BaseModel):
    """
    Container for evaluation steps

    Used by G-Eval to store generated evaluation steps.

    Attributes:
        steps: List of evaluation step descriptions

    Example:
        >>> steps = EvaluationSteps(steps=[
        ...     "Analyze the image content",
        ...     "Compare with text prompt",
        ...     "Evaluate alignment"
        ... ])
    """

    steps: List[str]


class Rubric(BaseModel):
    """
    Scoring rubric for evaluation

    Defines expected outcomes for score ranges.

    Attributes:
        score_range: Tuple of (min_score, max_score)
        expected_outcome: Description of what this score range means

    Example:
        >>> rubric = Rubric(
        ...     score_range=(7, 9),
        ...     expected_outcome="High quality with minor issues"
        ... )
    """

    score_range: Tuple[int, int]
    expected_outcome: str

    @field_validator("score_range")
    @classmethod
    def validate_score_range(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        """
        Validate that score range is valid

        Args:
            value: Tuple of (start, end) scores

        Returns:
            Validated score range

        Raises:
            ValueError: If scores are out of range or invalid
        """
        start, end = value
        if not (0 <= start <= 10 and 0 <= end <= 10):
            raise ValueError(
                "Both Rubric's 'score_range' values must be between 0 and 10 inclusive.",
            )
        if start > end:
            raise ValueError(
                "Rubric's 'score_range' start must be less than or equal to end.",
            )
        return value


__all__ = [
    "EvaluationSteps",
    "Rubric",
]
