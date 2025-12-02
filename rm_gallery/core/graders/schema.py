# -*- coding: utf-8 -*-
"""
Schemas for grading tasks.

This module defines the data schemas used in grading tasks, including grader modes,
result structures, and error handling.
"""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    This enum defines the two primary modes that graders can operate in:
    pointwise (evaluating individual samples) and listwise (ranking multiple samples).

    Attributes:
        POINTWISE: Pointwise grader mode.
        LISTWISE: Listwise grader mode.

    Example:
        >>> mode = GraderMode.POINTWISE
        >>> print(mode.value)
        pointwise
        >>>
        >>> mode = GraderMode.LISTWISE
        >>> print(mode.value)
        listwise
    """

    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class GraderResult(BaseModel):
    """Base class for grader results.

    This Pydantic model defines the structure for grader results,
    which include a reason and optional metadata.

    Attributes:
        name (str): The name of the grader.
        reason (str): The reason for the result.
        metadata (Dict[str, Any]): The metadata of the grader result.

    Example:
        >>> result = GraderResult(
        ...     name="test_grader",
        ...     reason="Test evaluation completed",
        ...     metadata={"duration": 0.1}
        ... )
        >>> print(result.name)
        test_grader
    """

    name: str = Field(default=..., description="The name of the grader")
    reason: str = Field(default="", description="The reason for the result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderScore(GraderResult):
    """Grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        score (float): A numerical score assigned by the grader.
        reason (str): Explanation of how the score was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> score_result = GraderScore(
        ...     name="accuracy_grader",
        ...     score=0.85,
        ...     reason="Answer is mostly accurate",
        ...     metadata={"confidence": 0.9}
        ... )
        >>> print(score_result.score)
        0.85
    """

    score: float = Field(default=..., description="score")


class GraderScoreCallback(BaseModel):
    """Callback for grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        score (float): A numerical score assigned by the grader.
        reason (str): Explanation of how the score was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> callback = GraderScoreCallback(
        ...     score=0.9,
        ...     reason="High confidence in evaluation",
        ...     metadata={"model_used": ""}
        ... )
        >>> print(callback.score)
        0.9
    """

    score: float = Field(default=..., description="score")
    reason: str = Field(default=..., description="reason")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderRank(GraderResult):
    """Grader rank result.

    Represents a ranking of items assigned by a grader along with a reason.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> rank_result = GraderRank(
        ...     name="relevance_ranker",
        ...     rank=[1, 3, 2],
        ...     reason="First response is most relevant",
        ...     metadata={"criteria": "relevance"}
        ... )
        >>> print(rank_result.rank)
        [1, 3, 2]
    """

    rank: List[int] = Field(default=..., description="rank")


class GraderRankCallback(BaseModel):
    """Callback for grader rank result, used for .

    Represents a ranking of items assigned by a grader along with a reason.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> callback = GraderRankCallback(
        ...     rank=[2, 1],
        ...     reason="Second response is more relevant",
        ...     metadata={"criteria": "clarity"}
        ... )
        >>> print(callback.rank)
        [2, 1]
    """

    rank: List[int] = Field(default=..., description="rank")
    reason: str = Field(default=..., description="reason")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderError(GraderResult):
    """Grader error result.

    Represents an error encountered during evaluation.

    Attributes:
        error (str): The error message.
        reason (str): Description of the error encountered during evaluation.
        metadata (Dict[str, Any]): Optional additional error information.

    Example:
        >>> error_result = GraderError(
        ...     name="test_grader",
        ...     error="Timeout occurred",
        ...     reason="Model took too long to respond",
        ...     metadata={"timeout_seconds": 30}
        ... )
        >>> print(error_result.error)
        Timeout occurred
    """

    error: str = Field(default=..., description="error")
