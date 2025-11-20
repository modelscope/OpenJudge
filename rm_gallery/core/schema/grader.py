# -*- coding: utf-8 -*-

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    Attributes:
        POINTWISE: Pointwise grader mode.
        LISTWISE: Listwise grader mode.
    """

    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class GraderResult(BaseModel):
    """Base class for grader results.

    This Pydantic model defines the structure for grader results,
    which include a reason and optional metadata.

    Attributes:
        reason (str): The reason for the result.
        metadata (Dict[str, Any]): The metadata of the grader result.
    """

    name: str = Field(default=..., description="The name of the grader")
    reason: str = Field(default=..., description="The reason for the result")
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
    """

    score: float = Field(default=..., description="score")


class _GraderScore(BaseModel):
    """Grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        score (float): A numerical score assigned by the grader.
        reason (str): Explanation of how the score was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.
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
    """

    rank: List[int] = Field(default=..., description="rank")


class _GraderRank(BaseModel):
    """Grader rank result.

    Represents a ranking of items assigned by a grader along with a reason.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.
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
        reason (str): Description of the error encountered during evaluation.
        metadata (Dict[str, Any]): Optional additional error information.
    """


class GraderInfo(BaseModel):
    """Grader info.

    Represents meta information about a grader.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description (str): The description of the grader.
    """

    name: str = Field(default=..., description="The name of the grader")
    mode: GraderMode = Field(default=..., description="The grader mode")
    description: str = Field(
        default=...,
        description="The description of the grader",
    )
