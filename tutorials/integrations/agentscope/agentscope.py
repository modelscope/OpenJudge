# -*- coding: utf-8 -*-

"""
Integration module for connecting RM-Gallery graders with AgentScope metrics.

This module provides a wrapper class that allows RM-Gallery graders to be used
as AgentScope metrics, enabling interoperability between the two frameworks.
"""

from typing import Any, Dict

from pydantic import BaseModel

from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderResult

# Type aliases for AgentScope components (using BaseModel as placeholder)
MetricBase = BaseModel
SolutionOutput = BaseModel
MetricResult = BaseModel


class RMGalleryMetric(MetricBase):
    """A wrapper class that adapts RM-Gallery graders to AgentScope metrics.

    This class serves as a bridge between RM-Gallery's grading system and
    AgentScope's metric system, allowing RM-Gallery graders to be used
    directly as AgentScope metrics.

    Attributes:
        grader (BaseGrader): The RM-Gallery grader instance to be wrapped.
    """

    def __init__(self, grader: BaseGrader):
        """Initialize the RMGalleryMetric wrapper.

        Args:
            grader: An RM-Gallery grader instance that will be used to
                   perform evaluations when this metric is called.
        """
        super().__init__()
        self.grader = grader

    async def _convert_solution_to_dict(
        self,
        solution: SolutionOutput,
    ) -> Dict[Any, Any]:
        """Convert an AgentScope solution to a dictionary for RM-Gallery.

        This abstract method needs to be implemented by subclasses to define
        how AgentScope solution objects should be converted to the dictionary
        format expected by RM-Gallery graders.

        Args:
            solution: An AgentScope solution object to be converted.

        Returns:
            A dictionary containing the data needed by the RM-Gallery grader.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented.
        """
        raise NotImplementedError(
            "Subclasses must implement _convert_solution_to_dict method",
        )

    async def _convert_grader_result_to_metric_result(
        self,
        grader_result: GraderResult,
    ) -> MetricResult:
        """Convert RM-Gallery grader result to AgentScope metric result.

        This abstract method needs to be implemented by subclasses to define
        how RM-Gallery grader results should be converted to AgentScope metric results.

        Args:
            grader_result: A result from an RM-Gallery grader evaluation.

        Returns:
            A metric result in the format expected by AgentScope.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented.
        """
        raise NotImplementedError(
            "Subclasses must implement _convert_grader_result_to_metric_result method",
        )

    async def __call__(self, solution: SolutionOutput, *args, **kwargs) -> MetricResult:
        """Evaluate the solution using the wrapped RM-Gallery grader.

        This method converts the AgentScope solution to the format expected
        by RM-Gallery, runs the evaluation, and converts the result back
        to the format expected by AgentScope.

        Args:
            solution: An AgentScope solution object to be evaluated.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation result in AgentScope metric format.
        """
        # Convert AgentScope solution to RM-Gallery compatible data format
        data = await self._convert_solution_to_dict(solution)

        # Evaluate using RM-Gallery grader
        grader_result = await self.grader.aevaluate(**data)

        # Convert result back to AgentScope format
        return await self._convert_grader_result_to_metric_result(grader_result)
