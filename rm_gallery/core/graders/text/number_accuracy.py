"""Number Accuracy Grader Module.

This module implements a grader that evaluates the numerical accuracy of response content
by comparing numbers extracted from the response text with those in a ground_truth text.
It is particularly useful for evaluating mathematical computations, data reporting,
and other content where numeric precision is important.

The NumberAccuracyGrader class identifies numerical values in both texts and compares
them with a configurable tolerance to determine accuracy scores.
"""

import re
from typing import Any, List

from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class NumberAccuracyGrader(BaseGrader):
    """
    Check numerical calculation accuracy by comparing numbers in response vs ground_truth content.

    This reward verifies if the numbers in the response content match
    the numbers in the ground_truth content within a specified tolerance.

    Methods:
        evaluate: Extracts and compares numbers between response and ground_truth content.

    Examples:
        >>> grader = NumberAccuracyGrader(tolerance=1e-6)
        >>> result = await grader.aevaluate(
        ...     response="The result is 3.14159",
        ...     ground_truth="The result is 3.14159"
        ... )
        >>> print(result.score)
        1.0
        >>> result = await grader.aevaluate(
        ...     response="The result is 3.14",
        ...     ground_truth="The result is 3.14159"
        ... )
        >>> print(result.score)
        0.0
    """

    def __init__(self, tolerance: float = 1e-6, **kwargs: Any) -> None:
        """"""
        super().__init__(
            name="number_accuracy",
            mode=GraderMode.POINTWISE,
            description="Check numerical calculation accuracy by comparing numbers in response vs ground_truth content",
            **kwargs,
        )
        self.tolerance = tolerance

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        # Match integers and floating point numbers
        number_pattern = r"-?\d+\.?\d*"
        numbers = re.findall(number_pattern, text)
        return [float(n) for n in numbers if n]

    async def aevaluate(self, response: str, ground_truth: str) -> GraderScore:
        """
        Calculate number accuracy by comparing extracted numbers from both texts.

        This method extracts numerical values from both the response and ground_truth texts,
        then compares them in order to compute an accuracy score. The score represents the
        proportion of numbers in the ground_truth that were correctly reproduced in the response text.
        Numbers are compared with a tolerance threshold to account for floating-point precision issues.

        Args:
            response (str): Generated content to evaluate. This is typically the output
                from a language model that we want to assess for numerical accuracy.
            ground_truth (str): Reference answer containing expected numbers. The numbers
                in this text are considered the ground truth.

        Returns:
            GraderScore: Result containing the number accuracy score and explanation.
                - score (float): Proportion of correctly matched numbers (between 0.0 and 1.0)
                - reason (str): Explanation of the scoring result including count of correct numbers
                - metadata (Dict): Contains details about extracted numbers and comparison including:
                    - accuracy (float): Computed accuracy score
                    - correct_numbers (int): Count of correctly matched numbers
                    - total_ground_truth_numbers (int): Total count of numbers in ground_truth text
                    - response_numbers (List[float]): Numbers extracted from response text
                    - ground_truth_numbers (List[float]): Numbers extracted from ground_truth text
                    - tolerance (float): Tolerance used for number comparison

        Examples:
            >>> grader = NumberAccuracyGrader(tolerance=1e-6)
            >>> result = await grader.aevaluate(
            ...     response="The result is 3.14159",
            ...     ground_truth="The result is 3.14159"
            ... )
            >>> print(result.score)
            1.0
            >>> print(result.reason)
            Number accuracy: 1/1 numbers correct
            >>> result = await grader.aevaluate(
            ...     response="The temperatures are 25.5 and 30.2 degrees",
            ...     ground_truth="The temperatures are 25.5 and 30.0 degrees"
            ... )
            >>> print(result.score)
            0.5
            >>> print(result.reason)
            Number accuracy: 1/2 numbers correct
        """
        response_numbers = self._extract_numbers(response)
        ground_truth_numbers = self._extract_numbers(ground_truth)

        if not ground_truth_numbers:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No ground_truth numbers to compare",
                metadata={
                    "response_numbers": response_numbers,
                    "ground_truth_numbers": ground_truth_numbers,
                },
            )
        if not response_numbers:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No numbers found in response content",
                metadata={
                    "response_numbers": response_numbers,
                    "ground_truth_numbers": ground_truth_numbers,
                },
            )

        # Compare numbers (match in order)
        correct = 0
        total = min(len(response_numbers), len(ground_truth_numbers))

        for i in range(total):
            if abs(response_numbers[i] - ground_truth_numbers[i]) <= self.tolerance:
                correct += 1

        accuracy = correct / len(ground_truth_numbers) if ground_truth_numbers else 0.0

        return GraderScore(
            name=self.name,
            score=accuracy,
            reason=f"Number accuracy: {correct}/{len(ground_truth_numbers)} numbers correct",
            metadata={
                "accuracy": accuracy,
                "correct_numbers": correct,
                "total_ground_truth_numbers": len(ground_truth_numbers),
                "response_numbers": response_numbers,
                "ground_truth_numbers": ground_truth_numbers,
                "tolerance": self.tolerance,
            },
        )
