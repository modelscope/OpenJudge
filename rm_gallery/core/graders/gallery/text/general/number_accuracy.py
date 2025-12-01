import re
from typing import List
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class NumberAccuracyGrader(BaseGrader):
    """
    Check numerical calculation accuracy by comparing numbers in generated vs reference content.

    This reward verifies if the numbers in the generated content match
    the numbers in the reference content within a specified tolerance.

    Methods:
        evaluate: Extracts and compares numbers between generated and reference content.

    Examples:
        >>> grader = NumberAccuracyGrader(tolerance=1e-6)
        >>> result = await grader.aevaluate(
        ...     generated="The result is 3.14159",
        ...     reference="The result is 3.14159"
        ... )
        >>> print(result.score)
        1.0
        >>> result = await grader.aevaluate(
        ...     generated="The result is 3.14",
        ...     reference="The result is 3.14159"
        ... )
        >>> print(result.score)
        0.0
    """

    def __init__(self, tolerance: float = 1e-6, **kwargs: Any) -> None:
        """"""
        super().__init__(
            name="number_accuracy",
            mode=GraderMode.POINTWISE,
            description="Check numerical calculation accuracy by comparing numbers in generated vs reference content",
            **kwargs,
        )
        self.tolerance = tolerance

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        # Match integers and floating point numbers
        number_pattern = r"-?\d+\.?\d*"
        numbers = re.findall(number_pattern, text)
        return [float(n) for n in numbers if n]

    async def aevaluate(self, generated: str, reference: str) -> GraderScore:
        """
        Calculate number accuracy by comparing extracted numbers from both texts.

        This method extracts numerical values from both the generated and reference texts,
        then compares them in order to compute an accuracy score. The score represents the
        proportion of numbers in the reference that were correctly reproduced in the generated text.
        Numbers are compared with a tolerance threshold to account for floating-point precision issues.

        Args:
            generated (str): Generated content to evaluate. This is typically the output
                from a language model that we want to assess for numerical accuracy.
            reference (str): Reference answer containing expected numbers. The numbers
                in this text are considered the ground truth.

        Returns:
            GraderScore: Result containing the number accuracy score and explanation.
                - score (float): Proportion of correctly matched numbers (between 0.0 and 1.0)
                - reason (str): Explanation of the scoring result including count of correct numbers
                - metadata (Dict): Contains details about extracted numbers and comparison including:
                    - accuracy (float): Computed accuracy score
                    - correct_numbers (int): Count of correctly matched numbers
                    - total_reference_numbers (int): Total count of numbers in reference text
                    - generated_numbers (List[float]): Numbers extracted from generated text
                    - reference_numbers (List[float]): Numbers extracted from reference text
                    - tolerance (float): Tolerance used for number comparison

        Examples:
            >>> grader = NumberAccuracyGrader(tolerance=1e-6)
            >>> result = await grader.aevaluate(
            ...     generated="The result is 3.14159",
            ...     reference="The result is 3.14159"
            ... )
            >>> print(result.score)
            1.0
            >>> print(result.reason)
            Number accuracy: 1/1 numbers correct
            >>> result = await grader.aevaluate(
            ...     generated="The temperatures are 25.5 and 30.2 degrees",
            ...     reference="The temperatures are 25.5 and 30.0 degrees"
            ... )
            >>> print(result.score)
            0.5
            >>> print(result.reason)
            Number accuracy: 1/2 numbers correct
        """
        generated_numbers = self._extract_numbers(generated)
        reference_numbers = self._extract_numbers(reference)

        if not reference_numbers:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No reference numbers to compare",
                metadata={
                    "generated_numbers": generated_numbers,
                    "reference_numbers": reference_numbers,
                },
            )
        if not generated_numbers:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No numbers found in generated content",
                metadata={
                    "generated_numbers": generated_numbers,
                    "reference_numbers": reference_numbers,
                },
            )

        # Compare numbers (match in order)
        correct = 0
        total = min(len(generated_numbers), len(reference_numbers))

        for i in range(total):
            if abs(generated_numbers[i] - reference_numbers[i]) <= self.tolerance:
                correct += 1

        accuracy = correct / len(reference_numbers) if reference_numbers else 0.0

        return GraderScore(
            name=self.name,
            score=accuracy,
            reason=f"Number accuracy: {correct}/{len(reference_numbers)} numbers correct",
            metadata={
                "accuracy": accuracy,
                "correct_numbers": correct,
                "total_reference_numbers": len(reference_numbers),
                "generated_numbers": generated_numbers,
                "reference_numbers": reference_numbers,
                "tolerance": self.tolerance,
            },
        )