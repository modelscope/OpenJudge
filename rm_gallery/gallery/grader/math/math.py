# -*- coding: utf-8 -*-
"""
Math Verification Graders

This module provides graders for evaluating mathematical problem solving capabilities
using the math_verify library to parse and verify mathematical expressions.
"""

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from rm_gallery.core.grader.base import Grader, GraderMode, GraderScore


class MathVerifyGrader(Grader):
    """
    Verifies mathematical expressions using the math_verify library, supporting both LaTeX and plain expressions
    """

    def __init__(self, timeout_score: float = 1.0, **kwargs: Any):
        """
        Initialize the MathVerifyGrader.

        Args:
            timeout_score: Score to assign on timeout
        """
        super().__init__(
            name="math_verify",
            mode=GraderMode.POINTWISE,
            description="Verifies mathematical expressions using the math_verify library",
            **kwargs,
        )
        self.timeout_score = timeout_score

    async def aevaluate(self, generated: str, reference: str) -> GraderScore:
        """
        Verify mathematical expressions for accuracy by parsing and comparing the generated answer
        against a reference answer using the math_verify library.

        This grader supports both LaTeX and plain mathematical expressions. It parses both the
        generated and reference expressions, then verifies if they are mathematically equivalent.

        Args:
            generated (str): The generated mathematical expression to be verified.
            reference (str): The reference (ground truth) mathematical expression.

        Returns:
            GraderScore: A GraderScore object containing:
                - score (float): 1.0 if expressions are mathematically equivalent, 0.0 otherwise.
                               On error or timeout, uses the configured timeout_score (default 1.0).
                - reason (str): Explanation of the verification result.
                - metadata (Dict[str, Any]): Additional information including the generated and
                                          reference expressions.

        Examples:
            >>> import asyncio
            >>> grader = MathVerifyGrader()
            >>> result = asyncio.run(grader.aevaluate("2+2", "4"))
            >>> print(result.score)
            1.0

            >>> result = asyncio.run(grader.aevaluate("x^2", "x*x"))
            >>> print(result.score)
            1.0

            >>> result = asyncio.run(grader.aevaluate("3+4", "8"))
            >>> print(result.score)
            0.0
        """
        score = 0.0
        reason = "Verification failed or timed out"

        try:
            # Parse the reference (gold) answer
            # Use both LatexExtractionConfig and ExprExtractionConfig for maximum flexibility
            gold_parsed = parse(
                reference,
                extraction_config=[
                    LatexExtractionConfig(),
                    ExprExtractionConfig(),
                ],
            )

            # Parse the generated answer
            pred_parsed = parse(
                generated,
                extraction_config=[
                    LatexExtractionConfig(),
                    ExprExtractionConfig(),
                ],
            )

            # If both parsing succeeded and we have results
            if gold_parsed and pred_parsed:
                # Use the first parsed result from each
                gold_expr = gold_parsed[0]
                pred_expr = pred_parsed[0]

                # Verify if they match
                if verify(gold_expr, pred_expr):
                    score = 1.0
                    reason = f"({gold_parsed}, {pred_parsed})"
                else:
                    score = 0.0
                    reason = f"({gold_parsed}, {pred_parsed})"
            else:
                score = 0.0
                reason = f"Parsing failed - gold: {gold_parsed}, pred: {pred_parsed}"

        except Exception as e:
            score = self.timeout_score
            reason = f"Exception occurred: {str(e)}"

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata={
                "generated": generated,
                "reference": reference,
                "score": score,
            },
        )
