"""
ROUGE-L Grader Module.

This module implements a grader that calculates the ROUGE-L score between generated
content and reference answers. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
measures the longest common subsequence between texts, accounting for sentence-level
structure and word order.

The RougeLGrader class provides methods for evaluating text generation quality
using the ROUGE-L metric which balances precision and recall.
"""

from typing import List
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class RougeLGrader(BaseGrader):
    """
    Calculate ROUGE-L score between generated content and reference answer.

    ROUGE-L measures the longest common subsequence (LCS) between the generated and
    reference texts, providing a balance between precision and recall. Unlike simpler
    n-gram based ROUGE metrics, ROUGE-L accounts for sentence-level structure and word order.

    Methods:
        evaluate: Computes ROUGE-L score between generated and reference texts.

    Examples:
        >>> grader = RougeLGrader()
        >>> result = await grader.aevaluate(
        ...     generated="the quick brown fox jumps over the lazy dog",
        ...     reference="the fast brown fox jumped over the sleepy dog"
        ... )
        >>> print(round(result.score, 3))
        0.727
    """

    def __init__(self) -> None:
        super().__init__(
            name="rouge_l",
            mode=GraderMode.POINTWISE,
            description="Calculate ROUGE-L score between generated content and reference answer",
        )

    async def aevaluate(self, generated: str, reference: str) -> GraderScore:
        """
        Calculate ROUGE-L score between generated content and reference answer.

        This method computes the ROUGE-L score which evaluates the quality of generated text
        by measuring the longest common subsequence (LCS) between the generated and reference texts.
        It provides a balance between precision (how much of the generated text is relevant) and
        recall (how much of the reference text is captured).

        Args:
            generated (str): Generated content to evaluate. This is typically the output
                from a language model that we want to assess.
            reference (str): Reference answer to compare against. This is the ground truth
                or expected correct answer.

        Returns:
            GraderScore: Result containing the ROUGE-L score and explanation.
                - score (float): ROUGE-L score between 0.0 and 1.0, where 1.0 indicates perfect overlap
                - reason (str): Explanation of the scoring result including the calculated ROUGE-L score
                - metadata (Dict): Additional computation details including:
                    - rouge_l (float): Computed ROUGE-L score
                    - generated_length (int): Number of tokens in the generated text
                    - reference_length (int): Number of tokens in the reference text
                    - lcs_length (int): Length of the longest common subsequence

        Examples:
            >>> grader = RougeLGrader()
            >>> result = await grader.aevaluate(
            ...     generated="the quick brown fox jumps over the lazy dog",
            ...     reference="the fast brown fox jumped over the sleepy dog"
            ... )
            >>> print(round(result.score, 3))
            0.727
            >>> print(result.reason)
            ROUGE-L score: 0.727
            >>> result = await grader.aevaluate(
            ...     generated="completely different text",
            ...     reference="the fast brown fox jumped over the sleepy dog"
            ... )
            >>> print(round(result.score, 3))
            0.0
        """

        def _lcs_length(x: List[str], y: List[str]) -> int:
            """Calculate longest common subsequence length"""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        # Tokenization
        generated_tokens = generated.split()
        reference_tokens = reference.split()

        if not generated_tokens and not reference_tokens:
            rouge_l = 1.0
        elif not generated_tokens or not reference_tokens:
            rouge_l = 0.0
        else:
            # Calculate LCS length
            lcs_len = _lcs_length(generated_tokens, reference_tokens)

            # Calculate ROUGE-L
            if len(generated_tokens) == 0 or len(reference_tokens) == 0:
                rouge_l = 0.0
            else:
                precision = lcs_len / len(generated_tokens)
                recall = lcs_len / len(reference_tokens)
                rouge_l = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return GraderScore(
            name=self.name,
            score=rouge_l,
            reason=f"ROUGE-L score: {rouge_l:.3f}",
            metadata={
                "rouge_l": rouge_l,
                "generated_length": len(generated_tokens),
                "reference_length": len(reference_tokens),
                "lcs_length": (
                    _lcs_length(generated_tokens, reference_tokens) if generated_tokens and reference_tokens else 0
                ),
            },
        )
