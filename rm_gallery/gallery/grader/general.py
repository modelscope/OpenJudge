# -*- coding: utf-8 -*-
"""
General Purpose Graders

This module provides general-purpose graders for common evaluation tasks including:
- AccuracyGrader: Exact match comparison between generated and reference text
- LengthGrader: Evaluates response length against a target
- KeywordPresenceGrader: Checks for presence of specific keywords
- RegexGrader: Validates response against regex patterns
"""

import re
from typing import Any, List, Literal

from rm_gallery.core.grader.base import Grader, GraderScore
from rm_gallery.core.schema.grader import GraderMode
from rm_gallery.core.utils.tokenizer import get_tokenizer


class AccuracyGrader(Grader):
    """
    Calculate accuracy between generated content and reference answer.

    This grader compares the generated content with the reference answer for exact match.
    Returns a score of 1.0 if they match exactly, 0.0 otherwise.

    Methods:
        evaluate: Compares generated content with reference for exact match.

    Examples:
        >>> grader = AccuracyGrader()
        >>> result = await grader.aevaluate(
        ...     generated="The capital of France is Paris",
        ...     reference="The capital of France is Paris"
        ... )
        >>> print(result.score)
        1.0
        >>> result = await grader.aevaluate(
        ...     generated="The capital of France is London",
        ...     reference="The capital of France is Paris"
        ... )
        >>> print(result.score)
        0.0
    """

    def __init__(self) -> None:
        super().__init__(
            name="accuracy",
            mode=GraderMode.POINTWISE,
            description="Calculate accuracy between generated content and reference answer",
        )

    async def aevaluate(self, generated: str, reference: str) -> GraderScore:
        """
        Calculate accuracy between generated content and reference answer.

        This method performs an exact string comparison between the generated content
        and the reference answer. It returns a score of 1.0 for an exact match and 0.0 otherwise.
        This is useful for evaluating tasks that require precise output matching such as
        factual questions with deterministic answers.

        Args:
            generated (str): Generated content to evaluate. This is typically the output
                from a language model that we want to assess.
            reference (str): Reference answer to compare against. This is the ground truth
                or expected correct answer.

        Returns:
            GraderScore: Result containing the accuracy score and explanation.
                - score (float): 1.0 if exact match, 0.0 otherwise
                - reason (str): Explanation of the scoring result including the calculated accuracy
                - metadata (Dict): Additional computation details including:
                    - generated (str): The generated content that was evaluated
                    - reference (str): The reference content for comparison

        Examples:
            >>> grader = AccuracyGrader()
            >>> result = await grader.aevaluate(
            ...     generated="The capital of France is Paris",
            ...     reference="The capital of France is Paris"
            ... )
            >>> print(result.score)
            1.0
            >>> print(result.reason)
            Accuracy: 1.000
            >>> result = await grader.aevaluate(
            ...     generated="The capital of France is London",
            ...     reference="The capital of France is Paris"
            ... )
            >>> print(result.score)
            0.0
            >>> print(result.reason)
            Accuracy: 0.000
        """
        accuracy = 1.0 if generated == reference else 0.0

        return GraderScore(
            name=self.name,
            score=accuracy,
            reason=f"Accuracy: {accuracy:.3f}",
            metadata={
                "generated": generated,
                "reference": reference,
            },
        )


class F1ScoreGrader(Grader):
    """
    Calculate F1 score between generated content and reference answer at word level.

    This reward computes precision, recall and F1 score by comparing word overlap
    between generated and reference texts. Uses configurable tokenizer to support
    multilingual content including Chinese and English.
    """

    def __init__(
        self,
        tokenizer_type: Literal["tiktoken", "jieba", "simple"] = "tiktoken",
        encoding_name: str = "cl100k_base",
        chinese_only: bool = False,
        **kwargs: Any,
    ):
        """ """
        super().__init__(
            name="f1_score",
            mode=GraderMode.POINTWISE,
            description="Calculate F1 score between generated content and reference answer at word level",
            **kwargs,
        )

        # Initialize tokenizer
        self.tokenizer_type = tokenizer_type
        self.encoding_name = encoding_name
        self.chinese_only = chinese_only
        self._tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            encoding_name=encoding_name,
            chinese_only=chinese_only,
        )

    async def aevaluate(self, generated: str, reference: str) -> GraderScore:
        """
        Calculate F1 score between generated content and reference answer at word level.

        Args:
            generated: Generated content to evaluate
            reference: Reference answer to compare against

        Returns:
            GraderScore: Result containing the F1 score and explanation.
                - score (float): F1 score between 0.0 and 1.0
                - reason (str): Explanation of the scoring result including precision and recall
                - metadata (Dict): Additional computation details including:
                    - f1_score (float): Computed F1 score
                    - precision (float): Precision value
                    - recall (float): Recall value
                    - generated_tokens (List[str]): Tokenized generated content
                    - reference_tokens (List[str]): Tokenized reference content
                    - tokenizer_type (str): Type of tokenizer used
                    - tokenizer_name (str): Name of tokenizer used

        Examples:
            >>> grader = F1ScoreGrader()
            >>> result = await grader.aevaluate(
            ...     generated="The quick brown fox",
            ...     reference="The quick brown fox jumps over the lazy dog"
            ... )
            >>> print(round(result.score, 3))
            0.571
            >>> result = await grader.aevaluate(
            ...     generated="Hello world",
            ...     reference="Hello world"
            ... )
            >>> print(result.score)
            1.0
        """

        # Tokenize using unified tokenizer
        generated_preprocessed = self._tokenizer.preprocess_text(
            generated,
            to_lower=True,
        )
        reference_preprocessed = self._tokenizer.preprocess_text(
            reference,
            to_lower=True,
        )

        generated_tokens = set(
            self._tokenizer.tokenize(generated_preprocessed),
        )
        reference_tokens = set(
            self._tokenizer.tokenize(reference_preprocessed),
        )

        # Calculate precision, recall and F1 score
        if not generated_tokens and not reference_tokens:
            precision = recall = f1 = 1.0
        elif not generated_tokens or not reference_tokens:
            precision = recall = f1 = 0.0
        else:
            intersection = generated_tokens.intersection(reference_tokens)
            precision = len(intersection) / len(generated_tokens)
            recall = len(intersection) / len(reference_tokens)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return GraderScore(
            name=self.name,
            score=f1,
            reason=f"F1 score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
            metadata={
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "generated_tokens": list(generated_tokens),
                "reference_tokens": list(reference_tokens),
                "tokenizer_type": self.tokenizer_type,
                "tokenizer_name": self._tokenizer.name,
            },
        )


class RougeLGrader(Grader):
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
                rouge_l = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

        return GraderScore(
            name=self.name,
            score=rouge_l,
            reason=f"ROUGE-L score: {rouge_l:.3f}",
            metadata={
                "rouge_l": rouge_l,
                "generated_length": len(generated_tokens),
                "reference_length": len(reference_tokens),
                "lcs_length": _lcs_length(generated_tokens, reference_tokens)
                if generated_tokens and reference_tokens
                else 0,
            },
        )


class NumberAccuracyGrader(Grader):
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
