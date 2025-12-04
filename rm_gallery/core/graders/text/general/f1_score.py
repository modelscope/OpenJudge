"""
F1 Score Grader Module.

This module implements a grader that calculates the F1 score between response
content and reference answers at word level. The F1 score is a measure of a
test's accuracy that considers both precision and recall.

The F1ScoreGrader class provides configurable tokenization options to support
multilingual content including Chinese and English, allowing flexible text
comparison for various use cases.
"""

from typing import Any, Literal

from rm_gallery.core.graders.base_grader import BaseGrader, GraderScore
from rm_gallery.core.graders.schema import GraderMode
from rm_gallery.core.utils.tokenizer import get_tokenizer


class F1ScoreGrader(BaseGrader):
    """
    Calculate F1 score between response content and reference answer at word level.

    This reward computes precision, recall and F1 score by comparing word overlap
    between response and reference texts. Uses configurable tokenizer to support
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
            description="Calculate F1 score between response content and reference answer at word level",
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

    async def aevaluate(self, response: str, reference: str) -> GraderScore:
        """
        Calculate F1 score between response content and reference answer at word level.

        Args:
            response: Generated content to evaluate
            reference: Reference answer to compare against

        Returns:
            GraderScore: Result containing the F1 score and explanation.
                - score (float): F1 score between 0.0 and 1.0
                - reason (str): Explanation of the scoring result including precision and recall
                - metadata (Dict): Additional computation details including:
                    - f1_score (float): Computed F1 score
                    - precision (float): Precision value
                    - recall (float): Recall value
                    - response_tokens (List[str]): Tokenized response content
                    - reference_tokens (List[str]): Tokenized reference content
                    - tokenizer_type (str): Type of tokenizer used
                    - tokenizer_name (str): Name of tokenizer used

        Examples:
            >>> grader = F1ScoreGrader()
            >>> result = await grader.aevaluate(
            ...     response="The quick brown fox",
            ...     reference="The quick brown fox jumps over the lazy dog"
            ... )
            >>> print(round(result.score, 3))
            0.571
            >>> result = await grader.aevaluate(
            ...     response="Hello world",
            ...     reference="Hello world"
            ... )
            >>> print(result.score)
            1.0
        """

        # Tokenize using unified tokenizer
        response_preprocessed = self._tokenizer.preprocess_text(
            response,
            to_lower=True,
        )
        reference_preprocessed = self._tokenizer.preprocess_text(
            reference,
            to_lower=True,
        )

        response_tokens = set(
            self._tokenizer.tokenize(response_preprocessed),
        )
        reference_tokens = set(
            self._tokenizer.tokenize(reference_preprocessed),
        )

        # Calculate precision, recall and F1 score
        if not response_tokens and not reference_tokens:
            precision = recall = f1 = 1.0
        elif not response_tokens or not reference_tokens:
            precision = recall = f1 = 0.0
        else:
            intersection = response_tokens.intersection(reference_tokens)
            precision = len(intersection) / len(response_tokens)
            recall = len(intersection) / len(reference_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return GraderScore(
            name=self.name,
            score=f1,
            reason=f"F1 score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
            metadata={
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "response_tokens": list(response_tokens),
                "reference_tokens": list(reference_tokens),
                "tokenizer_type": self.tokenizer_type,
                "tokenizer_name": self._tokenizer.name,
            },
        )
