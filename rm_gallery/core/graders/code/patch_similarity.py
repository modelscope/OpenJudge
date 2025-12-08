"""Patch Similarity Evaluator Module.

This module provides functionality to calculate similarity between response
code patches and ground truth (oracle) patches using difflib.SequenceMatcher.
It includes the PatchSimilarityGrader class which evaluates the quality of
code modifications by comparing them with ground truth implementations.

The module is designed for use in code generation evaluation scenarios where
the similarity between a response patch and a correct ground truth patch needs
to be quantified.
"""

import difflib

from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class PatchSimilarityGrader(BaseGrader):
    """
    Calculate similarity between response patch and oracle patch using difflib.SequenceMatcher.

    This reward measures how similar the response patch is to the ground truth patch,
    providing a similarity score and detailed diff information.
    """

    def __init__(self):
        super().__init__(
            name="patch_similarity",
            mode=GraderMode.POINTWISE,
            description="Calculate similarity between response patch and oracle patch using difflib.SequenceMatcher",
        )

    async def aevaluate(self, response: str, ground_truth: str) -> GraderScore:
        """Calculate similarity between response and ground truth patches.

        Uses difflib.SequenceMatcher to calculate the similarity ratio between
        a response patch and a ground truth (oracle) patch. This is useful for
        evaluating the quality of code modifications or response diffs.

        Args:
            response (str): The response patch or code modification to evaluate.
            ground_truth (str): The ground truth (oracle) patch that is considered correct.

        Returns:
            GraderScore: The patch similarity evaluation result.
                - score (float): Similarity score between response and ground truth patches (0.0-1.0)
                - reason (str): Explanation of the similarity score
                - metadata (Dict[str, Any]): Additional information including:
                    * similarity: The calculated similarity ratio
                    * response: The response patch
                    * ground truth: The ground truth patch
                    * opcodes: Detailed diff operations from SequenceMatcher

        Example:
            >>> grader = PatchSimilarityGrader()
            >>> response = "def add(a, b):\\n    return a + b"
            >>> ground_truth = "def add(x, y):\\n    return x + y"
            >>> result = await grader.aevaluate(response, ground_truth)
            >>> print(result.score, result.reason)
            0.8 Patch similarity: 0.800 based on sequence matching
        """

        # Use SequenceMatcher to calculate similarity
        matcher = difflib.SequenceMatcher(None, response, ground_truth)
        similarity = matcher.ratio()

        # Get detailed diff information
        opcodes = list(matcher.get_opcodes())

        return GraderScore(
            name=self.name,
            score=similarity,
            reason=f"Patch similarity: {similarity:.3f} based on sequence matching",
            metadata={
                "similarity": similarity,
                "response": response,
                "ground truth": ground_truth,
                "opcodes": opcodes,
            },
        )
