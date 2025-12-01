import difflib
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class PatchSimilarityGrader(BaseGrader):
    """
    Calculate similarity between generated patch and oracle patch using difflib.SequenceMatcher.

    This reward measures how similar the generated patch is to the reference patch,
    providing a similarity score and detailed diff information.
    """

    def __init__(self):
        super().__init__(
            name="patch_similarity",
            mode=GraderMode.POINTWISE,
            description="Calculate similarity between generated patch and oracle patch using difflib.SequenceMatcher",
        )

    async def aevaluate(self, generated: str, reference: str) -> GraderScore:
        """Calculate similarity between generated and reference patches.

        Uses difflib.SequenceMatcher to calculate the similarity ratio between
        a generated patch and a reference (oracle) patch. This is useful for
        evaluating the quality of code modifications or generated diffs.

        Args:
            generated (str): The generated patch or code modification to evaluate.
            reference (str): The reference (oracle) patch that is considered correct.

        Returns:
            GraderScore: The patch similarity evaluation result.
                - score (float): Similarity score between generated and reference patches (0.0-1.0)
                - reason (str): Explanation of the similarity score
                - metadata (Dict[str, Any]): Additional information including:
                    * similarity: The calculated similarity ratio
                    * generated: The generated patch
                    * reference: The reference patch
                    * opcodes: Detailed diff operations from SequenceMatcher

        Example:
            >>> grader = PatchSimilarityGrader()
            >>> generated = "def add(a, b):\\n    return a + b"
            >>> reference = "def add(x, y):\\n    return x + y"
            >>> result = await grader.aevaluate(generated, reference)
            >>> print(result.score, result.reason)
            0.8 Patch similarity: 0.800 based on sequence matching
        """

        # Use SequenceMatcher to calculate similarity
        matcher = difflib.SequenceMatcher(None, generated, reference)
        similarity = matcher.ratio()

        # Get detailed diff information
        opcodes = list(matcher.get_opcodes())

        return GraderScore(
            name=self.name,
            score=similarity,
            reason=f"Patch similarity: {similarity:.3f} based on sequence matching",
            metadata={
                "similarity": similarity,
                "generated": generated,
                "reference": reference,
                "opcodes": opcodes,
            },
        )

