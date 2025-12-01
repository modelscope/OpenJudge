import ast
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore


class SyntaxCheckGrader(BaseGrader):
    """Check code syntax using Abstract Syntax Tree to validate Python code blocks.

    This grader evaluates Python code for syntax correctness by parsing code blocks
    using Python's Abstract Syntax Tree (AST) parser. It extracts code blocks from
    markdown-style code fences and checks each one for syntax errors.
    """

    def __init__(self):
        super().__init__(
            name="syntax_check",
            mode=GraderMode.POINTWISE,
            description="Check code syntax using Abstract Syntax Tree to validate Python code blocks.",
        )

    async def aevaluate(self, answer: str) -> GraderScore:
        """Check code syntax in the provided answer.

        Extracts Python code blocks from markdown-style code fences and validates
        their syntax using Python's AST parser. Returns a score based on the ratio
        of syntactically correct code blocks to total code blocks found.

        Args:
            answer (str): The answer containing code blocks to check for syntax errors.
                        Code blocks should be enclosed in markdown-style fences (```python).

        Returns:
            GraderScore: The syntax check result.
                - score (float): Ratio of valid code blocks (0.0-1.0), with a penalty
                               applied for syntax errors
                - reason (str): Explanation of the syntax check result
                - metadata (Dict[str, Any]): Additional information including:
                    * code_blocks: List of extracted code blocks
                    * valid_blocks: Number of syntactically valid blocks
                    * total_blocks: Total number of code blocks found
                    * syntax_errors: List of syntax errors encountered

        Example:
            >>> grader = SyntaxCheckGrader()
            >>> result = await grader.aevaluate("Here's a function:\\n```python\\ndef hello():\\n    print('Hello')\\n```")
            >>> print(result.score, result.reason)
            1.0 Syntax check: 1/1 blocks valid, 0 errors

            >>> # Example with syntax error
            >>> result = await grader.aevaluate("Here's a function with error:\\n```python\\ndef hello(\\n    print('Hello')\\n```")
            >>> print(result.score, result.reason)
            -0.5 Syntax check: 0/1 blocks valid, 1 errors
        """

        # Extract code blocks
        code_pattern = r"```(?:python)?\n(.*?)\n```"
        code_blocks = re.findall(code_pattern, answer, re.DOTALL)

        if not code_blocks:
            # No code blocks, return neutral score
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No code blocks found to check",
                metadata={"code_blocks": [], "syntax_errors": []},
            )

        syntax_errors = []
        valid_blocks = 0

        for i, code in enumerate(code_blocks):
            try:
                ast.parse(code.strip())
                valid_blocks += 1
            except SyntaxError as e:
                syntax_errors.append(
                    {
                        "block": i,
                        "error": str(e),
                        "line": e.lineno,
                        "offset": e.offset,
                    },
                )

        # Calculate score: ratio of valid code blocks
        score = valid_blocks / len(code_blocks) if code_blocks else 0.0

        # Apply penalty if syntax errors exist
        if syntax_errors:
            score -= 0.5

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Syntax check: {valid_blocks}/{len(code_blocks)} blocks valid, {len(syntax_errors)} errors",
            metadata={
                "code_blocks": code_blocks,
                "valid_blocks": valid_blocks,
                "total_blocks": len(code_blocks),
                "syntax_errors": syntax_errors,
            },
        )
