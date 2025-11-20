# -*- coding: utf-8 -*-
import ast
import difflib
import json
import re
import traceback
from typing import Any

from rm_gallery.core.grader.base import Grader, GraderMode, GraderScore


class SyntaxCheckGrader(Grader):
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


class CodeStyleGrader(Grader):
    """Basic code style checking including indentation consistency and naming conventions."""

    def __init__(self):
        super().__init__(
            name="code_style",
            mode=GraderMode.POINTWISE,
            description="Basic code style checking including indentation consistency and naming conventions.",
        )

    def _check_indentation(self, code: str) -> tuple[bool, str]:
        """Check indentation consistency"""
        lines = code.split("\n")
        indent_type = None  # 'spaces' or 'tabs'
        indent_size = None

        for line in lines:
            if line.strip():  # Non-empty line
                leading = len(line) - len(line.lstrip())
                if leading > 0:
                    if line.startswith(" "):
                        if indent_type is None:
                            indent_type = "spaces"
                            indent_size = leading
                        elif indent_type != "spaces":
                            return (
                                False,
                                "Mixed indentation types (spaces and tabs)",
                            )
                    elif line.startswith("\t"):
                        if indent_type is None:
                            indent_type = "tabs"
                        elif indent_type != "tabs":
                            return (
                                False,
                                "Mixed indentation types (spaces and tabs)",
                            )

        return True, "Consistent indentation"

    def _check_naming(self, code: str) -> tuple[float, str]:
        """Check naming conventions"""
        # Simple naming check
        function_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        variable_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*="

        functions = re.findall(function_pattern, code)
        variables = re.findall(variable_pattern, code)

        total_names = len(functions) + len(variables)
        if total_names == 0:
            return 1.0, "No names to check"

        good_names = 0

        # Check function names (should be snake_case)
        for func in functions:
            if re.match(r"^[a-z_][a-z0-9_]*$", func):
                good_names += 1

        # Check variable names (should be snake_case)
        for var in variables:
            if re.match(r"^[a-z_][a-z0-9_]*$", var):
                good_names += 1

        score = good_names / total_names
        return (
            score,
            f"Naming convention: {good_names}/{total_names} names follow snake_case",
        )

    async def aevaluate(self, answer: str) -> GraderScore:
        """Evaluate code style in the provided answer.

        Performs basic code style checking including indentation consistency and
        naming conventions. Extracts Python code blocks from markdown-style code
        fences and evaluates each one for style compliance.

        Args:
            answer (str): The answer containing code blocks to check for style issues.
                        Code blocks should be enclosed in markdown-style fences (```python).

        Returns:
            GraderScore: The code style evaluation result.
                - score (float): Average style score across all code blocks (0.0-1.0)
                - reason (str): Explanation of the style check result with details
                - metadata (Dict[str, Any]): Additional information including:
                    * code_blocks: List of extracted code blocks
                    * average_score: Average style score
                    * details: Detailed breakdown of style checks

        Example:
            >>> grader = CodeStyleGrader()
            >>> result = await grader.aevaluate("Here's a function:\\n```python\\ndef calculate_sum(a, b):\\n    return a + b\\n```")
            >>> print(result.score, result.reason)
            1.0 Code style score: 1.000; Consistent indentation; Naming convention: 2/2 names follow snake_case

            >>> # Example with style issues
            >>> result = await grader.aevaluate("Here's a function with style issues:\\n```python\\ndef CalculateSum(a,b):\\n	return a+b\\n```")
            >>> print(result.score, result.reason)
            0.5 Code style score: 0.500; Consistent indentation; Naming convention: 1/2 names follow snake_case
        """
        # Extract code blocks
        code_pattern = r"```(?:python)?\n(.*?)\n```"
        code_blocks = re.findall(code_pattern, answer, re.DOTALL)

        if not code_blocks:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No code blocks found to check style",
                metadata={"code_blocks": []},
            )

        total_score = 0.0
        details = []

        for i, code in enumerate(code_blocks):
            block_score = 0.0

            # Check indentation
            indent_ok, indent_msg = self._check_indentation(code)
            if indent_ok:
                block_score += 0.5
            details.append(f"Block {i}: {indent_msg}")

            # Check naming
            naming_score, naming_msg = self._check_naming(code)
            block_score += naming_score * 0.5
            details.append(f"Block {i}: {naming_msg}")

            total_score += block_score

        # Average score
        average_score = total_score / len(code_blocks)
        return GraderScore(
            name=self.name,
            score=average_score,
            reason=f"Code style score: {average_score:.3f}; " + "; ".join(details),
            metadata={
                "code_blocks": code_blocks,
                "average_score": average_score,
                "details": details,
            },
        )


class PatchSimilarityGrader(Grader):
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


class CodeExecutionGrader(Grader):
    """
    Executes code against test cases and evaluates correctness based on test case results.

    This reward model evaluates code by executing it against test cases using a testing framework
    that supports both call-based and standard input code evaluation methods.
    """

    def __init__(
        self,
        continuous: bool = True,
        timeout: int = 10,
        test_framework_available: bool = True,
        compute_score: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            name="code_execution",
            mode=GraderMode.POINTWISE,
            description="Executes code against test cases and evaluates correctness based on test case results",
            **kwargs,
        )

        self.continuous = continuous
        self.timeout = timeout
        self.test_framework_available = test_framework_available
        self.compute_score = compute_score

        try:
            from rm_gallery.gallery.grader.code.prime_code import compute_score

            self.compute_score = compute_score
            self.test_framework_available = True
        except ImportError:
            print(
                "Warning: Code testing framework not available. Please ensure rm_gallery.gallery.rm.code.prime_code is properly installed.",
            )
            self.test_framework_available = False

    def _extract_code(self, content: str) -> str:
        """
        Extract code from content

        Args:
            content: Text content that may contain code blocks

        Returns:
            Extracted code
        """
        # Try to find Python code in various formats
        code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try other formats
        code_match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # If no code block markers, assume the entire content is code
        return content

    async def aevaluate(self, answer: str) -> GraderScore:
        """Evaluate code by executing it against test cases.

        Tests the functional correctness of generated code by executing it
        with predefined test cases. This grader checks if the code produces
        the expected outputs for given inputs, verifying its correctness.

        Args:
            answer (str): The code to evaluate, potentially containing
                         markdown-style code blocks.

        Returns:
            GraderScore: The code execution evaluation result.
                - score (float): Execution score (0.0-1.0) based on test case results
                - reason (str): Explanation of the execution results
                - metadata (Dict[str, Any]): Additional information including:
                    * extracted_code: The code extracted from the answer
                    * execution_results: Detailed results of code execution
                    * passed_tests: Number of passed test cases
                    * total_tests: Total number of test cases

        Example:
            >>> grader = CodeExecutionGrader()
            >>> code_answer = "def add(a, b):\\n    return a + b"
            >>> result = await grader.aevaluate(code_answer)
            >>> print(result.score, result.reason)
            1.0 Code executed successfully and passed all tests

            >>> # Example with failing code
            >>> bad_code = "def add(a, b):\\n    return a - b"
            >>> result = await grader.aevaluate(bad_code)
            >>> print(result.score, result.reason)
            0.0 Code execution failed or did not pass tests
        """
        # Extract code from response
        content = answer
        extracted_code = self._extract_code(content)

        # Default values
        score = 0.0
        reason = "No evaluation performed"
        extra_data = {"extracted_code": extracted_code}

        # Check if testing framework is available
        if not self.test_framework_available:
            reason = "Code testing framework not available"
            extra_data["error"] = reason
        else:
            # Get test cases from sample metadata or label
            test_cases = None
            # if sample.metadata and "inputs_outputs" in sample.metadata:
            #     test_cases = sample.metadata["inputs_outputs"]
            # elif (
            #     sample.output[0].answer.label
            #     and "inputs_outputs" in sample.output[0].answer.label
            # ):
            #     test_cases = sample.output[0].answer.label["inputs_outputs"]

            if not test_cases:
                reason = "No test cases available for evaluation"
            elif not extracted_code:
                score = 0.0
                reason = "No valid code extracted from response"
                extra_data["test_cases"] = test_cases
            else:
                # Convert test cases to string if needed
                if isinstance(test_cases, dict):
                    test_cases_str = json.dumps(test_cases)
                else:
                    test_cases_str = test_cases

                # Evaluate code using testing framework
                try:
                    success, metadata = self.compute_score(
                        completion=extracted_code,
                        test_cases=test_cases_str,
                        continuous=self.continuous,
                    )

                    # Determine score based on success rate
                    if isinstance(success, bool):
                        pass_rate = 1.0 if success else 0.0
                    else:
                        pass_rate = float(success)

                    # Score is always between 0 and 1
                    score = pass_rate

                    # Generate reason based on results
                    if pass_rate == 1.0:
                        reason = "All test cases passed successfully"
                    elif pass_rate == 0.0:
                        reason = "No test cases passed"
                    else:
                        reason = f"Partial success: {pass_rate * 100:.1f}% of test cases passed"

                    # Include metadata in extra_data
                    extra_data = {
                        "extracted_code": extracted_code,
                        "test_cases": test_cases,
                        "pass_rate": pass_rate,
                    }

                except Exception as e:
                    error_traceback = traceback.format_exc()
                    score = 0.0
                    reason = f"Evaluation error: {str(e)}"
                    extra_data = {
                        "extracted_code": extracted_code,
                        "test_cases": test_cases,
                        "error": str(e),
                        "traceback": error_traceback,
                    }

        # Single return statement at the end of the function
        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=extra_data,
        )
