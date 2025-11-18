# -*- coding: utf-8 -*-
from typing import List
from rm_gallery.core.grader.base import (
    GraderMode,
    LLMGrader,
    GraderScore,
    GraderRank,
)
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import (
    BaseHelpfulnessGrader,
)


RUBRICS = """Functional Correctness: The code must correctly implement the required functionality without syntax errors or logical flaws. It should pass basic test cases related to the problem.
Code Quality and Readability: The code should be well-structured, readable, and follow common programming conventions. Variable names should be meaningful, and complex logic should be appropriately commented.
Efficiency and Optimization: The solution should be reasonably efficient in terms of time and space complexity. Avoid unnecessary computations or memory usage.
Language-Specific Best Practices: The code should follow language-specific conventions and idioms, making it appear as if written by an experienced developer in that language.
Error Handling: Appropriate error handling should be implemented for edge cases and invalid inputs, making the code robust and production-ready.
"""


CODE_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of a code response.
You should assess the code based on functional correctness, readability, efficiency, and best practices.
Be as objective as possible.

# Rubrics
{rubrics}

# Query
{query}

# Answer
{answer}

# Output Requirement
```json
{
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the code."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

CODE_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.


# Rubrics
{rubrics}

# Query
{query}

# Answers
{answer}

# Output Requirement
```json
{
    "rank": ["The rank score of the answer in the list."]
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)


class CodeGrader(BaseHelpfulnessGrader):
    """Code: Generates functional, efficient, and readable code that solves programming problems."""

    _point_template = CODE_SCORE_TEMPLATE
    _list_template = CODE_RANK_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="Code",
            mode=mode,
            model=model,
            template=template,
            description="Generates functional, efficient, and readable code that solves programming problems.",
            **kwargs,
        )

    
    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the code response based on the query.

        Evaluates code responses for their ability to generate functional,
        efficient, and readable code that solves programming problems. The
        grader focuses on functional correctness, code quality and readability,
        and efficiency and optimization.

        Args:
            query (str): The programming task or question to be solved.
            answer (str | List[str]): The code response(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical helpfulness score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of code responses by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise code grader
            >>> grader = CodeGrader(mode=GraderMode.POINTWISE)
            >>> result = await grader.evaluate(
            ...     query="Write a function to calculate factorial of a number",
            ...     answer="def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)"
            ... )
            >>> print(result.score, result.reason)

            >>> # Example for listwise code grader
            >>> ranking_grader = CodeGrader(mode=GraderMode.LISTWISE)
            >>> result = await ranking_grader.evaluate(
            ...     query="Write a function to reverse a string",
            ...     answer=["def reverse(s):\\n    return s[::-1]", "def reverse(s):\\n    result = ''\\n    for char in s:\\n        result = char + result\\n    return result"]
            ... )
            >>> print(result.rank, result.reason)
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
