# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Functional Correctness:\n    "
    "The code must correctly implement the required functionality without syntax "
    "errors or logical flaws. It should pass basic test cases related to the "
    "problem.\n"
    "Code Quality and Readability:\n    "
    "The code should be well-structured, readable, and follow common programming "
    "conventions. Variable names should be meaningful, and complex logic should "
    "be appropriately commented.\n"
    "Efficiency and Optimization:\n    "
    "The solution should be reasonably efficient in terms of time and space "
    "complexity. Avoid unnecessary computations or memory usage.\n"
    "Language-Specific Best Practices:\n    "
    "The code should follow language-specific conventions and idioms, making it "
    "appear as if written by an experienced developer in that language.\n"
    "Error Handling:\n    "
    "Appropriate error handling should be implemented for edge cases and invalid "
    "inputs, making the code robust and production-ready."
)


# Code Score System Prompt
CODE_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Code Score User Prompt
CODE_POINTWISE_USER_PROMPT = """# Task Description
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
"""

# Code Rank System Prompt
CODE_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Code Rank User Prompt
CODE_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.


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
"""

CODE_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=CODE_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=CODE_POINTWISE_USER_PROMPT,
        ),
    ],
)

CODE_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=CODE_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=CODE_LISTWISE_USER_PROMPT,
        ),
    ],
)


class CodeGrader(BaseHelpfulnessGrader):
    """Code: Generates correct, efficient, and readable code solutions to programming problems."""

    _point_template = CODE_POINTWISE_TEMPLATE
    _list_template = CODE_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the CodeGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            mode: The grader mode. Defaults to LISTWISE.
            rubrics: Custom rubrics for evaluation. If None, default rubrics will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="code",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Generates correct, efficient, and readable code solutions to programming problems.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the code response based on the query.

        Evaluates code responses for their correctness, efficiency, readability,
        and adherence to best practices. The grader focuses on functional correctness,
        code quality, efficiency, and language-specific conventions.

        Args:
            query (str): The programming problem or query to evaluate.
            answer (str | List[str]): The code solution(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical code quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of code solutions by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise code grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = CodeGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Write a function to calculate the factorial of a number",
            ...     answer="def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)"
            ... ))
            >>> print(result.score, result.reason)
            0.9 The code correctly implements factorial calculation using recursion.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
