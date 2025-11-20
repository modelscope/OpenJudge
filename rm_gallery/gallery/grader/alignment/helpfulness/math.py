# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Mathematical Accuracy:\n    "
    "Ensure all calculations, formula applications, and logical steps are "
    "error-free, as even minor inaccuracies (e.g., arithmetic mistakes, "
    "misapplied rules) invalidate results despite otherwise correct "
    "methodologies.\n"
    "Logical Coherence:\n    "
    "Present solutions with clear, sequential reasoning where each step follows "
    "logically from the previous one, enabling easy verification and "
    "understanding of the problem-solving process.\n"
    "Notational Clarity:\n    "
    "Use appropriate mathematical notation, symbols, and formatting consistently "
    "throughout the solution to avoid ambiguity and enhance readability.\n"
    "Problem-Specific Precision:\n    "
    "Address all aspects of the given problem completely, providing exact "
    "solutions in the required format without omitting critical details or "
    "providing extraneous information."
)


# Math Score System Prompt
MATH_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Math Score User Prompt
MATH_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the mathematical correctness of a response.
You should assess the response based on mathematical accuracy, logical coherence, and notational clarity.
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
    "score": "A numerical score from 0.0 to 1.0 representing the mathematical correctness of the response."
    "reason": "The reason for the score."
}
```
"""

# Math Rank System Prompt
MATH_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Math Rank User Prompt
MATH_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. \
These rubrics are ordered from highest to lowest importance. You must check each \
candidate answer in turn to see if it violates any rubric, and provide reasons for \
any violations you find. These reasons should be used as references for ranking \
the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.

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

MATH_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=MATH_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=MATH_POINTWISE_USER_PROMPT,
        ),
    ],
)

MATH_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=MATH_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=MATH_LISTWISE_USER_PROMPT,
        ),
    ],
)


class MathGrader(BaseHelpfulnessGrader):
    """Math: Solves mathematical problems with accuracy, logical coherence, and proper notation."""

    _point_template = MATH_POINTWISE_TEMPLATE
    _list_template = MATH_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the MathGrader.

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
            name="math",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Solves mathematical problems with accuracy, logical coherence, and proper notation.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the mathematical correctness of the response based on the query.

        Evaluates math responses for accuracy, logical coherence, and notational clarity.
        The grader focuses on error-free calculations, clear reasoning steps, and
        appropriate mathematical notation.

        Args:
            query (str): The mathematical problem or query to evaluate.
            answer (str | List[str]): The mathematical solution(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical mathematical correctness score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of solutions by mathematical correctness
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise math grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = MathGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Solve for x: 2x + 5 = 15",
            ...     answer="2x + 5 = 15\\n2x = 15 - 5\\n2x = 10\\nx = 5"
            ... ))
            >>> print(result.score, result.reason)
            1.0 The solution correctly solves for x with clear step-by-step reasoning.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
