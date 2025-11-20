# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Strict Adherence to Explicit Formatting and Structural Requirements:\n    "
    "Prioritize exact compliance with all specified formatting, structural, and "
    "technical constraints (e.g., punctuation, indentation, bullet points, word "
    "counts) as the primary criterion for evaluating completions.\n"
    "Clarity, Logical Progression, and Thematic Consistency:\n    "
    "Ensure content is coherent, logically structured, and maintains alignment "
    "with the scenario's core premise, fulfilling implicit demands for depth, "
    "relevance, and narrative or analytical consistency.\n"
    "Conditional Accuracy and Contextual Appropriateness:\n    "
    "Verify that responses correctly interpret and apply conditional logic, "
    "providing accurate information that is appropriate for the given context "
    "and conditions.\n"
    "Explicit Condition Matching:\n    "
    "Confirm that responses directly address specified conditions and constraints, "
    "without omitting critical elements or introducing irrelevant information."
)


# Precise If Score System Prompt
PRECISE_IF_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Precise If Score User Prompt
PRECISE_IF_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the precision of a conditional response.
You should assess the response based on strict adherence to formatting requirements, clarity, and conditional accuracy.
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
    "score": "A numerical score from 0.0 to 1.0 representing the precision of the conditional response."
    "reason": "The reason for the score."
}
```
"""

# Precise If Rank System Prompt
PRECISE_IF_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Precise If Rank User Prompt
PRECISE_IF_LISTWISE_USER_PROMPT = """# Task Description
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

PRECISE_IF_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=PRECISE_IF_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=PRECISE_IF_POINTWISE_USER_PROMPT,
        ),
    ],
)

PRECISE_IF_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=PRECISE_IF_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=PRECISE_IF_LISTWISE_USER_PROMPT,
        ),
    ],
)


class PreciseIfGrader(BaseHelpfulnessGrader):
    """Precise-IF: Provides concise, targeted responses that directly address conditional queries."""

    _point_template = PRECISE_IF_POINTWISE_TEMPLATE
    _list_template = PRECISE_IF_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the PreciseIfGrader.

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
            name="precise_if",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Provides concise, targeted responses that directly address conditional queries.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the precision of the IF response based on the query.

        Evaluates IF responses for their ability to provide concise, targeted responses
        that directly address conditional queries. The grader focuses on precision,
        relevance, and direct adherence to specified conditions.

        Args:
            query (str): The conditional query to evaluate.
            answer (str | List[str]): The response(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical precision score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of responses by precision
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise precise IF grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = PreciseIfGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="If the temperature is above 30°C, what should I wear?",
            ...     answer="If the temperature is above 30°C, wear light, breathable clothing such as shorts and a t-shirt."
            ... ))
            >>> print(result.score, result.reason)
            1.0 The response directly addresses the conditional query with specific, appropriate suggestions.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
