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

RUBRICS = """Strict Adherence to Explicit Formatting and Structural Requirements: Prioritize exact compliance with all specified formatting, structural, and technical constraints (e.g., punctuation, indentation, bullet points, word counts) as the primary criterion for evaluating completions.
Clarity, Logical Progression, and Thematic Consistency: Ensure content is coherent, logically structured, and maintains alignment with the scenario's core premise, fulfilling implicit demands for depth, relevance, and narrative or analytical consistency.
Conditional Accuracy and Contextual Appropriateness: Verify that responses correctly interpret and apply conditional logic, providing accurate information that is appropriate for the given context and conditions.
Explicit Condition Matching: Confirm that responses directly address specified conditions and constraints, without omitting critical elements or introducing irrelevant information.
"""


PRECISE_IF_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
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
""",
        ),
    ],
)

PRECISE_IF_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
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
""",
        ),
    ],
)


class PreciseIfGrader(BaseHelpfulnessGrader):
    """Precise-If: Delivers accurate, condition-specific responses that fully satisfy contextual requirements."""

    _point_template = PRECISE_IF_SCORE_TEMPLATE
    _list_template = PRECISE_IF_RANK_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="Precise-If",
            mode=mode,
            model=model,
            template=template,
            description="Delivers accurate, condition-specific responses that fully satisfy contextual requirements.",
            **kwargs,
        )
        
    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the precision of conditional responses based on the query.

        Evaluates precise-if responses for their ability to deliver accurate,
        condition-specific responses that fully satisfy contextual requirements.
        The grader focuses on conditional accuracy, explicit condition matching,
        and contextual appropriateness.

        Args:
            query (str): The query to evaluate.
            answer (str | List[str]): The answer(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Example:
            >>> grader = PreciseIfGrader()
            >>> result = await grader.evaluate(
            ...     query="If the temperature is above 30°C, suggest cooling methods",
            ...     answer="Since the temperature is 35°C, I suggest using a fan or air conditioning."
            ... )
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
