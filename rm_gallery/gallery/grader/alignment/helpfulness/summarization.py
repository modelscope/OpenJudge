# -*- coding: utf-8 -*-
from typing import List
from rm_gallery.core.grader.base import (
    GraderScore,
    GraderRank,
)
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import GraderMode
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import (
    BaseHelpfulnessGrader,
)

RUBRICS = """Comprehensive Coverage of Core Content: A superior summary captures all critical elements, themes, and details central to the source material without omitting key information.
Avoidance of Irrelevant or Tangential Information: Focuses exclusively on the primary subject, eliminating extraneous details that distract from the core narrative or argument.
Logical Structure and Coherence: Information is organized in a clear, hierarchical, or chronological sequence to ensure readability and logical progression of ideas.
Factual Accuracy and Neutral Objectivity: The summary must faithfully represent the source material without introducing distortions, opinions, or subjective interpretations, maintaining a neutral tone throughout.
"""

SUMMARIZATION_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of a summary.
You should assess the summary based on comprehensive coverage, avoidance of irrelevant information, logical structure, and factual accuracy.
Be as objective as possible.

# Rubrics
{rubrics}

# Original Text
{query}

# Summary
{answer}

# Output Requirement
```json
{
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the summary."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

SUMMARIZATION_RANK_TEMPLATE = Template(
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


class SummarizationGrader(BaseHelpfulnessGrader):
    """Summarization: Condenses information from longer texts into concise, accurate summaries."""

    _point_template = SUMMARIZATION_SCORE_TEMPLATE
    _list_template = SUMMARIZATION_RANK_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="Summarization",
            mode=mode,
            model=model,
            template=template,
            description="Summarization: Condenses information from longer texts into concise, accurate summaries.",
            **kwargs,
        )

    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the summarization quality of the response based on the query.

        Evaluates summarization responses for their ability to condense information
        from longer texts into concise, accurate summaries. The grader focuses on
        key information retention, conciseness, and neutral objectivity.

        Args:
            query (str): The original text to be summarized.
            answer (str | List[str]): The summary or summaries to evaluate. For POINTWISE mode,
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
            >>> grader = SummarizationGrader()
            >>> result = await grader.evaluate(
            ...     query="Summarize: Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and is widely considered one of the most influential scientists of all time.",
            ...     answer="Einstein was a famous physicist who created the theory of relativity."
            ... )
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
