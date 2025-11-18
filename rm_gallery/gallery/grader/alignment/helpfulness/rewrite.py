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

RUBRICS = """Preservation of Meaning: The rewritten text must accurately preserve the original meaning without introducing factual errors or distortions.
Enhancement of Clarity: The rewritten text should be clearer and easier to understand than the original, with improved logical flow.
Style and Format Adaptation: The rewritten text should appropriately adapt its style and format to better suit the target audience or purpose.
Elimination of Redundancy: Remove unnecessary repetition while retaining essential information.
Improved Readability: Enhance sentence structure, vocabulary, and overall readability appropriate to the target audience."""

REWRITE_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in evaluating rewritten text quality. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of a rewritten text.
You should critically and accurately assess the rewritten text based on the provided rubrics.
Focus on how well the rewriting improves clarity, style, or format while preserving the original meaning.

# Rubrics
{rubrics}

# Original Text
{query}

# Rewritten Text
{answer}

# Output Requirement
```json
{
    "score": "Numerical rewrite quality score (0.0-1.0)."
    "reason": "Explanation of how the score was determined."
}
```
""",
        ),
    ],
)

REWRITE_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in evaluating rewritten text quality. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with an original text and several rewritten versions. Your task is to select the single best rewritten version from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each rewritten version in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the rewritten versions.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.

# Rubrics
{rubrics}

# Original Text
{query}

# Rewritten Versions
{answer}

# Output Requirement
```json
{
    "rank": ["The rank score of the rewritten versions in the list."]
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)


class RewriteGrader(BaseHelpfulnessGrader):
    """Rewrite: Improves existing text by enhancing clarity, style, or format while preserving meaning."""

    _point_template = REWRITE_POINTWISE_TEMPLATE
    _list_template = REWRITE_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="",
            mode=mode,
            model=model,
            template=template,
            description="Rewrite: Improves existing text by enhancing clarity, style, or format while preserving meaning.",
            **kwargs,
        )

    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the rewrite quality of the response based on the query.

        Evaluates rewrite responses for their ability to improve existing text by
        enhancing clarity, style, or format while preserving meaning. The grader
        focuses on preservation of meaning, enhancement of clarity, and appropriate
        style adaptation.

        Args:
            query (str): The original text to be rewritten.
            answer (str | List[str]): The rewritten text to evaluate. For pointwise evaluation,
                this is a single string. For listwise evaluation, this is a list of rewritten versions.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical rewrite quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of rewritten versions by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise rewrite grader
            >>> grader = RewriteGrader(mode=GraderMode.POINTWISE)
            >>> result = await grader.evaluate(
            ...     query="Photosynthesis is the process by which plants convert light energy into chemical energy.",
            ...     answer="Plants use sunlight to make their own food."
            ... )
            >>> print(result.score, result.reason)

            >>> # Example for listwise rewrite grader
            >>> ranking_grader = RewriteGrader(mode=GraderMode.LISTWISE)
            >>> result = await ranking_grader.evaluate(
            ...     query="Photosynthesis is the process by which plants convert light energy into chemical energy.",
            ...     answer=[
            ...         "Plants use sunlight to make their own food.",
            ...         "The process where plants transform light into chemical energy is called photosynthesis."
            ...     ]
            ... )
            >>> print(result.rank, result.reason)
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
